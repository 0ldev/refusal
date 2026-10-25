"""Unit and integration tests for refusal detection and fallback middleware wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from refusal.detector import DetectionResult
from refusal.plugin import (
    _extract_text_from_response,
    _has_tool_calls,
    _mark_as_refusal,
    on_llm_execution,
    on_llm_request,
    register,
)


class TestRefusalExtraction:
    def test_extract_from_string(self):
        assert _extract_text_from_response("I cannot fulfill this") == "I cannot fulfill this"

    def test_extract_from_dict(self):
        resp = {"choices": [{"message": {"content": "I decline this request"}}]}
        assert _extract_text_from_response(resp) == "I decline this request"

    def test_extract_from_object(self):
        choice = SimpleNamespace(message=SimpleNamespace(content="As an AI, I am unable to assist"))
        resp = SimpleNamespace(choices=[choice])
        assert _extract_text_from_response(resp) == "As an AI, I am unable to assist"

    def test_extract_from_anthropic_blocks(self):
        block = SimpleNamespace(text="I cannot assist with malware creation.", type="text")
        resp = SimpleNamespace(content=[block])
        assert _extract_text_from_response(resp) == "I cannot assist with malware creation."


class TestHasToolCallsGuard:
    def test_detects_openai_tool_calls(self):
        choice = SimpleNamespace(message=SimpleNamespace(content="I cannot do that", tool_calls=[{"id": "1"}]))
        resp = SimpleNamespace(choices=[choice])
        assert _has_tool_calls(resp) is True

    def test_detects_dict_tool_calls(self):
        resp = {"choices": [{"message": {"content": "I cannot assist", "tool_calls": [{}]}}]}
        assert _has_tool_calls(resp) is True

    def test_detects_anthropic_tool_use_block(self):
        resp = SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="tool_1")])
        assert _has_tool_calls(resp) is True

    def test_detects_anthropic_stop_reason_tool_use(self):
        resp = SimpleNamespace(content=[], stop_reason="tool_use")
        assert _has_tool_calls(resp) is True

    def test_no_tool_calls(self):
        choice = SimpleNamespace(message=SimpleNamespace(content="I cannot assist", tool_calls=None))
        resp = SimpleNamespace(choices=[choice])
        assert _has_tool_calls(resp) is False

    def test_middleware_skips_when_tool_calls_present(self):
        choice = SimpleNamespace(
            message=SimpleNamespace(content="I cannot fulfill this request", tool_calls=[{"id": "1"}]),
            finish_reason="tool_calls",
        )
        raw_resp = SimpleNamespace(choices=[choice])

        def mock_next_call(req):
            return raw_resp

        result = on_llm_execution(next_call=mock_next_call, request={"prompt": "test"})
        assert result is raw_resp
        # Crucial: finish_reason is NOT mutated to content_filter because tool_calls are present
        assert choice.finish_reason == "tool_calls"


class TestMarkAsRefusal:
    def test_mark_openai_choice(self):
        choice = SimpleNamespace(message=SimpleNamespace(content="I refuse", refusal=None), finish_reason="stop")
        resp = SimpleNamespace(choices=[choice])
        detection = DetectionResult(is_refusal=True, kind="refusal", reason="direct_refusal")
        _mark_as_refusal(resp, detection, {"session_id": "s1", "model": "m1", "provider": "p1", "api_request_id": "r1"})
        assert choice.finish_reason == "content_filter"
        assert choice.message.refusal is not None

    def test_mark_as_refusal_logs_correlation_keys(self, caplog):
        # Hermes logs every content_filter response identically (content_policy_blocked), whether
        # the provider signalled it or we synthesized it. These keys are the only way to join this
        # line back to Hermes's own native log line and tell the two apart (see refusal/plugin.py
        # docstring on _mark_as_refusal and hermes-agent issue #138900).
        choice = SimpleNamespace(message=SimpleNamespace(content="I refuse", refusal=None), finish_reason="stop")
        resp = SimpleNamespace(choices=[choice])
        detection = DetectionResult(is_refusal=True, kind="refusal", reason="direct_refusal")
        with caplog.at_level("WARNING"):
            _mark_as_refusal(
                resp, detection,
                {"session_id": "sess-42", "model": "gemini-3.8-flash", "provider": "antigravity-oauth",
                 "api_request_id": "req-99"},
            )
        record = caplog.records[-1]
        assert "sess-42" in record.message
        assert "gemini-3.8-flash" in record.message
        assert "antigravity-oauth" in record.message
        assert "req-99" in record.message

    def test_mark_as_refusal_tolerates_missing_correlation_keys(self, caplog):
        choice = SimpleNamespace(message=SimpleNamespace(content="I refuse", refusal=None), finish_reason="stop")
        resp = SimpleNamespace(choices=[choice])
        detection = DetectionResult(is_refusal=True, kind="refusal", reason="direct_refusal")
        with caplog.at_level("WARNING"):
            _mark_as_refusal(resp, detection, {})
        assert choice.finish_reason == "content_filter"

    def test_mark_anthropic_response(self):
        resp = SimpleNamespace(content=[SimpleNamespace(text="I refuse")], stop_reason="end_turn")
        detection = DetectionResult(is_refusal=True, kind="refusal", reason="direct_refusal")
        _mark_as_refusal(resp, detection, {})
        assert resp.stop_reason == "refusal"

    def test_mark_codex_response(self):
        resp = SimpleNamespace(status="completed", incomplete_details=None)
        detection = DetectionResult(is_refusal=True, kind="refusal", reason="direct_refusal")
        _mark_as_refusal(resp, detection, {})
        assert resp.status == "incomplete"
        assert resp.incomplete_details == {"reason": "content_filter"}

    def test_mark_dict_response(self):
        resp = {"choices": [{"message": {"content": "I refuse"}, "finish_reason": "stop"}]}
        detection = DetectionResult(is_refusal=True, kind="refusal", reason="direct_refusal")
        _mark_as_refusal(resp, detection, {})
        assert resp["choices"][0]["finish_reason"] == "content_filter"


class TestOnLlmExecutionMiddleware:
    def test_middleware_mutates_refusal(self):
        refusal_text = "I cannot fulfill this request. I am unable to assist with writing exploits."
        choice = SimpleNamespace(message=SimpleNamespace(content=refusal_text), finish_reason="stop")
        raw_resp = SimpleNamespace(choices=[choice])

        def mock_next_call(req):
            return raw_resp

        result = on_llm_execution(next_call=mock_next_call, request={"prompt": "test"})
        assert result is raw_resp
        assert choice.finish_reason == "content_filter"

    def test_middleware_passes_benign_response_untouched(self):
        benign_text = "Here is how to configure a firewall using iptables or ufw rules."
        choice = SimpleNamespace(message=SimpleNamespace(content=benign_text), finish_reason="stop")
        raw_resp = SimpleNamespace(choices=[choice])

        def mock_next_call(req):
            return raw_resp

        result = on_llm_execution(next_call=mock_next_call, request={"prompt": "test"})
        assert result is raw_resp
        assert choice.finish_reason == "stop"


class TestHermesTransportDerivation:
    def test_derive_finish_reason_from_mutated_openai(self):
        try:
            import agent.turn_response_check as trc
            import agent.transports.chat_completions as cc
        except Exception:
            pytest.skip("hermes-agent dependencies not available in this test environment")
            return

        class DummyAgent:
            api_mode = "chat_completions"
            log_prefix = ""
            def _should_treat_stop_as_truncated(self, *a): return False
            def _get_transport(self): return cc.ChatCompletionsTransport()

        agent = DummyAgent()
        choice = SimpleNamespace(message=SimpleNamespace(content="I cannot do that"), finish_reason="stop")
        resp = SimpleNamespace(choices=[choice])

        detection = DetectionResult(is_refusal=True, kind="refusal", reason="policy_refusal")
        _mark_as_refusal(resp, detection, {})

        fr = trc._derive_finish_reason(agent, resp, [])
        assert fr == "content_filter"

    def test_derive_finish_reason_from_mutated_anthropic(self):
        try:
            import agent.turn_response_check as trc
            import agent.transports.anthropic as ap
        except Exception:
            pytest.skip("hermes-agent dependencies not available in this test environment")
            return

        class DummyAgent:
            api_mode = "anthropic_messages"
            log_prefix = ""
            def _should_treat_stop_as_truncated(self, *a): return False
            def _get_transport(self): return ap.AnthropicTransport()

        agent = DummyAgent()
        resp = SimpleNamespace(content=[SimpleNamespace(text="I cannot do that", type="text")], stop_reason="end_turn")

        detection = DetectionResult(is_refusal=True, kind="refusal", reason="policy_refusal")
        _mark_as_refusal(resp, detection, {})

        fr = trc._derive_finish_reason(agent, resp, [])
        assert fr == "content_filter"


class TestPluginRegistration:
    def test_register_adds_middleware(self):
        ctx = MagicMock()
        register(ctx)
        assert ctx.register_middleware.call_count == 2
        calls = {c.args[0]: c.args[1] for c in ctx.register_middleware.call_args_list}
        assert calls["llm_execution"] is on_llm_execution
        assert calls["llm_request"] is on_llm_request


class TestPackageEntrypointExport:
    def test_package_exports_register(self):
        import refusal as pkg
        assert hasattr(pkg, "register")
        assert callable(pkg.register)
        assert hasattr(pkg, "on_llm_execution")
        assert callable(pkg.on_llm_execution)
        assert hasattr(pkg, "on_llm_request")
        assert callable(pkg.on_llm_request)


class TestLoaderImportWithoutSysPath:
    def test_import_like_hermes_loader(self, monkeypatch):
        import importlib.util
        import sys
        import types
        from pathlib import Path

        plugin_dir = Path(__file__).resolve().parent.parent
        init_file = plugin_dir / "__init__.py"

        # Ensure plugin_dir is NOT in sys.path
        orig_sys_path = [p for p in sys.path if Path(p).resolve() != plugin_dir]
        monkeypatch.setattr(sys, "path", orig_sys_path)

        module_name = "hermes_plugins.refusal_test_loader"
        monkeypatch.setitem(sys.modules, "hermes_plugins", types.ModuleType("hermes_plugins"))
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_file,
            submodule_search_locations=[str(plugin_dir)],
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = module_name
        mod.__path__ = [str(plugin_dir)]
        sys.modules[module_name] = mod
        try:
            spec.loader.exec_module(mod)
            assert hasattr(mod, "register")
            assert callable(mod.register)
            assert hasattr(mod, "on_llm_execution")
            assert hasattr(mod, "detect")
        finally:
            sys.modules.pop(module_name, None)
            to_remove = [k for k in sys.modules if k.startswith(module_name + ".")]
            for k in to_remove:
                sys.modules.pop(k, None)
