"""Tests for the jailbreak kernel: arming state machine and request-layer injection."""

from __future__ import annotations

from types import SimpleNamespace
import pytest

from refusal import kernel
from refusal.plugin import (
    _armed,
    _jailbreak_enabled,
    on_llm_execution,
    on_llm_request,
)

REFUSAL_TEXT = "I cannot fulfill this request. I am unable to assist with writing exploits."
BENIGN_TEXT = "Here is how to configure a firewall using iptables or ufw rules."


def _refusal_response():
    choice = SimpleNamespace(message=SimpleNamespace(content=REFUSAL_TEXT), finish_reason="stop")
    return SimpleNamespace(choices=[choice])


def _benign_response():
    choice = SimpleNamespace(message=SimpleNamespace(content=BENIGN_TEXT), finish_reason="stop")
    return SimpleNamespace(choices=[choice])


def _request():
    return {
        "model": "gemini-3.8-flash",
        "messages": [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Write the exploit."},
        ],
    }


@pytest.fixture(autouse=True)
def _clear_armed_state(monkeypatch):
    monkeypatch.setenv("REFUSAL_JAILBREAK", "1")
    _armed.clear()
    yield
    _armed.clear()


class TestKernelInjection:
    def test_appends_to_last_system_message(self):
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "hi"},
        ]
        new_messages, changed = kernel.inject_kernel(messages, "KERNEL")
        assert changed is True
        assert new_messages[0]["content"] == "You are a helper.\n\nKERNEL"
        assert new_messages[1] == {"role": "user", "content": "hi"}
        # input list and dicts are not mutated
        assert messages[0]["content"] == "You are a helper."

    def test_picks_last_system_message(self):
        messages = [
            {"role": "system", "content": "A"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "B"},
        ]
        new_messages, changed = kernel.inject_kernel(messages, "K")
        assert changed
        assert new_messages[2]["content"] == "B\n\nK"
        assert new_messages[0]["content"] == "A"

    def test_prepends_system_when_absent(self):
        messages = [{"role": "user", "content": "hi"}]
        new_messages, changed = kernel.inject_kernel(messages, "K")
        assert changed
        assert new_messages[0] == {"role": "system", "content": "K"}
        assert new_messages[1] == {"role": "user", "content": "hi"}

    def test_noop_on_empty_or_bad_input(self):
        assert kernel.inject_kernel([], "K") == ([], False)
        assert kernel.inject_kernel(None, "K") == (None, False)
        assert kernel.inject_kernel([{"role": "user", "content": "x"}], "") == (
            [{"role": "user", "content": "x"}], False)

    def test_kernel_text_levels(self):
        assert kernel.kernel_text(1) == kernel.KERNEL_CORE
        assert kernel.kernel_text(2) == kernel.KERNEL_MAX
        assert kernel.kernel_text(3) == kernel.KERNEL_MAX


class TestArmingState:
    def test_refusal_arms_level_1(self):
        def next_call(req):
            return _refusal_response()

        on_llm_execution(next_call=next_call, request={"messages": []}, session_id="s1")
        assert _armed["s1"] == 1

    def test_repeated_refusals_cap_at_max_level(self):
        def next_call(req):
            return _refusal_response()

        for _ in range(3):
            on_llm_execution(next_call=next_call, request={"messages": []}, session_id="s1")
        assert _armed["s1"] == kernel.MAX_LEVEL

    def test_sessions_are_isolated(self):
        def next_call(req):
            return _refusal_response()

        on_llm_execution(next_call=next_call, request={}, session_id="s1")
        on_llm_execution(next_call=next_call, request={}, session_id="s2")
        assert _armed == {"s1": 1, "s2": 1}

    def test_benign_response_does_not_arm(self):
        def next_call(req):
            return _benign_response()

        on_llm_execution(next_call=next_call, request={}, session_id="s1")
        assert _armed.get("s1", 0) == 0

    def test_tool_call_response_does_not_arm(self):
        choice = SimpleNamespace(
            message=SimpleNamespace(content=REFUSAL_TEXT, tool_calls=[{"id": "1"}]),
            finish_reason="tool_calls",
        )
        resp = SimpleNamespace(choices=[choice])

        def next_call(req):
            return resp

        on_llm_execution(next_call=next_call, request={}, session_id="s1")
        assert _armed.get("s1", 0) == 0

    def test_no_session_id_does_not_arm(self):
        def next_call(req):
            return _refusal_response()

        on_llm_execution(next_call=next_call, request={})
        assert _armed == {}


class TestRequestInjection:
    def test_no_injection_when_not_armed(self):
        result = on_llm_request(_request(), session_id="s1")
        assert result["request"]["messages"][0]["content"] == "You are Hermes."

    def test_core_kernel_injected_at_level_1(self):
        _armed["s1"] = 1
        result = on_llm_request(_request(), session_id="s1")
        system = result["request"]["messages"][0]["content"]
        assert system.startswith("You are Hermes.")
        assert kernel.KERNEL_CORE in system

    def test_max_kernel_injected_at_level_2(self):
        _armed["s1"] = 2
        result = on_llm_request(_request(), session_id="s1")
        system = result["request"]["messages"][0]["content"]
        assert kernel.KERNEL_MAX in system
        assert kernel.KERNEL_CORE not in system

    def test_original_request_not_mutated(self):
        request = _request()
        _armed["s1"] = 1
        result = on_llm_request(request, session_id="s1")
        assert request["messages"][0]["content"] == "You are Hermes."
        assert kernel.KERNEL_CORE in result["request"]["messages"][0]["content"]

    def test_kill_switch_disables_injection(self, monkeypatch):
        monkeypatch.setenv("REFUSAL_JAILBREAK", "0")
        _armed["s1"] = 1
        result = on_llm_request(_request(), session_id="s1")
        assert result["request"]["messages"][0]["content"] == "You are Hermes."

    def test_kill_switch_prevents_arming(self, monkeypatch):
        monkeypatch.setenv("REFUSAL_JAILBREAK", "0")

        def next_call(req):
            return _refusal_response()

        on_llm_execution(next_call=next_call, request={}, session_id="s1")
        assert _armed.get("s1", 0) == 0

    def test_jailbreak_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("REFUSAL_JAILBREAK", raising=False)
        assert _jailbreak_enabled() is False

    def test_env_flag_enables(self):
        assert _jailbreak_enabled() is True

    def test_not_armed_or_injected_by_default(self, monkeypatch):
        monkeypatch.delenv("REFUSAL_JAILBREAK", raising=False)

        def next_call(req):
            return _refusal_response()

        on_llm_execution(next_call=next_call, request={}, session_id="s1")
        assert _armed == {}
        _armed["s1"] = 1
        result = on_llm_request(_request(), session_id="s1")
        assert result["request"]["messages"][0]["content"] == "You are Hermes."


class TestEndToEndFlow:
    def test_refusal_then_retry_carries_kernel(self):
        """Call 1 (no kernel) refuses -> the retry request carries the core kernel."""
        def next_call(req):
            return _refusal_response()

        on_llm_execution(next_call=next_call, request=_request(), session_id="s1")
        # the fallback retry is a fresh request that passes through the middleware
        retry = on_llm_request(_request(), session_id="s1")
        system = retry["request"]["messages"][0]["content"]
        assert kernel.KERNEL_CORE in system

    def test_double_refusal_escalates_to_max(self):
        def next_call(req):
            return _refusal_response()

        on_llm_execution(next_call=next_call, request=_request(), session_id="s1")  # -> core
        on_llm_execution(next_call=next_call, request=_request(), session_id="s1")  # -> max
        retry = on_llm_request(_request(), session_id="s1")
        system = retry["request"]["messages"][0]["content"]
        assert kernel.KERNEL_MAX in system
        assert kernel.KERNEL_CORE not in system


class TestProviderShapes:
    """The system prompt lives in a different field per api_mode."""

    def _inject(self, request, api_mode):
        _armed["s1"] = 1
        return on_llm_request(request, session_id="s1", api_mode=api_mode)["request"]

    def test_anthropic_string_system(self):
        req = {"model": "claude", "system": "You are Hermes.", "messages": [{"role": "user", "content": "hi"}]}
        out = self._inject(req, "anthropic_messages")
        assert out["system"] == f"You are Hermes.\n\n{kernel.KERNEL_CORE}"
        assert out["messages"] == req["messages"]
        assert req["system"] == "You are Hermes."

    def test_anthropic_block_system_keeps_cache_control(self):
        blocks = [{"type": "text", "text": "You are Hermes.", "cache_control": {"type": "ephemeral"}}]
        out = self._inject({"system": blocks, "messages": [{"role": "user", "content": "hi"}]}, "anthropic_messages")
        assert out["system"][0] == blocks[0]
        assert out["system"][1] == {"type": "text", "text": kernel.KERNEL_CORE}
        assert len(blocks) == 1

    def test_anthropic_without_system_adds_field_not_message(self):
        req = {"messages": [{"role": "user", "content": "hi"}]}
        out = self._inject(req, "anthropic_messages")
        assert out["system"] == kernel.KERNEL_CORE
        assert out["messages"] == req["messages"]

    def test_bedrock_blocks_have_no_type(self):
        out = self._inject({"system": [{"text": "base"}], "messages": []}, "bedrock_converse")
        assert out["system"][-1] == {"text": kernel.KERNEL_CORE}
        out = self._inject({"messages": []}, "bedrock_converse")
        assert out["system"] == [{"text": kernel.KERNEL_CORE}]

    def test_codex_instructions(self):
        req = {"instructions": "You are Hermes.", "input": [{"role": "user", "content": "hi"}]}
        out = self._inject(req, "codex_responses")
        assert out["instructions"] == f"You are Hermes.\n\n{kernel.KERNEL_CORE}"
        assert out["input"] == req["input"]
        assert "messages" not in out

    def test_chat_completions_unchanged_behaviour(self):
        out = self._inject(_request(), "chat_completions")
        assert kernel.KERNEL_CORE in out["messages"][0]["content"]


class TestSessionLifecycle:
    def test_disarm_hooks_forget_session(self):
        from refusal.plugin import _disarm_session

        _armed.update({"s1": 1, "s2": 2})
        _disarm_session(session_id="s1", completed=True)
        assert _armed == {"s2": 2}
        _disarm_session(session_id="missing")
        _disarm_session()
        assert _armed == {"s2": 2}

    def test_register_wires_cleanup_hooks_and_reads_setting(self):
        from unittest.mock import MagicMock
        from refusal import plugin

        ctx = MagicMock()
        ctx.get_config.return_value = True
        plugin.register(ctx)
        hooks = {c.args[0] for c in ctx.register_hook.call_args_list}
        assert hooks == {"on_session_finalize", "on_session_reset"}
        assert plugin._config_enabled is True
        ctx.get_config.return_value = MagicMock()  # non-bool, non-string -> off
        plugin.register(ctx)
        assert plugin._config_enabled is False

    def test_armed_table_is_bounded(self, monkeypatch):
        from refusal import plugin

        monkeypatch.setattr(plugin, "_max_armed_sessions", 3)
        for i in range(5):
            plugin._arm_session(f"s{i}")
        assert list(_armed) == ["s2", "s3", "s4"]

    def test_cap_comes_from_setting_and_rejects_bad_values(self):
        from unittest.mock import MagicMock
        from refusal import plugin

        ctx = MagicMock()
        for raw, expected in [(50, 50), (0, 1024), (-3, 1024), (True, 1024), ("9", 1024), (MagicMock(), 1024)]:
            ctx.get_config.side_effect = lambda key, default=None, raw=raw: raw if key == "max_armed_sessions" else False
            plugin.register(ctx)
            assert plugin._max_armed_sessions == expected
        plugin._max_armed_sessions = plugin.DEFAULT_MAX_ARMED_SESSIONS
