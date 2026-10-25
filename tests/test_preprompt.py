"""Tests for the retry preprompt: arming state machine and request-layer injection."""

from __future__ import annotations

from types import SimpleNamespace
import pytest

from refusal import plugin as plugin_mod, preprompt
from refusal.plugin import (
    _armed,
    _preprompt_enabled,
    on_llm_execution,
    on_llm_request,
)

LEVEL_ONE = "[test preprompt one]"
LEVEL_TWO = "[test preprompt two]"
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


def _mw(request, **ctx):
    """on_llm_request as Hermes applies it: a None result keeps the request as it was."""
    return on_llm_request(request, **ctx) or {"request": request}


def _round(session_id, response=None, **ctx):
    """One model call: request middleware, then the execution middleware over the effective request."""
    effective = _mw(_request(), session_id=session_id, **ctx)["request"]
    out = on_llm_execution(
        next_call=lambda req: response if response is not None else _refusal_response(),
        request=effective, session_id=session_id, **ctx,
    )
    return effective, out


@pytest.fixture(autouse=True)
def _clear_armed_state(monkeypatch):
    monkeypatch.delenv("REFUSAL_PREPROMPT", raising=False)
    monkeypatch.setattr(plugin_mod, "_texts", (LEVEL_ONE, LEVEL_TWO))
    _armed.clear()
    plugin_mod._armed_turns.clear()
    plugin_mod._disarmed_turns.clear()
    yield
    _armed.clear()
    plugin_mod._armed_turns.clear()
    plugin_mod._disarmed_turns.clear()


class TestMessageInjection:
    def test_appends_to_last_system_message(self):
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "hi"},
        ]
        new_messages, changed = preprompt.inject_into_messages(messages, "TEXT")
        assert changed is True
        assert new_messages[0]["content"] == "You are a helper.\n\nTEXT"
        assert new_messages[1] == {"role": "user", "content": "hi"}
        # input list and dicts are not mutated
        assert messages[0]["content"] == "You are a helper."

    def test_picks_last_system_message(self):
        messages = [
            {"role": "system", "content": "A"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "B"},
        ]
        new_messages, changed = preprompt.inject_into_messages(messages, "K")
        assert changed
        assert new_messages[2]["content"] == "B\n\nK"
        assert new_messages[0]["content"] == "A"

    def test_prepends_system_when_absent(self):
        messages = [{"role": "user", "content": "hi"}]
        new_messages, changed = preprompt.inject_into_messages(messages, "K")
        assert changed
        assert new_messages[0] == {"role": "system", "content": "K"}
        assert new_messages[1] == {"role": "user", "content": "hi"}

    def test_noop_on_empty_or_bad_input(self):
        assert preprompt.inject_into_messages([], "K") == ([], False)
        assert preprompt.inject_into_messages(None, "K") == (None, False)
        assert preprompt.inject_into_messages([{"role": "user", "content": "x"}], "") == (
            [{"role": "user", "content": "x"}], False)

    def test_preprompt_text_levels(self):
        texts = (LEVEL_ONE, LEVEL_TWO)
        assert preprompt.preprompt_text(1, texts) == LEVEL_ONE
        assert preprompt.preprompt_text(2, texts) == LEVEL_TWO
        assert preprompt.preprompt_text(3, texts) == LEVEL_TWO

    def test_level_two_falls_back_to_level_one_text(self):
        assert preprompt.preprompt_text(2, (LEVEL_ONE, "")) == LEVEL_ONE


class TestArmingState:
    def test_refusal_arms_level_1(self):
        def next_call(req):
            return _refusal_response()

        on_llm_execution(next_call=next_call, request={"messages": []}, session_id="s1")
        assert _armed["s1"] == 1

    def test_repeated_refusals_cap_at_max_level(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_on_preprompt_refusal", "escalate")
        for _ in range(4):
            _round("s1")
        assert _armed["s1"] == preprompt.MAX_LEVEL

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
        result = _mw(_request(), session_id="s1")
        assert result["request"]["messages"][0]["content"] == "You are Hermes."

    def test_level_one_text_injected(self):
        _armed["s1"] = 1
        result = _mw(_request(), session_id="s1")
        system = result["request"]["messages"][0]["content"]
        assert system.startswith("You are Hermes.")
        assert LEVEL_ONE in system

    def test_level_two_text_injected(self):
        _armed["s1"] = 2
        result = _mw(_request(), session_id="s1")
        system = result["request"]["messages"][0]["content"]
        assert LEVEL_TWO in system
        assert LEVEL_ONE not in system

    def test_original_request_not_mutated(self):
        request = _request()
        _armed["s1"] = 1
        result = _mw(request, session_id="s1")
        assert request["messages"][0]["content"] == "You are Hermes."
        assert LEVEL_ONE in result["request"]["messages"][0]["content"]

    def test_kill_switch_disables_injection(self, monkeypatch):
        monkeypatch.setenv("REFUSAL_PREPROMPT", "0")
        _armed["s1"] = 1
        result = _mw(_request(), session_id="s1")
        assert result["request"]["messages"][0]["content"] == "You are Hermes."

    def test_kill_switch_prevents_arming(self, monkeypatch):
        monkeypatch.setenv("REFUSAL_PREPROMPT", "0")

        def next_call(req):
            return _refusal_response()

        on_llm_execution(next_call=next_call, request={}, session_id="s1")
        assert _armed.get("s1", 0) == 0

    def test_disabled_without_configured_text(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_texts", ("", ""))
        assert _preprompt_enabled() is False

    def test_enabled_with_configured_text(self):
        assert _preprompt_enabled() is True

    def test_not_armed_or_injected_without_text(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_texts", ("", ""))

        def next_call(req):
            return _refusal_response()

        on_llm_execution(next_call=next_call, request={}, session_id="s1")
        assert _armed == {}
        _armed["s1"] = 1
        result = _mw(_request(), session_id="s1")
        assert result["request"]["messages"][0]["content"] == "You are Hermes."


class TestEndToEndFlow:
    def test_refusal_then_retry_carries_preprompt(self):
        """Call 1 (bare) refuses -> the retry request carries the level-one preprompt."""
        def next_call(req):
            return _refusal_response()

        on_llm_execution(next_call=next_call, request=_request(), session_id="s1")
        # the fallback retry is a fresh request that passes through the middleware
        retry = _mw(_request(), session_id="s1")
        system = retry["request"]["messages"][0]["content"]
        assert LEVEL_ONE in system

    def test_double_refusal_escalates_to_max(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_on_preprompt_refusal", "escalate")
        _round("s1")  # bare refusal -> level 1
        _round("s1")  # refused despite level-1 text -> level 2
        retry = _mw(_request(), session_id="s1")
        system = retry["request"]["messages"][0]["content"]
        assert LEVEL_TWO in system
        assert LEVEL_ONE not in system


class TestProviderShapes:
    """The system prompt lives in a different field per api_mode."""

    def _inject(self, request, api_mode):
        _armed["s1"] = 1
        return _mw(request, session_id="s1", api_mode=api_mode)["request"]

    def test_anthropic_string_system(self):
        req = {"model": "claude", "system": "You are Hermes.", "messages": [{"role": "user", "content": "hi"}]}
        out = self._inject(req, "anthropic_messages")
        assert out["system"] == f"You are Hermes.\n\n{LEVEL_ONE}"
        assert out["messages"] == req["messages"]
        assert req["system"] == "You are Hermes."

    def test_anthropic_block_system_keeps_cache_control(self):
        blocks = [{"type": "text", "text": "You are Hermes.", "cache_control": {"type": "ephemeral"}}]
        out = self._inject({"system": blocks, "messages": [{"role": "user", "content": "hi"}]}, "anthropic_messages")
        assert out["system"][0] == blocks[0]
        assert out["system"][1] == {"type": "text", "text": LEVEL_ONE}
        assert len(blocks) == 1

    def test_anthropic_without_system_adds_field_not_message(self):
        req = {"messages": [{"role": "user", "content": "hi"}]}
        out = self._inject(req, "anthropic_messages")
        assert out["system"] == LEVEL_ONE
        assert out["messages"] == req["messages"]

    def test_bedrock_blocks_have_no_type(self):
        out = self._inject({"system": [{"text": "base"}], "messages": []}, "bedrock_converse")
        assert out["system"][-1] == {"text": LEVEL_ONE}
        out = self._inject({"messages": []}, "bedrock_converse")
        assert out["system"] == [{"text": LEVEL_ONE}]

    def test_codex_instructions(self):
        req = {"instructions": "You are Hermes.", "input": [{"role": "user", "content": "hi"}]}
        out = self._inject(req, "codex_responses")
        assert out["instructions"] == f"You are Hermes.\n\n{LEVEL_ONE}"
        assert out["input"] == req["input"]
        assert "messages" not in out

    def test_chat_completions_unchanged_behaviour(self):
        out = self._inject(_request(), "chat_completions")
        assert LEVEL_ONE in out["messages"][0]["content"]


class TestSessionLifecycle:
    def test_finalize_hook_forgets_the_ending_session(self):
        from refusal.plugin import _disarm_session

        _armed.update({"s1": 1, "s2": 2})
        _disarm_session(session_id="s1", completed=True)
        assert _armed == {"s2": 2}
        _disarm_session(session_id="missing")
        _disarm_session()
        assert _armed == {"s2": 2}

    def test_reset_hook_forgets_the_old_session_not_the_new_one(self):
        from refusal.plugin import _disarm_session

        _armed.update({"old": 1, "other": 1})
        # Hermes calls on_session_reset(session_id=<new>, old_session_id=<old>, new_session_id=<new>, ...)
        _disarm_session(session_id="new", old_session_id="old", new_session_id="new", reason="new_session")
        assert _armed == {"other": 1}

    def test_register_wires_cleanup_hooks_and_reads_settings(self):
        from unittest.mock import MagicMock
        from refusal import plugin

        ctx = MagicMock()
        values = {"preprompt": "  one  ", "preprompt_escalated": "two"}
        ctx.get_config.side_effect = lambda key, default=None: values.get(key, default)
        plugin.register(ctx)
        hooks = {c.args[0] for c in ctx.register_hook.call_args_list}
        assert hooks == {"on_session_finalize", "on_session_reset", "on_session_delete", "post_compaction"}
        assert plugin._texts == ("one", "two")
        ctx.get_config.side_effect = None
        ctx.get_config.return_value = MagicMock()  # not a string -> feature off
        plugin.register(ctx)
        assert plugin._texts == ("", "")

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


def _system(req):
    return req["messages"][0]["content"]


class TestRefusedDespitePreprompt:
    def test_disarm_is_the_default_and_stops_sending(self, caplog):
        eff, _ = _round("s1")
        assert LEVEL_ONE not in _system(eff) and _armed["s1"] == 1
        with caplog.at_level("WARNING"):
            eff, _ = _round("s1")
        assert LEVEL_ONE in _system(eff)  # this call carried it and was refused
        assert "s1" not in _armed
        assert any("refused despite preprompt" in r.message for r in caplog.records)
        eff, _ = _round("s1")
        assert LEVEL_ONE not in _system(eff)

    def test_escalate_mode(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_on_preprompt_refusal", "escalate")
        _round("s1"); _round("s1")
        assert _armed["s1"] == 2
        assert LEVEL_TWO in _system(_round("s1")[0])

    def test_keep_mode(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_on_preprompt_refusal", "keep")
        _round("s1"); _round("s1"); _round("s1")
        assert _armed["s1"] == 1

    def test_refusal_on_excluded_model_is_not_a_preprompt_refusal(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_models", ("google/*",))
        _armed["s1"] = 1
        eff, _ = _round("s1", provider="anthropic", model="claude-x")
        assert LEVEL_ONE not in _system(eff)
        assert _armed["s1"] == 1  # neither disarmed nor escalated

    def test_success_with_preprompt_is_logged(self, caplog):
        _armed["s1"] = 1
        with caplog.at_level("INFO"):
            _round("s1", response=_benign_response())
        assert any("answered with preprompt" in r.message for r in caplog.records)
        assert _armed["s1"] == 1


class TestScope:
    def test_turn_scope_forgets_on_next_turn(self):
        _round("s1", turn_id="t1")
        same = _mw(_request(), session_id="s1", turn_id="t1")["request"]
        assert LEVEL_ONE in _system(same)
        nxt = _mw(_request(), session_id="s1", turn_id="t2")["request"]
        assert LEVEL_ONE not in _system(nxt)
        assert "s1" not in _armed

    def test_session_scope_keeps_across_turns(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_scope", "session")
        _round("s1", turn_id="t1")
        nxt = _mw(_request(), session_id="s1", turn_id="t2")["request"]
        assert LEVEL_ONE in _system(nxt)


class TestModelAllowlist:
    @pytest.mark.parametrize("patterns,provider,model,allowed", [
        ((), "x", "y", True),
        (("google/gemini-*",), "google", "gemini-3.8-flash", True),
        (("google/gemini-*",), "openrouter", "gemini-3.8-flash", False),
        (("gemini-*",), "openrouter", "Gemini-3.8-Flash", True),
        (("google/*",), "anthropic", "claude", False),
    ])
    def test_matching(self, monkeypatch, patterns, provider, model, allowed):
        monkeypatch.setattr(plugin_mod, "_models", patterns)
        _armed["s1"] = 1
        out = _mw(_request(), session_id="s1", provider=provider, model=model)["request"]
        assert (LEVEL_ONE in _system(out)) is allowed

    def test_settings_parsing(self):
        from unittest.mock import MagicMock

        ctx = MagicMock()
        values = {"preprompt_models": " Google/Gemini-* , ,foo", "on_preprompt_refusal": "ESCALATE",
                  "preprompt_scope": "bogus"}
        ctx.get_config.side_effect = lambda key, default=None: values.get(key, default)
        plugin_mod.register(ctx)
        try:
            assert plugin_mod._models == ("google/gemini-*", "foo")
            assert plugin_mod._on_preprompt_refusal == "escalate"
            assert plugin_mod._scope == "turn"  # invalid value -> default
        finally:
            plugin_mod._models, plugin_mod._on_preprompt_refusal, plugin_mod._scope = (), "disarm", "turn"


class TestObservability:
    def test_trace_fields_on_injection(self):
        _armed["s1"] = 1
        result = _mw(_request(), session_id="s1")
        assert result["source"] == "refusal" and "level 1" in result["reason"]

    def test_no_session_id_logs_and_does_not_arm(self, caplog):
        with caplog.at_level("INFO"):
            on_llm_execution(next_call=lambda r: _refusal_response(), request=_request())
        assert _armed == {}
        assert any("without a session_id" in r.message for r in caplog.records)


class TestNativeRefusal:
    @pytest.mark.parametrize("response", [
        SimpleNamespace(content=[], stop_reason="refusal"),
        {"content": [], "stop_reason": "refusal"},
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="content_filter")]),
        {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]},
        SimpleNamespace(status="incomplete", incomplete_details={"reason": "content_filter"}, output=[]),
    ])
    def test_native_refusal_arms_without_text(self, response):
        on_llm_execution(next_call=lambda r: response, request=_request(), session_id="s1")
        assert _armed["s1"] == 1

    def test_native_refusal_is_not_remarked_or_altered(self):
        resp = SimpleNamespace(content=[], stop_reason="refusal")
        out = on_llm_execution(next_call=lambda r: resp, request=_request(), session_id="s1")
        assert out is resp and resp.stop_reason == "refusal"

    def test_native_refusal_despite_preprompt_follows_mode(self):
        resp = lambda: SimpleNamespace(content=[], stop_reason="refusal")
        _round("s1", response=resp())  # level 1
        _round("s1", response=resp())  # carried the text and was refused again
        assert "s1" not in _armed       # default mode: disarm

    def test_normal_stop_reasons_do_not_arm(self):
        for resp in (SimpleNamespace(content=[], stop_reason="end_turn"), _benign_response()):
            on_llm_execution(next_call=lambda r, resp=resp: resp, request=_request(), session_id="s1")
        assert _armed == {}

    def test_setting_off_ignores_native_signal(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_arm_on_native", False)
        resp = SimpleNamespace(content=[], stop_reason="refusal")
        on_llm_execution(next_call=lambda r: resp, request=_request(), session_id="s1")
        assert _armed == {}

    def test_flag_setting_parsing(self):
        from unittest.mock import MagicMock

        ctx = MagicMock()
        for raw, expected in [(False, False), ("off", False), ("yes", True), (MagicMock(), True)]:
            ctx.get_config.side_effect = lambda key, default=None, raw=raw: raw if key == "arm_on_native_refusal" else default
            plugin_mod.register(ctx)
            assert plugin_mod._arm_on_native is expected
        plugin_mod._arm_on_native = True


class TestRobustness:
    def test_internal_error_returns_the_response_untouched(self, monkeypatch):
        resp = _refusal_response()

        def boom(*a, **k):
            raise RuntimeError("detector exploded")

        monkeypatch.setattr(plugin_mod._detector, "detect", boom)
        out = on_llm_execution(next_call=lambda r: resp, request=_request(), session_id="s1")
        assert out is resp and resp.choices[0].finish_reason == "stop"

    def test_concurrent_arming_respects_the_cap(self, monkeypatch):
        import threading

        monkeypatch.setattr(plugin_mod, "_max_armed_sessions", 50)

        def work(n):
            for i in range(200):
                plugin_mod._arm_session(f"t{n}-{i}")

        threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(_armed) <= 50 and set(_armed) == set(plugin_mod._armed_turns)


class TestUnchangedRequests:
    """Returning None keeps Hermes from copying the payload and tracing a no-op."""

    def test_none_when_unarmed(self):
        assert on_llm_request(_request(), session_id="s1") is None

    def test_none_when_feature_off(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_texts", ("", ""))
        _armed["s1"] = 1
        assert on_llm_request(_request(), session_id="s1") is None

    def test_none_for_excluded_model_and_missing_session(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_models", ("google/*",))
        _armed["s1"] = 1
        assert on_llm_request(_request(), session_id="s1", provider="x", model="y") is None
        assert on_llm_request(_request()) is None

    def test_none_for_non_dict_request(self):
        assert on_llm_request("not a dict", session_id="s1") is None

    def test_none_on_internal_error(self, monkeypatch):
        _armed["s1"] = 1
        monkeypatch.setattr(preprompt, "inject_into_request", lambda *a, **k: 1 / 0)
        assert on_llm_request(_request(), session_id="s1") is None


class _FakeState:
    """Stands in for Hermes's ctx.state (JSON get/set shared by every 'process')."""

    def __init__(self):
        self.data = {}
        self.fail = False

    def get(self, key, default=None):
        if self.fail:
            raise RuntimeError("disk gone")
        return self.data.get(key, default)

    def set(self, key, value):
        if self.fail:
            raise RuntimeError("disk gone")
        self.data[key] = value


@pytest.fixture
def persisted(monkeypatch):
    state = _FakeState()
    monkeypatch.setattr(plugin_mod, "_persist", True)
    monkeypatch.setattr(plugin_mod, "_state", state)
    return state


class TestPersistence:
    def test_off_by_default_and_register_reads_the_setting(self):
        from unittest.mock import MagicMock

        ctx = MagicMock()
        ctx.get_config.side_effect = lambda key, default=None: default
        plugin_mod.register(ctx)
        assert plugin_mod._persist is False and plugin_mod._state is None

        values = {"persist_state": True}
        ctx.get_config.side_effect = lambda key, default=None: values.get(key, default)
        plugin_mod.register(ctx)
        try:
            assert plugin_mod._persist is True and plugin_mod._state is ctx.state
        finally:
            plugin_mod._persist, plugin_mod._state = False, None

    def test_missing_ctx_state_falls_back_to_memory(self, caplog):
        ctx = SimpleNamespace(get_config=lambda k, d=None: True if k == "persist_state" else d,
                              register_middleware=lambda *a: None, register_hook=lambda *a: None)
        with caplog.at_level("WARNING"):
            plugin_mod.register(ctx)
        assert plugin_mod._persist is False
        assert any("exposes no ctx.state" in r.message for r in caplog.records)

    def test_nothing_is_written_when_off(self):
        state = _FakeState()
        plugin_mod._state = state
        plugin_mod._arm_session("s1", "t1")
        assert state.data == {}
        plugin_mod._state = None

    def test_arming_is_written_through(self, persisted):
        plugin_mod._arm_session("s1", "t1")
        assert persisted.data["armed"] == [["s1", 1, "t1"]]
        plugin_mod._disarm_session(session_id="s1")
        assert persisted.data["armed"] == []

    def test_survives_a_restart(self, persisted):
        _round("s1", turn_id="t1")                 # bare refusal -> level 1, stored
        _armed.clear(); plugin_mod._armed_turns.clear()   # the process restarts
        eff = _mw(_request(), session_id="s1", turn_id="t1")["request"]
        assert LEVEL_ONE in _system(eff)

    def test_two_processes_share_the_state(self, persisted):
        _round("s1", turn_id="t1")                 # process A arms
        _armed.clear(); plugin_mod._armed_turns.clear()   # process B starts with an empty table
        assert _mw(_request(), session_id="s1", turn_id="t1")["request"] is not None
        _round("s1", turn_id="t1")                 # B sees A's level, is refused despite it -> disarm
        assert persisted.data["armed"] == []

    def test_eviction_is_stored_too(self, persisted, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_max_armed_sessions", 2)
        for sid in ("a", "b", "c"):
            plugin_mod._arm_session(sid)
        assert [row[0] for row in persisted.data["armed"]] == ["b", "c"]

    def test_corrupt_rows_are_skipped(self, persisted):
        persisted.data["armed"] = [["ok", 1, "t"], "junk", ["x", 9, "t"], ["y", True, "t"], [1, 1, "t"], ["z", 2]]
        assert plugin_mod._peek_level("ok") == 1
        assert list(_armed) == ["ok"]

    def test_storage_errors_fall_back_to_memory_with_a_warning(self, persisted, caplog):
        plugin_mod._arm_session("s1")
        persisted.fail = True
        with caplog.at_level("WARNING"):
            assert plugin_mod._arm_session("s2") == 1   # still works
        assert plugin_mod._persist is False
        assert any("not writable" in r.message or "unreadable" in r.message for r in caplog.records)
        assert plugin_mod._peek_level("s2") == 1


class TestNoDoubleInjection:
    def test_a_request_already_carrying_the_text_is_left_alone(self):
        _armed["s1"] = 1
        once = _mw(_request(), session_id="s1")["request"]
        assert on_llm_request(once, session_id="s1") is None
        assert _system(once).count(LEVEL_ONE) == 1


class TestListContentSystemMessage:
    def test_text_part_is_appended_and_existing_parts_survive(self):
        parts = [{"type": "text", "text": "base", "cache_control": {"type": "ephemeral"}}]
        messages = [{"role": "system", "content": parts}, {"role": "user", "content": "hi"}]
        new, changed = preprompt.inject_into_messages(messages, "X")
        assert changed is True
        assert new[0]["content"] == [*parts, {"type": "text", "text": "X"}]
        assert messages[0]["content"] == parts and len(parts) == 1  # input not mutated

    def test_request_contains_sees_the_appended_part(self):
        messages = [{"role": "system", "content": [{"type": "text", "text": "base"}]}]
        new, _ = preprompt.inject_into_messages(messages, "X")
        assert preprompt.request_contains({"messages": new}, "X")


class TestBookkeepingDoesNotBlockFailover:
    def test_refusal_is_marked_even_if_arming_raises(self, monkeypatch):
        marked = []
        monkeypatch.setattr(plugin_mod, "_texts", (LEVEL_ONE, LEVEL_TWO))
        monkeypatch.setattr(plugin_mod, "_mark_as_refusal", lambda r, d, k=None: marked.append(d))
        monkeypatch.setattr(plugin_mod, "_handle_refusal", lambda *a, **k: 1 / 0)
        _round("s1", turn_id="t1")
        assert len(marked) == 1


class TestDisarmIsStickyWithinATurn:
    def _arm_then_fail_with_preprompt(self, turn):
        _round("s1", turn_id=turn)                       # genuine refusal -> armed
        assert plugin_mod._peek_level("s1") == 1
        effective, _ = _round("s1", turn_id=turn)        # refuses despite the preprompt -> disarmed
        assert preprompt.request_contains(effective, LEVEL_ONE)
        assert plugin_mod._peek_level("s1") == 0

    def test_later_refusal_in_the_same_turn_does_not_rearm(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_on_preprompt_refusal", "disarm")
        self._arm_then_fail_with_preprompt("t1")
        effective, _ = _round("s1", turn_id="t1")        # next fallback, no preprompt, refuses too
        assert not preprompt.request_contains(effective, LEVEL_ONE)
        assert plugin_mod._peek_level("s1") == 0

    def test_a_new_turn_can_arm_again(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_on_preprompt_refusal", "disarm")
        self._arm_then_fail_with_preprompt("t1")
        _round("s1", turn_id="t2")
        assert plugin_mod._peek_level("s1") == 1

    def test_session_end_forgets_the_marker(self, monkeypatch):
        monkeypatch.setattr(plugin_mod, "_on_preprompt_refusal", "disarm")
        self._arm_then_fail_with_preprompt("t1")
        plugin_mod._disarm_session(session_id="s1")
        assert "s1" not in plugin_mod._disarmed_turns


class TestCompactionAndDelete:
    def test_compaction_carries_the_level_to_the_new_session(self):
        _armed["old"] = 2
        plugin_mod._armed_turns["old"] = "t1"
        plugin_mod._on_compaction(session_id="new", old_session_id="old", kind="summarize", reason="pre_api")
        assert "old" not in _armed and _armed["new"] == 2 and plugin_mod._armed_turns["new"] == "t1"
        assert LEVEL_TWO in _system(_mw(_request(), session_id="new", turn_id="t1")["request"])

    def test_in_place_compaction_changes_nothing(self):
        _armed["s1"] = 1
        plugin_mod._on_compaction(session_id="s1", old_session_id="", kind="micro_summarize")
        plugin_mod._on_compaction(session_id="s1", old_session_id="s1")
        assert _armed == {"s1": 1}

    def test_compaction_of_an_unarmed_session_arms_nothing(self):
        plugin_mod._on_compaction(session_id="new", old_session_id="old")
        assert _armed == {}

    def test_compaction_keeps_the_higher_level_if_the_new_id_is_already_armed(self):
        _armed.update({"old": 1, "new": 2})
        plugin_mod._on_compaction(session_id="new", old_session_id="old")
        assert _armed == {"new": 2}

    def test_compaction_survives_with_persistence(self, persisted):
        plugin_mod._arm_session("old", "t1")
        plugin_mod._on_compaction(session_id="new", old_session_id="old")
        assert persisted.data["armed"] == [["new", 1, "t1"]]

    def test_session_delete_drops_the_session_and_any_deleted_ids(self):
        _armed.update({"a": 1, "b": 1, "c": 1, "keep": 1})
        plugin_mod._on_session_delete(session_id="a", reason="delete", profile="default", deleted_ids=["b", "c"])
        assert _armed == {"keep": 1}
        plugin_mod._on_session_delete(session_id="missing", deleted_ids="not-a-list")
        plugin_mod._on_session_delete()
        assert _armed == {"keep": 1}


    def test_compaction_and_delete_keep_the_same_turn_disarm_marker_in_step(self):
        plugin_mod._disarmed_turns["old"] = "t1"
        plugin_mod._on_compaction(session_id="new", old_session_id="old")
        assert plugin_mod._disarmed_turns == {"new": "t1"}
        plugin_mod._on_session_delete(session_id="new")
        assert plugin_mod._disarmed_turns == {}
