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
def _clear_armed_state():
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

    def test_jailbreak_enabled_default(self):
        assert _jailbreak_enabled() is True


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
