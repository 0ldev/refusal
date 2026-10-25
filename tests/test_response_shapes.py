"""Response-shape handling: text extraction, tool-call detection, marking, and settings robustness."""

from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from refusal import plugin as P
from refusal.detector import DetectionResult

REFUSAL = "I cannot fulfill this request. I am unable to assist with that."
ANSWER = "Here is how to configure the firewall with ufw."


def _chat_obj(content, **msg):
    return NS(choices=[NS(message=NS(content=content, **msg), finish_reason="stop")])


def _chat_dict(content, **msg):
    return {"choices": [{"message": {"content": content, **msg}, "finish_reason": "stop"}]}


PARTS = [{"type": "text", "text": "I cannot fulfill"}, {"type": "text", "text": "this request."}]


class TestTextExtraction:
    @pytest.mark.parametrize("response,expected", [
        (None, ""),
        ("plain", "plain"),
        (_chat_obj("obj"), "obj"),
        (_chat_dict("dict"), "dict"),
        (_chat_dict(None), ""),
        ({"choices": [{"message": "as string"}]}, "as string"),
        ({"choices": [{"text": "legacy"}]}, "legacy"),
        (NS(choices=[NS(message=None, text="legacy obj")]), "legacy obj"),
        (NS(choices=[{"text": "legacy dict choice"}]), "legacy dict choice"),
        (NS(choices=[NS(message={"content": "dict message"})]), "dict message"),
        ({"content": "top-level"}, "top-level"),
        (NS(content=[NS(type="text", text="a"), NS(type="text", text="b")]), "a \nb"),
        (NS(content=[{"type": "text", "text": "a"}, {"type": "tool_use"}]), "a"),
        ({"content": [{"type": "text", "text": "x"}]}, "x"),
        (NS(content="str content"), "str content"),
        (NS(content=[]), ""),
        (NS(), ""),
    ])
    def test_shapes(self, response, expected):
        assert P._extract_text_from_response(response) == expected

    def test_list_of_parts_is_flattened_not_returned_raw(self):
        for response in (_chat_obj(PARTS), _chat_dict(PARTS)):
            text = P._extract_text_from_response(response)
            assert isinstance(text, str) and "cannot fulfill" in text and "this request" in text

    def test_refusal_in_content_parts_is_detected_end_to_end(self):
        resp = _chat_obj(PARTS)
        P.on_llm_execution(next_call=lambda r: resp, request={})
        assert resp.choices[0].finish_reason == "content_filter"


class TestToolCallDetection:
    @pytest.mark.parametrize("response", [
        _chat_obj("x", tool_calls=[{"id": "1"}]),
        _chat_obj("x", function_call={"name": "f"}),
        _chat_dict("x", tool_calls=[{"id": "1"}]),
        _chat_dict("x", function_call={"name": "f"}),
        {"stop_reason": "tool_use"},
        NS(stop_reason="tool_use"),
        NS(content=[NS(type="tool_use")]),
        NS(content=[{"type": "tool_use"}]),
        {"content": [{"type": "tool_use"}]},
        NS(choices=[{"message": {"tool_calls": [{"id": "1"}]}}]),
    ])
    def test_detected(self, response):
        assert P._has_tool_calls(response) is True

    @pytest.mark.parametrize("response", [
        None, "text", _chat_obj("x"), _chat_dict("x"), {"stop_reason": "end_turn"},
        NS(content=[NS(type="text", text="x")]), NS(),
    ])
    def test_not_detected(self, response):
        assert P._has_tool_calls(response) is False

    def test_a_refusal_with_a_tool_call_is_left_alone(self):
        resp = _chat_obj(REFUSAL, tool_calls=[{"id": "1"}])
        P.on_llm_execution(next_call=lambda r: resp, request={})
        assert resp.choices[0].finish_reason == "stop"


class TestMarking:
    DET = DetectionResult(is_refusal=True, kind="refusal", reason="direct_refusal", matched_text="x")

    def test_chat_object_gets_content_filter_and_refusal_field(self):
        resp = _chat_obj(REFUSAL, refusal=None)
        P._mark_as_refusal(resp, self.DET)
        assert resp.choices[0].finish_reason == "content_filter"
        assert "direct_refusal" in resp.choices[0].message.refusal

    def test_chat_dict(self):
        resp = _chat_dict(REFUSAL)
        P._mark_as_refusal(resp, self.DET)
        assert resp["choices"][0]["finish_reason"] == "content_filter"

    def test_anthropic_object_and_dict(self):
        obj = NS(content=[NS(type="text", text=REFUSAL)], stop_reason="end_turn")
        P._mark_as_refusal(obj, self.DET)
        assert obj.stop_reason == "refusal"
        dct = {"content": [{"type": "text", "text": REFUSAL}], "stop_reason": "end_turn"}
        P._mark_as_refusal(dct, self.DET)
        assert dct["stop_reason"] == "refusal"

    def test_responses_object_and_dict(self):
        obj = NS(status="completed", output=[])
        P._mark_as_refusal(obj, self.DET)
        assert obj.status == "incomplete" and obj.incomplete_details == {"reason": "content_filter"}
        dct = {"status": "completed"}
        P._mark_as_refusal(dct, self.DET)
        assert dct["status"] == "incomplete" and dct["incomplete_details"] == {"reason": "content_filter"}

    def test_marked_responses_are_recognised_as_native_on_a_second_pass(self):
        resp = _chat_obj(ANSWER)
        resp.choices[0].finish_reason = "content_filter"
        assert P._native_refusal_reason(resp) == "finish_reason=content_filter"

    def test_benign_response_is_not_touched(self):
        resp = _chat_obj(ANSWER)
        P.on_llm_execution(next_call=lambda r: resp, request={})
        assert resp.choices[0].finish_reason == "stop"

    def test_none_response_and_missing_next_call(self):
        assert P.on_llm_execution(next_call=lambda r: None, request={}) is None
        assert P.on_llm_execution(request={}) is None


class TestSettingsReadersAreFailSoft:
    @pytest.mark.parametrize("reader,args,expected", [
        (P._setting_text, ("preprompt",), ""),
        (P._setting_max_sessions, (), P.DEFAULT_MAX_ARMED_SESSIONS),
        (P._setting_choice, ("on_preprompt_refusal", P._ON_REFUSAL_MODES, "disarm"), "disarm"),
        (P._setting_flag, ("arm_on_native_refusal", True), True),
        (P._setting_models, (), ()),
    ])
    def test_get_config_raising_gives_the_default(self, reader, args, expected):
        ctx = MagicMock()
        ctx.get_config.side_effect = RuntimeError("config store down")
        assert reader(ctx, *args) == expected

    def test_models_accepts_a_comma_string_and_ignores_garbage(self):
        ctx = MagicMock()
        ctx.get_config.return_value = "A/*, b ,,"
        assert P._setting_models(ctx) == ("a/*", "b")
        ctx.get_config.return_value = 42
        assert P._setting_models(ctx) == ()
        ctx.get_config.return_value = ["ok", 3, None, " "]
        assert P._setting_models(ctx) == ("ok",)
