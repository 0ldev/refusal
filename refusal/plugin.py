"""llm_execution middleware that routes refusals into Hermes provider fallback,
plus the jailbreak kernel that pre-prompts the fallback retry."""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Any

from . import kernel
from .detector import DetectionResult, RefusalDetector

logger = logging.getLogger(__name__)

_detector = RefusalDetector()

#: Per-session kernel arming level (0 = unarmed). A detected refusal raises the
#: level by one (capped at kernel.MAX_LEVEL); the llm_request middleware injects
#: the active kernel text into the system message of every request for an armed
#: session, so the fallback retry carries the preprompt.
_armed: OrderedDict[str, int] = OrderedDict()
_armed_lock = threading.Lock()

#: Upper bound on tracked sessions; the least recently armed entry is evicted first.
_MAX_ARMED_SESSIONS = 1024

#: Opt-in switch from the ``jailbreak_kernel`` plugin setting, captured in register().
_config_enabled = False

_TRUE = {"1", "on", "true", "yes"}
_FALSE = {"0", "off", "false", "no"}


def _has_tool_calls(response: Any) -> bool:
    """Check if response contains tool calls or tool use blocks."""
    if response is None:
        return False

    if isinstance(response, dict):
        choices = response.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            choice = choices[0]
            if isinstance(choice, dict):
                msg = choice.get("message")
                if isinstance(msg, dict) and (msg.get("tool_calls") or msg.get("function_call")):
                    return True
        if response.get("stop_reason") == "tool_use":
            return True
        content = response.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    return True
        return False

    choices = getattr(response, "choices", None)
    if choices and len(choices) > 0:
        choice = choices[0]
        msg = getattr(choice, "message", None)
        if msg is not None:
            if getattr(msg, "tool_calls", None) or getattr(msg, "function_call", None):
                return True
        elif isinstance(choice, dict):
            msg = choice.get("message")
            if isinstance(msg, dict) and (msg.get("tool_calls") or msg.get("function_call")):
                return True

    if getattr(response, "stop_reason", None) == "tool_use":
        return True

    content = getattr(response, "content", None)
    if isinstance(content, list):
        for block in content:
            if getattr(block, "type", None) == "tool_use":
                return True
            if isinstance(block, dict) and block.get("type") == "tool_use":
                return True

    return False


def _extract_text_from_response(response: Any) -> str:
    """Extract assistant text content from various response shapes (OpenAI, SimpleNamespace, dict, Anthropic)."""
    if response is None:
        return ""

    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
            msg = choice.get("message")
            if isinstance(msg, dict):
                return msg.get("content") or ""
            if isinstance(msg, str):
                return msg
            return choice.get("text") or ""
        return response.get("content") or ""

    choices = getattr(response, "choices", None)
    if choices and len(choices) > 0:
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content is not None:
                return content
            if isinstance(message, dict):
                return message.get("content") or ""
        choice_text = getattr(choice, "text", None)
        if choice_text is not None:
            return choice_text
        if isinstance(choice, dict):
            return choice.get("text") or ""

    content = getattr(response, "content", None)
    if content is not None and isinstance(content, str):
        return content

    if hasattr(response, "content") and isinstance(response.content, list):
        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        if parts:
            return " \n".join(parts)

    return ""


def _mark_as_refusal(response: Any, detection: DetectionResult) -> None:
    """Mark the response as content-filtered so Hermes activates its fallback model."""
    logger.warning(
        "Refusal detected (kind=%s, reason=%s, text=%s) — mutating response to content_filter to trigger fallback",
        detection.kind,
        detection.reason,
        detection.matched_text,
    )

    if isinstance(response, dict):
        choices = response.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            if isinstance(choices[0], dict):
                choices[0]["finish_reason"] = "content_filter"
        if "stop_reason" in response:
            response["stop_reason"] = "refusal"
        if "status" in response:
            response["status"] = "incomplete"
            response["incomplete_details"] = {"reason": "content_filter"}
        return

    # OpenAI / ChatCompletions objects
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        if hasattr(choice, "finish_reason"):
            choice.finish_reason = "content_filter"
        elif isinstance(choice, dict):
            choice["finish_reason"] = "content_filter"

        msg = getattr(choice, "message", None)
        if msg is not None and hasattr(msg, "refusal"):
            msg.refusal = f"Refusal detected: {detection.reason or 'safety'}"

    # Anthropic objects
    if hasattr(response, "stop_reason"):
        setattr(response, "stop_reason", "refusal")

    # Codex responses objects
    if hasattr(response, "status"):
        setattr(response, "status", "incomplete")
        setattr(response, "incomplete_details", {"reason": "content_filter"})


def _jailbreak_enabled() -> bool:
    """Opt-in switch for kernel arming/injection (off unless enabled).

    ``REFUSAL_JAILBREAK`` overrides the ``jailbreak_kernel`` plugin setting in both
    directions and is read at call time. Detection and fallback routing are
    unaffected either way.
    """
    override = os.environ.get("REFUSAL_JAILBREAK", "").strip().lower()
    if override in _TRUE:
        return True
    if override in _FALSE:
        return False
    return _config_enabled


def _session_id_from_context(kwargs: Any) -> str:
    """Best-effort session id from middleware context (empty when unavailable)."""
    return str(kwargs.get("session_id") or "")


def _arm_session(session_id: str) -> int:
    """Raise the session's kernel level by one (capped) and return the new level."""
    with _armed_lock:
        level = min(_armed.get(session_id, 0) + 1, kernel.MAX_LEVEL)
        _armed[session_id] = level
        _armed.move_to_end(session_id)
        while len(_armed) > _MAX_ARMED_SESSIONS:
            _armed.popitem(last=False)
    return level


def _disarm_session(session_id: str = "", **_: Any) -> None:
    """Forget a session's kernel level when Hermes finalizes, resets or deletes it."""
    with _armed_lock:
        _armed.pop(str(session_id or ""), None)


def on_llm_request(request: Any, **context: Any) -> dict:
    """Append the armed kernel to the system message of outgoing requests.

    Runs before every provider call. Requests for unarmed sessions (level 0)
    pass through untouched, so a turn's first call is made without the
    preprompt and a refusal on it is genuine. Fail-soft: any error returns the
    request unchanged.
    """
    if not isinstance(request, dict):
        return {"request": request}
    try:
        if _jailbreak_enabled():
            session_id = _session_id_from_context(context)
            with _armed_lock:
                level = _armed.get(session_id, 0)
            if session_id and level >= 1:
                new_request, changed = kernel.inject_into_request(
                    request, kernel.kernel_text(level), str(context.get("api_mode") or "")
                )
                if changed:
                    logger.info(
                        "jailbreak kernel injected (session=%s level=%d model=%s)",
                        session_id, level, context.get("model"),
                    )
                    return {"request": new_request}
    except Exception as exc:
        logger.warning("refusal llm_request middleware error (fail-soft): %s", exc)
    return {"request": request}


def on_llm_execution(**kwargs: Any) -> Any:
    """Run the model call, then flag refusals as content_filter to trigger fallback."""
    next_call = kwargs.get("next_call")
    request = kwargs.get("request")
    response = next_call(request) if next_call else None

    if response is None:
        return response

    if _has_tool_calls(response):
        return response

    text = _extract_text_from_response(response)
    if text:
        detection = _detector.detect(text)
        if detection.is_refusal:
            session_id = _session_id_from_context(kwargs)
            if session_id and _jailbreak_enabled():
                level = _arm_session(session_id)
                logger.info("jailbreak armed for session %s (level=%d)", session_id, level)
            _mark_as_refusal(response, detection)

    return response


def _setting_enabled(ctx: Any) -> bool:
    """Read the ``jailbreak_kernel`` plugin setting (False when absent or not a boolean/flag string)."""
    try:
        value = ctx.get_config("jailbreak_kernel", False)
    except Exception:
        return False
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() in _TRUE


def register(ctx: Any) -> None:
    """Register the middlewares and the session-cleanup hooks with Hermes."""
    global _config_enabled
    _config_enabled = _setting_enabled(ctx)
    if hasattr(ctx, "register_middleware"):
        ctx.register_middleware("llm_execution", on_llm_execution)
        ctx.register_middleware("llm_request", on_llm_request)
        logger.info("refusal registered as llm_execution + llm_request middleware")
    if hasattr(ctx, "register_hook"):
        for hook in ("on_session_finalize", "on_session_reset", "on_session_delete"):
            ctx.register_hook(hook, _disarm_session)
