"""llm_execution middleware that routes refusals into Hermes provider fallback,
plus the optional user-supplied preprompt for the fallback retry."""

from __future__ import annotations

import fnmatch
import logging
import os
import threading
from collections import OrderedDict
from typing import Any

from . import preprompt
from .detector import DetectionResult, RefusalDetector

logger = logging.getLogger(__name__)

_detector = RefusalDetector()

#: Per-session arming level (0 = unarmed). A detected refusal raises the
#: level by one (capped at preprompt.MAX_LEVEL); the llm_request middleware injects
#: the configured preprompt into the system message of every request for an armed
#: session, so the fallback retry carries the preprompt.
_armed: OrderedDict[str, int] = OrderedDict()
_armed_lock = threading.Lock()

#: Default upper bound on tracked sessions; the least recently armed entry is evicted
#: first. Overridable with the ``max_armed_sessions`` plugin setting.
DEFAULT_MAX_ARMED_SESSIONS = 1024
_max_armed_sessions = DEFAULT_MAX_ARMED_SESSIONS

#: User-supplied preprompt texts (level 1, level 2+) from the plugin settings, captured
#: in register(). Both empty means the feature is off.
_texts: tuple[str, str] = ("", "")

#: What a refusal that happened *with* the preprompt does to the session: ``escalate`` (level 2
#: text), ``disarm`` (stop sending the preprompt) or ``keep`` (leave the level alone).
_ON_REFUSAL_MODES = ("escalate", "disarm", "keep")
_on_preprompt_refusal = "disarm"

#: ``turn`` forgets the level when a new turn starts; ``session`` keeps it until the session ends.
_SCOPES = ("turn", "session")
_scope = "turn"

#: ``provider/model`` (or bare ``model``) glob patterns the preprompt may be sent to; empty = any.
_models: tuple[str, ...] = ()

#: turn_id in which each armed session was last armed (for ``turn`` scope).
_armed_turns: dict[str, str] = {}

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


def _preprompt_enabled() -> bool:
    """True when the user configured a preprompt and has not switched it off.

    ``REFUSAL_PREPROMPT=0`` (or off/false/no) disables arming and injection, read at call
    time. Detection and fallback routing are unaffected either way.
    """
    if os.environ.get("REFUSAL_PREPROMPT", "").strip().lower() in _FALSE:
        return False
    return bool(_texts[0])


def _session_id_from_context(kwargs: Any) -> str:
    """Best-effort session id from middleware context (empty when unavailable)."""
    return str(kwargs.get("session_id") or "")


def _set_level(session_id: str, level: int, turn_id: str = "") -> None:
    """Record a session's level (0 removes it); evicts the least recently armed over the cap."""
    with _armed_lock:
        if level <= 0:
            _armed.pop(session_id, None)
            _armed_turns.pop(session_id, None)
            return
        _armed[session_id] = level
        _armed.move_to_end(session_id)
        _armed_turns[session_id] = turn_id
        while len(_armed) > _max_armed_sessions:
            evicted, _ = _armed.popitem(last=False)
            _armed_turns.pop(evicted, None)


def _arm_session(session_id: str, turn_id: str = "") -> int:
    """Raise the session's level by one (capped) and return the new level."""
    level = min(_armed.get(session_id, 0) + 1, preprompt.MAX_LEVEL)
    _set_level(session_id, level, turn_id)
    return level


def _disarm_session(session_id: str = "", **_: Any) -> None:
    """Forget a session's level when Hermes finalizes or resets it."""
    _set_level(str(session_id or ""), 0)


def _model_allowed(context: Any) -> bool:
    """True when the request's provider/model matches ``preprompt_models`` (empty list = any)."""
    if not _models:
        return True
    model = str(context.get("model") or "").lower()
    provider = str(context.get("provider") or "").lower()
    names = {model, f"{provider}/{model}"}
    return any(fnmatch.fnmatchcase(n, pat) for pat in _models for n in names)


def _current_level(session_id: str, turn_id: str) -> int:
    """The session's level, forgetting it first when ``turn`` scope says a new turn began."""
    with _armed_lock:
        level = _armed.get(session_id, 0)
        armed_turn = _armed_turns.get(session_id, "")
    if level and _scope == "turn" and turn_id and armed_turn and armed_turn != turn_id:
        _set_level(session_id, 0)
        return 0
    return level


def on_llm_request(request: Any, **context: Any) -> dict:
    """Append the configured preprompt to the system prompt of outgoing requests.

    Runs before every provider call. Requests for unarmed sessions (level 0), or to a
    provider/model outside ``preprompt_models``, pass through untouched, so a turn's
    first call is made without the preprompt and a refusal on it is genuine.
    Fail-soft: any error returns the request unchanged.
    """
    if not isinstance(request, dict):
        return {"request": request}
    try:
        if _preprompt_enabled():
            session_id = _session_id_from_context(context)
            level = _current_level(session_id, str(context.get("turn_id") or "")) if session_id else 0
            if level >= 1 and _model_allowed(context):
                new_request, changed = preprompt.inject_into_request(
                    request, preprompt.preprompt_text(level, _texts), str(context.get("api_mode") or "")
                )
                if changed:
                    logger.info(
                        "preprompt injected (session=%s level=%d provider=%s model=%s)",
                        session_id, level, context.get("provider"), context.get("model"),
                    )
                    return {
                        "request": new_request,
                        "source": "refusal",
                        "reason": f"appended retry preprompt (level {level}) after an earlier refusal",
                    }
    except Exception as exc:
        logger.warning("refusal llm_request middleware error (fail-soft): %s", exc)
    return {"request": request}


def _handle_refusal(request: Any, kwargs: Any) -> None:
    """Update the session's level after a detected refusal and log what happened."""
    session_id = _session_id_from_context(kwargs)
    if not _preprompt_enabled():
        return
    if not session_id:
        logger.info("refusal detected without a session_id; preprompt not armed")
        return
    turn_id = str(kwargs.get("turn_id") or "")
    level = _current_level(session_id, turn_id)
    sent = level >= 1 and preprompt.request_contains(request, preprompt.preprompt_text(level, _texts))
    where = "session=%s level=%d provider=%s model=%s" % (
        session_id, level, kwargs.get("provider"), kwargs.get("model"))
    if sent:
        if _on_preprompt_refusal == "escalate":
            new_level = _arm_session(session_id, turn_id)
        elif _on_preprompt_refusal == "disarm":
            new_level = 0
            _set_level(session_id, 0)
        else:
            new_level = level
        logger.warning("refused despite preprompt (%s); mode=%s -> level %d",
                       where, _on_preprompt_refusal, new_level)
    elif level == 0:
        logger.info("preprompt armed (%s) -> level %d", where, _arm_session(session_id, turn_id))
    else:
        logger.info("refused without preprompt (%s); level unchanged", where)


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
            _handle_refusal(request, kwargs)
            _mark_as_refusal(response, detection)
        elif _preprompt_enabled() and preprompt.request_contains(
            request, preprompt.preprompt_text(_armed.get(_session_id_from_context(kwargs), 0), _texts)
        ):
            logger.info("answered with preprompt (session=%s provider=%s model=%s)",
                        _session_id_from_context(kwargs), kwargs.get("provider"), kwargs.get("model"))

    return response


def _setting_text(ctx: Any, key: str) -> str:
    """Read a string plugin setting (empty when absent or not a string)."""
    try:
        value = ctx.get_config(key, "")
    except Exception:
        return ""
    return value.strip() if isinstance(value, str) else ""


def _setting_max_sessions(ctx: Any) -> int:
    """Read ``max_armed_sessions`` (a positive int), falling back to the default."""
    try:
        value = ctx.get_config("max_armed_sessions", DEFAULT_MAX_ARMED_SESSIONS)
    except Exception:
        return DEFAULT_MAX_ARMED_SESSIONS
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return DEFAULT_MAX_ARMED_SESSIONS
    return value


def _setting_choice(ctx: Any, key: str, choices: tuple[str, ...], default: str) -> str:
    """Read a string setting restricted to ``choices`` (the default when invalid)."""
    try:
        value = ctx.get_config(key, default)
    except Exception:
        return default
    value = value.strip().lower() if isinstance(value, str) else ""
    return value if value in choices else default


def _setting_models(ctx: Any) -> tuple[str, ...]:
    """Read ``preprompt_models`` (a list of glob patterns, or one comma-separated string)."""
    try:
        value = ctx.get_config("preprompt_models", [])
    except Exception:
        return ()
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(v.strip().lower() for v in value if isinstance(v, str) and v.strip())


def register(ctx: Any) -> None:
    """Register the middlewares and the session-cleanup hooks with Hermes."""
    global _texts, _max_armed_sessions, _on_preprompt_refusal, _scope, _models
    _texts = (_setting_text(ctx, "preprompt"), _setting_text(ctx, "preprompt_escalated"))
    _max_armed_sessions = _setting_max_sessions(ctx)
    _on_preprompt_refusal = _setting_choice(ctx, "on_preprompt_refusal", _ON_REFUSAL_MODES, "disarm")
    _scope = _setting_choice(ctx, "preprompt_scope", _SCOPES, "turn")
    _models = _setting_models(ctx)
    if hasattr(ctx, "register_middleware"):
        ctx.register_middleware("llm_execution", on_llm_execution)
        ctx.register_middleware("llm_request", on_llm_request)
        logger.info("refusal registered as llm_execution + llm_request middleware")
    if hasattr(ctx, "register_hook"):
        for hook in ("on_session_finalize", "on_session_reset"):
            ctx.register_hook(hook, _disarm_session)
