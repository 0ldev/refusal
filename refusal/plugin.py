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

#: turn_id in which a session was last disarmed after refusing despite the preprompt, so a later
#: refusal in that same turn (a second fallback, a tool-loop retry) does not arm it again.
#: Process-local; only effective when Hermes supplies a turn_id.
_disarmed_turns: dict[str, str] = {}

#: Also arm on provider-signalled refusals (``stop_reason=refusal``, ``finish_reason=content_filter``),
#: which carry no matchable text. Hermes falls back on these by itself.
_arm_on_native = True

#: Opt-in: keep armed levels in Hermes's plugin state (``ctx.state``) so they survive a restart
#: and are shared between processes. ``_state`` is that facade, captured in register().
_persist = False
_state: Any = None
_STATE_KEY = "armed"

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


def _content_to_text(content: Any) -> str:
    """Flatten message content: a string, or a list of text parts/blocks (OpenAI parts, Anthropic blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return " \n".join(parts)
    return ""


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
                return _content_to_text(msg.get("content"))
            if isinstance(msg, str):
                return msg
            return _content_to_text(choice.get("text"))
        return _content_to_text(response.get("content"))

    choices = getattr(response, "choices", None)
    if choices and len(choices) > 0:
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content is not None:
                return _content_to_text(content)
            if isinstance(message, dict):
                return _content_to_text(message.get("content"))
        choice_text = getattr(choice, "text", None)
        if choice_text is not None:
            return _content_to_text(choice_text)
        if isinstance(choice, dict):
            return _content_to_text(choice.get("text"))

    return _content_to_text(getattr(response, "content", None))


def _mark_as_refusal(response: Any, detection: DetectionResult, kwargs: Any) -> None:
    """Mark the response as content-filtered so Hermes activates its fallback model.

    Hermes logs every content_filter response identically — ``reason=content_policy_blocked``,
    ``error_type=ContentPolicyBlocked`` — whether the provider actually signalled a refusal or a
    plugin like this one synthesized it (see agent/turn_truncation.py). There is no field that
    tells the two apart downstream, so this line carries the same correlation keys Hermes's own
    native log line prints (session, model, provider, api_request_id) — the only way an operator
    can join the two lines and know which content_policy_blocked events were ours, not the
    provider's. Hermes-agent issue #138900 asked for a public seam to do this properly (a
    registered failover reason distinct from content_policy_blocked); it was closed once this
    plugin shipped the finish_reason mutation as a workaround, without the seam landing.
    """
    logger.warning(
        "Refusal detected (kind=%s, reason=%s, text=%s, session=%s, model=%s, provider=%s, "
        "api_request_id=%s) — mutating response to content_filter to trigger fallback; this is a "
        "plugin-synthesized refusal, not a provider signal, but Hermes will log and count it "
        "identically to one (content_policy_blocked/ContentPolicyBlocked)",
        detection.kind,
        detection.reason,
        detection.matched_text,
        kwargs.get("session_id") or "",
        kwargs.get("model") or "",
        kwargs.get("provider") or "",
        kwargs.get("api_request_id") or "",
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


def _native_refusal_reason(response: Any) -> str:
    """Why the provider itself flagged ``response`` as a refusal ('' when it did not)."""

    def field(obj: Any, name: str) -> Any:
        return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

    if field(response, "stop_reason") == "refusal":
        return "stop_reason=refusal"
    choices = field(response, "choices")
    if isinstance(choices, (list, tuple)) and choices:
        if field(choices[0], "finish_reason") == "content_filter":
            return "finish_reason=content_filter"
    details = field(response, "incomplete_details")
    if field(response, "status") == "incomplete" and field(details, "reason") == "content_filter":
        return "status=incomplete (content_filter)"
    return ""


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


def _refresh_locked() -> None:
    """With persistence on, replace the in-memory table with the stored one (caller holds the lock).

    The stored value is a list of ``[session_id, level, turn_id]`` in least-recently-armed-first
    order. Any storage error turns persistence off for the rest of the process.
    """
    global _persist
    if not (_persist and _state is not None):
        return
    try:
        rows = _state.get(_STATE_KEY, [])
        table: OrderedDict[str, int] = OrderedDict()
        turns: dict[str, str] = {}
        for row in rows if isinstance(rows, list) else []:
            if not (isinstance(row, (list, tuple)) and len(row) == 3):
                continue
            sid, level, turn = row
            if isinstance(sid, str) and type(level) is int and 1 <= level <= preprompt.MAX_LEVEL:
                table[sid] = level
                turns[sid] = turn if isinstance(turn, str) else ""
    except Exception as exc:
        _persist = False
        logger.warning("refusal: plugin state unreadable, keeping armed levels in memory only: %s", exc)
        return
    _armed.clear()
    _armed.update(table)
    _armed_turns.clear()
    _armed_turns.update(turns)


def _flush_locked() -> None:
    """With persistence on, write the in-memory table to the plugin state (caller holds the lock)."""
    global _persist
    if not (_persist and _state is not None):
        return
    try:
        _state.set(_STATE_KEY, [[sid, lvl, _armed_turns.get(sid, "")] for sid, lvl in _armed.items()])
    except Exception as exc:
        _persist = False
        logger.warning("refusal: plugin state not writable, keeping armed levels in memory only: %s", exc)


def _set_level_locked(session_id: str, level: int, turn_id: str = "") -> None:
    """Record a session's level (0 removes it); evicts the least recently armed over the cap.

    The caller holds ``_armed_lock``.
    """
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


def _set_level(session_id: str, level: int, turn_id: str = "") -> None:
    """Locking wrapper around :func:`_set_level_locked`; merges with and writes back the stored table."""
    with _armed_lock:
        _refresh_locked()
        _set_level_locked(session_id, level, turn_id)
        _flush_locked()


def _arm_session(session_id: str, turn_id: str = "") -> int:
    """Raise the session's level by one (capped) and return the new level."""
    with _armed_lock:
        _refresh_locked()
        level = min(_armed.get(session_id, 0) + 1, preprompt.MAX_LEVEL)
        _set_level_locked(session_id, level, turn_id)
        _flush_locked()
    return level


def _disarm_session(session_id: str = "", old_session_id: str = "", **_: Any) -> None:
    """Forget the level of the session that is ending.

    ``on_session_finalize`` passes the ending session as ``session_id``; ``on_session_reset``
    passes the *new* session as ``session_id`` and the ending one as ``old_session_id``.
    """
    for sid in (session_id, old_session_id):
        if sid:
            _set_level(str(sid), 0)
            _disarmed_turns.pop(str(sid), None)


def _on_session_delete(session_id: str = "", deleted_ids: Any = None, **_: Any) -> None:
    """Forget deleted or archived sessions (``on_session_delete``; Hermes 0.22.3+)."""
    ids = [session_id] + [d for d in (deleted_ids if isinstance(deleted_ids, (list, tuple)) else ()) if d]
    for sid in ids:
        if sid:
            _set_level(str(sid), 0)
            _disarmed_turns.pop(str(sid), None)


def _on_compaction(session_id: str = "", old_session_id: str = "", **_: Any) -> None:
    """Carry an armed level over when compaction rotates the session id (``post_compaction``; Hermes 0.22.3+).

    ``old_session_id`` is empty when the transcript was compacted in place, in which case the id
    did not change and there is nothing to move.
    """
    old, new = str(old_session_id or ""), str(session_id or "")
    if not old or not new or old == new:
        return
    with _armed_lock:
        # A disarm made earlier in this turn must also hold for the session's new id.
        if old in _disarmed_turns:
            _disarmed_turns[new] = _disarmed_turns.pop(old)
        _refresh_locked()
        level = _armed.get(old, 0)
        if not level:
            return
        turn = _armed_turns.get(old, "")
        _set_level_locked(old, 0)
        _set_level_locked(new, max(level, _armed.get(new, 0)), turn)
        _flush_locked()
    logger.info("preprompt level %d carried over a compaction (%s -> %s)", level, old, new)


def _model_allowed(context: Any) -> bool:
    """True when the request's provider/model matches ``preprompt_models`` (empty list = any)."""
    if not _models:
        return True
    model = str(context.get("model") or "").lower()
    provider = str(context.get("provider") or "").lower()
    names = {model, f"{provider}/{model}"}
    return any(fnmatch.fnmatchcase(n, pat) for pat in _models for n in names)


def _peek_level(session_id: str) -> int:
    """The session's stored level without applying ``turn`` scope."""
    with _armed_lock:
        _refresh_locked()
        return _armed.get(session_id, 0)


def _current_level(session_id: str, turn_id: str) -> int:
    """The session's level, forgetting it first when ``turn`` scope says a new turn began."""
    with _armed_lock:
        _refresh_locked()
        level = _armed.get(session_id, 0)
        armed_turn = _armed_turns.get(session_id, "")
    if level and _scope == "turn" and turn_id and armed_turn and armed_turn != turn_id:
        _set_level(session_id, 0)
        return 0
    return level


def on_llm_request(request: Any, **context: Any) -> dict | None:
    """Append the configured preprompt to the system prompt of outgoing requests.

    Runs before every provider call. Requests for unarmed sessions (level 0), or to a
    provider/model outside ``preprompt_models``, are left alone by returning ``None``
    (Hermes then neither copies the payload nor records a trace entry), so a turn's
    first call is made without the preprompt and a refusal on it is genuine.
    Fail-soft: any error leaves the request unchanged.
    """
    if not isinstance(request, dict):
        return None
    try:
        if _preprompt_enabled():
            session_id = _session_id_from_context(context)
            level = _current_level(session_id, str(context.get("turn_id") or "")) if session_id else 0
            text = preprompt.preprompt_text(level, _texts)
            if level >= 1 and _model_allowed(context) and not preprompt.request_contains(request, text):
                new_request, changed = preprompt.inject_into_request(
                    request, text, str(context.get("api_mode") or "")
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
    return None


def _remember_disarm(session_id: str, turn_id: str) -> None:
    """Note that the session was disarmed in this turn (bounded like the armed table)."""
    if not turn_id:
        return
    with _armed_lock:
        _disarmed_turns.pop(session_id, None)
        _disarmed_turns[session_id] = turn_id
        while len(_disarmed_turns) > _max_armed_sessions:
            _disarmed_turns.pop(next(iter(_disarmed_turns)))


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
            _remember_disarm(session_id, turn_id)
        else:
            new_level = level
        logger.warning("refused despite preprompt (%s); mode=%s -> level %d",
                       where, _on_preprompt_refusal, new_level)
    elif level == 0 and turn_id and _disarmed_turns.get(session_id) == turn_id:
        logger.info("refused again after disarm in this turn (%s); staying disarmed", where)
    elif level == 0:
        logger.info("preprompt armed (%s) -> level %d", where, _arm_session(session_id, turn_id))
    else:
        logger.info("refused without preprompt (%s); level unchanged", where)


def on_llm_execution(**kwargs: Any) -> Any:
    """Run the model call, then flag refusals as content_filter to trigger fallback.

    Everything after the provider call is fail-soft: an error in detection or in the
    preprompt bookkeeping is logged and the response is returned as received.
    """
    next_call = kwargs.get("next_call")
    request = kwargs.get("request")
    response = next_call(request) if next_call else None

    if response is None:
        return response

    try:
        _inspect_response(request, response, kwargs)
    except Exception as exc:
        logger.warning("refusal llm_execution middleware error (fail-soft): %s", exc)
    return response


def _inspect_response(request: Any, response: Any, kwargs: Any) -> None:
    """Detect a refusal in ``response``, update the session level and mark it for failover."""
    if _has_tool_calls(response):
        return

    native = _native_refusal_reason(response) if _arm_on_native else ""
    if native:
        # The provider already marked it, so Hermes will fall back; only record the refusal.
        logger.info("provider-signalled refusal (%s)", native)
        _handle_refusal(request, kwargs)
        return

    text = _extract_text_from_response(response)
    if not text:
        return
    detection = _detector.detect(text)
    if detection.is_refusal:
        # Mark first: failover must not depend on the optional preprompt bookkeeping.
        _mark_as_refusal(response, detection, kwargs)
        _handle_refusal(request, kwargs)
    elif _preprompt_enabled() and preprompt.request_contains(
        request, preprompt.preprompt_text(_peek_level(_session_id_from_context(kwargs)) or 1, _texts)
    ):
        logger.info("answered with preprompt (session=%s provider=%s model=%s)",
                    _session_id_from_context(kwargs), kwargs.get("provider"), kwargs.get("model"))


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


def _setting_flag(ctx: Any, key: str, default: bool) -> bool:
    """Read a boolean plugin setting (the default when absent or not a bool/flag string)."""
    try:
        value = ctx.get_config(key, default)
    except Exception:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        return True if v in _TRUE else False if v in _FALSE else default
    return default


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
    """Register the middlewares and the session-lifecycle hooks with Hermes."""
    global _texts, _max_armed_sessions, _on_preprompt_refusal, _scope, _models, _arm_on_native, _persist, _state
    _texts = (_setting_text(ctx, "preprompt"), _setting_text(ctx, "preprompt_escalated"))
    _max_armed_sessions = _setting_max_sessions(ctx)
    _on_preprompt_refusal = _setting_choice(ctx, "on_preprompt_refusal", _ON_REFUSAL_MODES, "disarm")
    _scope = _setting_choice(ctx, "preprompt_scope", _SCOPES, "turn")
    _models = _setting_models(ctx)
    _arm_on_native = _setting_flag(ctx, "arm_on_native_refusal", True)
    _persist = _setting_flag(ctx, "persist_state", False)
    _state = getattr(ctx, "state", None) if _persist else None
    if _persist and _state is None:
        _persist = False
        logger.warning("refusal: persist_state is on but this Hermes exposes no ctx.state; using memory only")
    if hasattr(ctx, "register_middleware"):
        ctx.register_middleware("llm_execution", on_llm_execution)
        ctx.register_middleware("llm_request", on_llm_request)
        logger.info("refusal registered as llm_execution + llm_request middleware")
    if hasattr(ctx, "register_hook"):
        ctx.register_hook("on_session_finalize", _disarm_session)
        ctx.register_hook("on_session_reset", _disarm_session)
        # Valid hook names from Hermes 0.22.3, which is the requires_hermes floor.
        ctx.register_hook("on_session_delete", _on_session_delete)
        ctx.register_hook("post_compaction", _on_compaction)
