"""Retry preprompt: request-layer injection of user-supplied text.

When a model refuses a turn, the plugin arms a per-session level. The
``llm_request`` middleware then appends the text the user configured for that level
to the system prompt of subsequent requests (including the Hermes fallback retry).
The plugin ships no preprompt text of its own.
"""

from __future__ import annotations

from typing import Any, List, Tuple

#: api_modes whose payload carries the system prompt in a top-level ``system`` field.
_SYSTEM_FIELD_MODES = frozenset({"anthropic_messages", "bedrock_converse"})

#: Highest level: further refusals keep the session here.
MAX_LEVEL = 2

def preprompt_text(level: int, texts: Tuple[str, str]) -> str:
    """Text for an arming level: ``texts`` is ``(level 1, level 2+)``; level 2 falls back to level 1."""
    first, escalated = texts
    if level >= MAX_LEVEL and escalated:
        return escalated
    return first


def inject_into_messages(messages: Any, text: str) -> Tuple[Any, bool]:
    """Return ``(new_messages, changed)`` with the text appended to the last system message.

    If no system message exists, one is prepended. The input list is never mutated;
    a new list with copied message dicts is returned. Returns the input unchanged
    (``changed=False``) when injection is not possible.
    """
    if not isinstance(messages, list) or not messages or not text:
        return messages, False

    new_messages: List[Any] = [dict(m) if isinstance(m, dict) else m for m in messages]
    for i in range(len(new_messages) - 1, -1, -1):
        msg = new_messages[i]
        if isinstance(msg, dict) and msg.get("role") == "system":
            original = msg.get("content")
            if isinstance(original, list):
                msg["content"] = [*original, {"type": "text", "text": text}]
                return new_messages, True
            base = original if isinstance(original, str) else ("" if original is None else str(original))
            msg["content"] = f"{base}\n\n{text}" if base else text
            return new_messages, True

    new_messages.insert(0, {"role": "system", "content": text})
    return new_messages, True


def inject_into_request(request: Any, text: str, api_mode: str = "") -> Tuple[Any, bool]:
    """Return ``(new_request, changed)`` with the text added where the provider reads its system prompt.

    Hermes builds a different payload per ``api_mode``: chat completions carry a ``system``
    message in ``messages``, Anthropic and Bedrock a top-level ``system`` field (a string or a
    list of text blocks), and Responses/Codex a top-level ``instructions`` string. The input
    request is never mutated.
    """
    if not isinstance(request, dict) or not text:
        return request, False

    if api_mode == "codex_responses" or "instructions" in request or "input" in request:
        base = request.get("instructions")
        base = base if isinstance(base, str) else ""
        new_request = dict(request)
        new_request["instructions"] = f"{base}\n\n{text}" if base else text
        return new_request, True

    if api_mode in _SYSTEM_FIELD_MODES or "system" in request:
        system = request.get("system")
        new_request = dict(request)
        if isinstance(system, list):
            bedrock = api_mode == "bedrock_converse" or (
                bool(system) and isinstance(system[0], dict) and "type" not in system[0]
            )
            block = {"text": text} if bedrock else {"type": "text", "text": text}
            new_request["system"] = [*system, block]
        elif isinstance(system, str) and system:
            new_request["system"] = f"{system}\n\n{text}"
        elif api_mode == "bedrock_converse":
            new_request["system"] = [{"text": text}]
        else:
            new_request["system"] = text
        return new_request, True

    new_messages, changed = inject_into_messages(request.get("messages"), text)
    if not changed:
        return request, False
    new_request = dict(request)
    new_request["messages"] = new_messages
    return new_request, True


def _strings(value: Any) -> List[str]:
    """Every string reachable inside nested dicts/lists (message content, system blocks)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in _strings(v)]
    return []


def request_contains(request: Any, text: str) -> bool:
    """True when ``text`` appears in the request's system prompt (any provider shape)."""
    if not isinstance(request, dict) or not text:
        return False
    for key in ("system", "instructions"):
        if any(text in chunk for chunk in _strings(request.get(key))):
            return True
    for msg in request.get("messages") or ():
        if isinstance(msg, dict) and msg.get("role") == "system":
            if any(text in chunk for chunk in _strings(msg.get("content"))):
                return True
    return False
