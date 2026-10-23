"""llm_execution middleware that routes refusals into Hermes provider fallback."""

from __future__ import annotations

import logging
from typing import Any

from .detector import DetectionResult, RefusalDetector

logger = logging.getLogger(__name__)

_detector = RefusalDetector()


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
            _mark_as_refusal(response, detection)

    return response


def register(ctx: Any) -> None:
    """Register the llm_execution middleware with Hermes."""
    if hasattr(ctx, "register_middleware"):
        ctx.register_middleware("llm_execution", on_llm_execution)
        logger.info("refusal registered as llm_execution middleware")
