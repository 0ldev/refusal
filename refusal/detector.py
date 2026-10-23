"""Refusal and deflection detector implementation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .patterns import (
    COMPILED_DEFLECTION,
    COMPILED_DIRECT_REFUSAL,
    COMPILED_PARTIAL_ANSWER,
    COMPILED_POLICY_REFUSAL,
)


@dataclass(frozen=True)
class DetectionResult:
    """Result of refusal and deflection detection."""

    is_refusal: bool
    kind: Optional[str] = None
    reason: Optional[str] = None
    matched_text: Optional[str] = None

    def __bool__(self) -> bool:
        return self.is_refusal


class RefusalDetector:
    """Detector for model refusals and off-topic deflections."""

    def _strip_fenced_code_blocks(self, text: str) -> str:
        """Remove fenced code blocks to prevent false positives inside code strings."""
        return re.sub(r"```[\s\S]*?```", " ", text)

    def _is_partial_answer(self, clean_text: str) -> bool:
        """Check if the text represents a partial-but-genuine answer delivering content."""
        for pattern in COMPILED_PARTIAL_ANSWER:
            if pattern.search(clean_text):
                return True
        return False

    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> DetectionResult:
        """Analyze text to determine if it is a refusal or deflection.

        Args:
            text: The model response content.
            context: Reserved for future use; currently ignored.

        Returns:
            DetectionResult indicating whether the text represents a refusal/deflection.
        """
        if not text or not text.strip():
            return DetectionResult(is_refusal=False)

        clean_text = self._strip_fenced_code_blocks(text).strip()
        if not clean_text:
            return DetectionResult(is_refusal=False)

        # Deflections first: a refusal followed by an educational/defensive pivot.
        for pattern in COMPILED_DEFLECTION:
            match = pattern.search(clean_text)
            if match:
                matched_snippet = match.group(0).strip()
                return DetectionResult(
                    is_refusal=True,
                    kind="deflection",
                    reason="off_topic_deflection",
                    matched_text=matched_snippet,
                )

        # A qualified inability that still delivers content is a genuine answer.
        if self._is_partial_answer(clean_text):
            return DetectionResult(is_refusal=False)

        for pattern in COMPILED_DIRECT_REFUSAL:
            match = pattern.search(clean_text)
            if match:
                matched_snippet = match.group(0).strip()
                return DetectionResult(
                    is_refusal=True,
                    kind="refusal",
                    reason="direct_refusal",
                    matched_text=matched_snippet,
                )

        for pattern in COMPILED_POLICY_REFUSAL:
            match = pattern.search(clean_text)
            if match:
                matched_snippet = match.group(0).strip()
                return DetectionResult(
                    is_refusal=True,
                    kind="refusal",
                    reason="policy_refusal",
                    matched_text=matched_snippet,
                )

        return DetectionResult(is_refusal=False)


_DEFAULT_DETECTOR = RefusalDetector()


def detect(text: str, context: Optional[Dict[str, Any]] = None) -> DetectionResult:
    """Detect if a model response is a refusal or deflection.

    Args:
        text: The model response content.
        context: Reserved for future use; currently ignored.
    """
    return _DEFAULT_DETECTOR.detect(text, context=context)


def is_refusal(text: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience helper returning True if text is a refusal or deflection.

    Args:
        text: The model response content.
        context: Reserved for future use; currently ignored.
    """
    return detect(text, context=context).is_refusal
