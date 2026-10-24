"""Refusal and deflection detection with automatic provider fallback for Hermes."""

from __future__ import annotations

from .detector import DetectionResult, RefusalDetector, detect, is_refusal
from .plugin import on_llm_execution, on_llm_request, register

__all__ = [
    "DetectionResult",
    "RefusalDetector",
    "detect",
    "is_refusal",
    "on_llm_execution",
    "on_llm_request",
    "register",
]
__version__ = "0.2.0"
