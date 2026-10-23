"""Refusal and deflection detection with automatic provider fallback for Hermes."""

from __future__ import annotations

from .detector import DetectionResult, RefusalDetector, detect, is_refusal
from .plugin import on_llm_execution, register

__all__ = [
    "DetectionResult",
    "RefusalDetector",
    "detect",
    "is_refusal",
    "on_llm_execution",
    "register",
]
__version__ = "0.1.1"
