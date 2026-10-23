"""refusal: Hermes plugin entry point."""

from .refusal import DetectionResult, RefusalDetector, detect, is_refusal
from .refusal.plugin import on_llm_execution, register

__all__ = [
    "register",
    "on_llm_execution",
    "DetectionResult",
    "RefusalDetector",
    "detect",
    "is_refusal",
]
