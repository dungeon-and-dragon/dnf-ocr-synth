"""Reusable, local-font nickname synthesis for DNF OCR datasets."""

from .fonts import FontPaths
from .nickname import ValidationResult, estimate_ui_scale, validate_nickname
from .render import Renderer, Sample
from .scale_model import SCALE_MODELS, ScaleModel, SeparableKernel

__all__ = [
    "SCALE_MODELS",
    "FontPaths",
    "Renderer",
    "Sample",
    "ScaleModel",
    "SeparableKernel",
    "ValidationResult",
    "estimate_ui_scale",
    "validate_nickname",
]
