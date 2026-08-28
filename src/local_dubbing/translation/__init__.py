"""Offline translation engine interfaces and adapters will live here."""
"""Local translation components."""

from .engine import ArgosPackageManager, ArgosTranslateEngine, TranslationEngine
from .manager import TranslationManager, format_translated_srt, format_translated_txt
from .models import TranslationResult, TranslationSegment

__all__ = [
    "ArgosPackageManager",
    "ArgosTranslateEngine",
    "TranslationEngine",
    "TranslationManager",
    "TranslationResult",
    "TranslationSegment",
    "format_translated_srt",
    "format_translated_txt",
]
