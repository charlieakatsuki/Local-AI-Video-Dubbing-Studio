"""Orchestration and export helpers for structured local translation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from local_dubbing.stt.formatter import format_srt

from .engine import ArgosTranslateEngine, TranslationEngine
from .models import (
    EmptyTranscriptError,
    LanguagePair,
    SUPPORTED_LANGUAGES,
    TranslationResult,
    UnsupportedLanguagePairError,
)


class TranslationManager:
    """Validate requests and delegate translations to a named local engine."""

    def __init__(self, engines: tuple[TranslationEngine, ...] | None = None) -> None:
        active_engines = engines or (ArgosTranslateEngine(),)
        self._engines = {engine.name: engine for engine in active_engines}

    @property
    def engine_names(self) -> tuple[str, ...]:
        return tuple(self._engines)

    def available_language_pairs(self, engine_name: str) -> frozenset[LanguagePair]:
        return self._engine(engine_name).available_language_pairs()

    def translate_transcription(
        self,
        transcription: Any,
        source_language: str,
        target_language: str,
        engine_name: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> TranslationResult:
        self._validate_languages(source_language, target_language)
        segments = tuple(transcription.segments)
        if not any(str(segment.text).strip() for segment in segments):
            raise EmptyTranscriptError("The transcription is empty. Transcribe media with spoken content first.")
        engine = self._engine(engine_name)
        translated_segments = engine.translate_segments(segments, source_language, target_language, progress_callback)
        return TranslationResult(translated_segments, source_language, target_language, engine.name)

    def _engine(self, engine_name: str) -> TranslationEngine:
        try:
            return self._engines[engine_name]
        except KeyError as error:
            raise UnsupportedLanguagePairError("The selected translation engine is unavailable.") from error

    @staticmethod
    def _validate_languages(source_language: str, target_language: str) -> None:
        supported_codes = set(SUPPORTED_LANGUAGES.values())
        if source_language not in supported_codes or target_language not in supported_codes:
            raise UnsupportedLanguagePairError("Choose supported source and target languages.")
        if source_language == target_language:
            raise UnsupportedLanguagePairError("Source and target languages must be different.")


def format_translated_txt(result: TranslationResult) -> str:
    """Format translated output as a UTF-8 text document."""
    return result.full_text + ("\n" if result.full_text else "")


def format_translated_srt(result: TranslationResult) -> str:
    """Format translated segments as SRT without altering their timestamps."""
    return format_srt(result.segments)
