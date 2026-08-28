"""Validation and backend selection for structured local TTS generation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from .engine import TextToSpeechEngine, VoxCPMEngine
from .models import (
    EmptyTTSTextError,
    InvalidTTSSegmentError,
    SUPPORTED_LANGUAGES,
    TTSConfig,
    TTSResult,
    TTSSegment,
    UnsupportedTTSEngineError,
)


class TTSManager:
    """Select a local engine without coupling callers to backend details."""

    def __init__(self, engines: tuple[TextToSpeechEngine, ...] | None = None) -> None:
        active_engines = (VoxCPMEngine(),) if engines is None else engines
        self._engines = {engine.name: engine for engine in active_engines}

    @property
    def engine_names(self) -> tuple[str, ...]:
        return tuple(self._engines)

    def synthesize_segments(
        self,
        segments: Iterable[TTSSegment],
        target_language: str,
        output_dir: Path,
        engine_name: str,
        config: TTSConfig,
        progress_callback: Callable[[str], None] | None = None,
    ) -> TTSResult:
        if target_language not in SUPPORTED_LANGUAGES:
            raise InvalidTTSSegmentError("Choose a supported target language for speech generation.")
        materialized_segments = tuple(segments)
        if not materialized_segments:
            raise EmptyTTSTextError("There are no translated segments to synthesize.")
        segment_ids = [segment.segment_id for segment in materialized_segments]
        if len(set(segment_ids)) != len(segment_ids):
            raise InvalidTTSSegmentError("Segment IDs must be unique.")
        engine = self._engine(engine_name)
        generated = engine.synthesize(
            materialized_segments,
            Path(output_dir),
            target_language,
            config,
            progress_callback,
        )
        return TTSResult(generated, target_language, engine.name, config.model_name)

    def _engine(self, engine_name: str) -> TextToSpeechEngine:
        try:
            return self._engines[engine_name]
        except KeyError as error:
            raise UnsupportedTTSEngineError("The selected text-to-speech engine is unavailable.") from error
