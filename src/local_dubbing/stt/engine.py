"""Pluggable faster-whisper implementation of local speech-to-text."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .models import (
    MissingSTTDependencyError,
    STTConfig,
    STTModelError,
    TranscriptionError,
    TranscriptionResult,
    TranscriptionSegment,
)

ProgressCallback = Callable[[str], None]


class SpeechToTextEngine(Protocol):
    """Boundary that lets future STT engines replace faster-whisper."""

    def transcribe(
        self, media_path: Path, config: STTConfig, progress_callback: ProgressCallback | None = None
    ) -> TranscriptionResult: ...


def cuda_available() -> bool:
    """Return whether CTranslate2 reports a CUDA-capable device, safely."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except (ImportError, OSError, AttributeError):
        return False


def resolve_device_and_compute_type(config: STTConfig) -> tuple[str, str]:
    """Resolve portable defaults while retaining explicit user overrides."""
    device = config.device
    if device == "auto":
        device = "cuda" if cuda_available() else "cpu"
    compute_type = config.compute_type or ("float16" if device == "cuda" else "int8")
    return device, compute_type


class FasterWhisperEngine:
    """Lazy faster-whisper adapter; no model is loaded until transcription."""

    def transcribe(
        self, media_path: Path, config: STTConfig, progress_callback: ProgressCallback | None = None
    ) -> TranscriptionResult:
        if not media_path.is_file() or media_path.stat().st_size == 0:
            raise TranscriptionError("The uploaded media file is missing or empty.")
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise MissingSTTDependencyError(
                "faster-whisper is not installed. Install the project dependencies and try again."
            ) from error

        device, compute_type = resolve_device_and_compute_type(config)
        if progress_callback:
            progress_callback(f"Loading the {config.model_name} model on {device}…")
        try:
            model = WhisperModel(config.model_name, device=device, compute_type=compute_type)
        except Exception as error:
            raise STTModelError(
                "The Whisper model could not be downloaded or loaded. Check your connection, disk space, "
                "and selected device settings."
            ) from error

        if progress_callback:
            progress_callback("Transcribing media locally…")
        try:
            raw_segments, info = model.transcribe(str(media_path), language=config.language)
            segments = []
            for raw_segment in raw_segments:
                segments.append(
                    TranscriptionSegment(
                        start=float(raw_segment.start), end=float(raw_segment.end), text=str(raw_segment.text).strip()
                    )
                )
            if progress_callback:
                progress_callback("Transcription complete.")
            return TranscriptionResult(
                segments=tuple(segments),
                detected_language=getattr(info, "language", None),
                language_probability=getattr(info, "language_probability", None),
            )
        except Exception as error:
            raise TranscriptionError(
                "The media could not be transcribed. Confirm it contains readable audio and FFmpeg is available."
            ) from error
