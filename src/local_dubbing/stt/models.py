"""Structured models and validation for speech-to-text processing."""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_MODELS = ("tiny", "base", "small")
SUPPORTED_LANGUAGES: dict[str, str] = {
    "English": "en",
    "Indonesian": "id",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}


class STTError(Exception):
    """Base exception for user-facing STT failures."""


class InvalidSTTConfigurationError(STTError):
    """Raised when the selected STT configuration is invalid."""


class MissingSTTDependencyError(STTError):
    """Raised when faster-whisper has not been installed."""


class STTModelError(STTError):
    """Raised when a Whisper model cannot be loaded or downloaded."""


class TranscriptionError(STTError):
    """Raised when media cannot be transcribed."""


@dataclass(frozen=True, slots=True)
class STTConfig:
    """User-selectable configuration for a local STT job."""

    model_name: str = "tiny"
    language: str | None = None
    device: str = "auto"
    compute_type: str | None = None

    def __post_init__(self) -> None:
        if self.model_name not in SUPPORTED_MODELS:
            raise InvalidSTTConfigurationError(
                f"Unsupported model '{self.model_name}'. Choose one of: {', '.join(SUPPORTED_MODELS)}."
            )
        if self.language is not None and self.language not in SUPPORTED_LANGUAGES.values():
            raise InvalidSTTConfigurationError("Unsupported source language.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise InvalidSTTConfigurationError("Device must be auto, cpu, or cuda.")
        if self.compute_type is not None and not self.compute_type.strip():
            raise InvalidSTTConfigurationError("Compute type cannot be empty.")


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """A timestamped piece of recognized speech, in seconds."""

    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("Segment timestamps must be non-negative and ordered.")


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Complete local transcription result."""

    segments: tuple[TranscriptionSegment, ...]
    detected_language: str | None
    language_probability: float | None = None

    @property
    def full_text(self) -> str:
        """Return all segment text as a readable transcript."""
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())
