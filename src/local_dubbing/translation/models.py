"""Structured models and user-facing errors for local translation."""

from __future__ import annotations

from dataclasses import dataclass


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


class TranslationError(Exception):
    """Base exception for safe, user-facing translation failures."""


class UnsupportedLanguagePairError(TranslationError):
    """Raised when a requested language code or pair is not supported."""


class MissingTranslationPackageError(TranslationError):
    """Raised when the required local Argos package has not been installed."""


class TranslationFailedError(TranslationError):
    """Raised when a backend cannot translate otherwise valid text."""


class EmptyTranscriptError(TranslationError):
    """Raised when there is no transcribed text to translate."""


@dataclass(frozen=True, slots=True)
class LanguagePair:
    """A direct source-to-target language package pairing."""

    source_language: str
    target_language: str


@dataclass(frozen=True, slots=True)
class TranslationSegment:
    """A translated STT segment retaining its original timestamps."""

    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("Segment timestamps must be non-negative and ordered.")


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Translation output suitable for display and subtitle export."""

    segments: tuple[TranslationSegment, ...]
    source_language: str
    target_language: str
    engine_name: str

    @property
    def full_text(self) -> str:
        """Return segment text as a readable translated transcript."""
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())
