"""Structured models and validation for local text-to-speech."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping


SUPPORTED_LANGUAGES = frozenset({"en", "id", "ja", "ko", "zh", "es", "fr", "de"})
DEFAULT_VOXCPM_MODEL = "openbmb/VoxCPM2"
_DEVICE_PATTERN = re.compile(r"^(auto|cpu|mps|cuda(?::\d+)?)$")


class TTSError(Exception):
    """Base exception for safe, user-facing TTS failures."""


class InvalidTTSConfigurationError(TTSError):
    """Raised when TTS configuration is invalid."""


class InvalidTTSSegmentError(TTSError):
    """Raised when a translated segment cannot be synthesized safely."""


class EmptyTTSTextError(InvalidTTSSegmentError):
    """Raised when a TTS segment contains no spoken text."""


class MissingTTSDependencyError(TTSError):
    """Raised when the selected local TTS backend is not installed."""


class TTSModelError(TTSError):
    """Raised when a local TTS model cannot be loaded."""


class TTSSynthesisError(TTSError):
    """Raised when speech generation or audio writing fails."""


class UnsupportedTTSEngineError(TTSError):
    """Raised when a requested TTS engine is unavailable."""


@dataclass(frozen=True, slots=True)
class TTSConfig:
    """Portable VoxCPM generation settings; model loading remains lazy."""

    model_name: str = DEFAULT_VOXCPM_MODEL
    device: str = "auto"
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    seed: int | None = 42
    load_denoiser: bool = False
    optimize: bool = False
    local_files_only: bool = False
    model_cache_dir: Path | None = None
    voice_description: str | None = None

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise InvalidTTSConfigurationError("A VoxCPM model name or local model path is required.")
        if not _DEVICE_PATTERN.fullmatch(self.device):
            raise InvalidTTSConfigurationError("Device must be auto, cpu, mps, cuda, or cuda:<index>.")
        if self.cfg_value <= 0:
            raise InvalidTTSConfigurationError("CFG value must be greater than zero.")
        if self.inference_timesteps <= 0:
            raise InvalidTTSConfigurationError("Inference timesteps must be greater than zero.")
        if self.seed is not None and self.seed < 0:
            raise InvalidTTSConfigurationError("Seed must be non-negative or None.")
        if self.voice_description is not None and not self.voice_description.strip():
            raise InvalidTTSConfigurationError("Voice description cannot be blank.")


@dataclass(frozen=True, slots=True)
class TTSSegment:
    """Translated text and original timing supplied to a TTS engine."""

    segment_id: str
    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        if not str(self.segment_id).strip():
            raise InvalidTTSSegmentError("Segment ID cannot be empty.")
        if self.start < 0 or self.end < self.start:
            raise InvalidTTSSegmentError("Segment timestamps must be non-negative and ordered.")
        if not self.text.strip():
            raise EmptyTTSTextError("Segment text cannot be empty.")


@dataclass(frozen=True, slots=True)
class GeneratedAudioSegment:
    """Generated speech for one translated segment, before synchronization."""

    segment_id: str
    audio_path: Path
    duration: float
    start: float
    end: float
    target_language: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.segment_id).strip():
            raise InvalidTTSSegmentError("Generated audio must retain a segment ID.")
        if self.duration < 0:
            raise InvalidTTSSegmentError("Generated audio duration cannot be negative.")
        if self.start < 0 or self.end < self.start:
            raise InvalidTTSSegmentError("Original timestamps must be non-negative and ordered.")
        if self.target_language not in SUPPORTED_LANGUAGES:
            raise InvalidTTSConfigurationError("Unsupported target language for generated audio.")


@dataclass(frozen=True, slots=True)
class TTSResult:
    """Structured generated audio returned by any local TTS backend."""

    segments: tuple[GeneratedAudioSegment, ...]
    target_language: str
    engine_name: str
    model_name: str
