"""Structured timing-alignment models for generated speech segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Mapping


class TimingAlignmentError(Exception):
    """Base exception for user-facing timing-alignment failures."""


class InvalidAlignmentConfigurationError(TimingAlignmentError):
    """Raised when timing-alignment settings are invalid."""


class InvalidAlignmentSegmentError(TimingAlignmentError):
    """Raised when generated audio cannot be planned safely."""


class AudioProcessingError(Exception):
    """Base exception for safe, user-facing audio-processing failures."""


class InvalidAudioProcessingInputError(AudioProcessingError):
    """Raised when an alignment plan or source WAV cannot be processed safely."""


class MissingAudioProcessingDependencyError(AudioProcessingError):
    """Raised when the local FFmpeg tools are unavailable."""


class AudioProcessingFailedError(AudioProcessingError):
    """Raised when local audio processing cannot produce a valid WAV file."""


class AlignmentAction(str, Enum):
    """Backend-neutral operation requested from a future audio processor."""

    KEEP = "keep"
    PAD_END = "pad_end"
    SPEED_UP = "speed_up"
    SPEED_UP_AND_TRIM = "speed_up_and_trim"


@dataclass(frozen=True, slots=True)
class TimingAlignmentConfig:
    """Policy for matching generated speech to its original timestamp slot."""

    tolerance_seconds: float = 0.05
    max_speed_up: float = 1.5

    def __post_init__(self) -> None:
        if not isfinite(self.tolerance_seconds) or self.tolerance_seconds < 0:
            raise InvalidAlignmentConfigurationError("Alignment tolerance must be finite and non-negative.")
        if not isfinite(self.max_speed_up) or self.max_speed_up < 1:
            raise InvalidAlignmentConfigurationError("Maximum speed-up must be finite and at least 1.0.")


@dataclass(frozen=True, slots=True)
class TimingAlignmentInstruction:
    """A deterministic processing plan for one generated audio segment.

    The source file is not modified. A later audio-processing backend can apply
    ``playback_rate``, then ``trim_end_seconds``, then ``pad_end_seconds``.
    """

    segment_id: str
    source_audio_path: Path
    timeline_start: float
    timeline_end: float
    source_duration: float
    target_duration: float
    difference_seconds: float
    action: AlignmentAction
    playback_rate: float = 1.0
    pad_end_seconds: float = 0.0
    trim_end_seconds: float = 0.0
    expected_duration: float = 0.0
    target_language: str = ""
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.segment_id).strip():
            raise InvalidAlignmentSegmentError("Alignment instruction requires a segment ID.")
        numeric_values = (
            self.timeline_start,
            self.timeline_end,
            self.source_duration,
            self.target_duration,
            self.difference_seconds,
            self.playback_rate,
            self.pad_end_seconds,
            self.trim_end_seconds,
            self.expected_duration,
        )
        if not all(isfinite(value) for value in numeric_values):
            raise InvalidAlignmentSegmentError("Alignment values must be finite numbers.")
        if self.timeline_start < 0 or self.timeline_end <= self.timeline_start:
            raise InvalidAlignmentSegmentError("Target timestamps must define a positive, ordered slot.")
        if self.source_duration <= 0 or self.target_duration <= 0:
            raise InvalidAlignmentSegmentError("Source and target durations must be greater than zero.")
        if self.playback_rate < 1:
            raise InvalidAlignmentSegmentError("Timing alignment cannot slow audio below its generated rate.")
        if self.pad_end_seconds < 0 or self.trim_end_seconds < 0:
            raise InvalidAlignmentSegmentError("Padding and trimming values cannot be negative.")
        if self.expected_duration <= 0:
            raise InvalidAlignmentSegmentError("Expected aligned duration must be greater than zero.")


@dataclass(frozen=True, slots=True)
class TimingAlignmentResult:
    """Complete plan for later audio processing and timeline placement."""

    instructions: tuple[TimingAlignmentInstruction, ...]
    tolerance_seconds: float
    max_speed_up: float

    @property
    def timeline_duration(self) -> float:
        """Return the end time of the last planned segment."""
        return max((instruction.timeline_end for instruction in self.instructions), default=0.0)


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    """Measured properties of one local audio stream."""

    duration: float
    sample_rate: int
    channels: int
    channel_layout: str | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.duration) or self.duration <= 0:
            raise InvalidAudioProcessingInputError("Audio duration must be a finite value greater than zero.")
        if self.sample_rate <= 0:
            raise InvalidAudioProcessingInputError("Audio sample rate must be greater than zero.")
        if self.channels <= 0:
            raise InvalidAudioProcessingInputError("Audio channel count must be greater than zero.")


@dataclass(frozen=True, slots=True)
class ProcessedAudioSegment:
    """A non-destructive aligned WAV ready for later timeline placement."""

    segment_id: str
    source_audio_path: Path
    processed_audio_path: Path
    duration: float
    timeline_start: float
    timeline_end: float
    target_language: str
    sample_rate: int
    channels: int
    channel_layout: str | None = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    processing_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.segment_id).strip():
            raise InvalidAudioProcessingInputError("Processed audio requires a segment ID.")
        if not all(isfinite(value) for value in (self.duration, self.timeline_start, self.timeline_end)):
            raise InvalidAudioProcessingInputError("Processed audio timing values must be finite.")
        if self.duration <= 0:
            raise InvalidAudioProcessingInputError("Processed audio duration must be greater than zero.")
        if self.timeline_start < 0 or self.timeline_end <= self.timeline_start:
            raise InvalidAudioProcessingInputError("Processed audio timestamps must define a positive slot.")
        if self.sample_rate <= 0 or self.channels <= 0:
            raise InvalidAudioProcessingInputError("Processed audio must retain valid stream information.")


@dataclass(frozen=True, slots=True)
class AudioProcessingResult:
    """Processed clips prepared for a future timeline mixer."""

    segments: tuple[ProcessedAudioSegment, ...]
    processor_name: str

    @property
    def timeline_duration(self) -> float:
        """Return the timeline end needed by a future mixer."""
        return max((segment.timeline_end for segment in self.segments), default=0.0)
