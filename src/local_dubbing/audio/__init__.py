"""Backend-neutral timing plans and local processing for generated speech."""

from .alignment import DefaultTimingAlignmentEngine, TimingAlignmentEngine
from .models import (
    AlignmentAction,
    AudioProcessingError,
    AudioProcessingFailedError,
    AudioProcessingResult,
    AudioStreamInfo,
    InvalidAlignmentConfigurationError,
    InvalidAlignmentSegmentError,
    InvalidAudioProcessingInputError,
    MissingAudioProcessingDependencyError,
    ProcessedAudioSegment,
    TimingAlignmentConfig,
    TimingAlignmentError,
    TimingAlignmentInstruction,
    TimingAlignmentResult,
)
from .processing import AudioTimingProcessor, FFmpegAudioTimingProcessor

__all__ = [
    "AlignmentAction",
    "AudioProcessingError",
    "AudioProcessingFailedError",
    "AudioProcessingResult",
    "AudioStreamInfo",
    "AudioTimingProcessor",
    "DefaultTimingAlignmentEngine",
    "FFmpegAudioTimingProcessor",
    "InvalidAlignmentConfigurationError",
    "InvalidAlignmentSegmentError",
    "InvalidAudioProcessingInputError",
    "MissingAudioProcessingDependencyError",
    "ProcessedAudioSegment",
    "TimingAlignmentConfig",
    "TimingAlignmentEngine",
    "TimingAlignmentError",
    "TimingAlignmentInstruction",
    "TimingAlignmentResult",
]
