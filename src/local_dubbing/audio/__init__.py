"""Backend-neutral timing alignment for generated speech audio."""

from .alignment import DefaultTimingAlignmentEngine, TimingAlignmentEngine
from .models import (
    AlignmentAction,
    InvalidAlignmentConfigurationError,
    InvalidAlignmentSegmentError,
    TimingAlignmentConfig,
    TimingAlignmentError,
    TimingAlignmentInstruction,
    TimingAlignmentResult,
)

__all__ = [
    "AlignmentAction",
    "DefaultTimingAlignmentEngine",
    "InvalidAlignmentConfigurationError",
    "InvalidAlignmentSegmentError",
    "TimingAlignmentConfig",
    "TimingAlignmentEngine",
    "TimingAlignmentError",
    "TimingAlignmentInstruction",
    "TimingAlignmentResult",
]
