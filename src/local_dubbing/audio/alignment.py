"""Backend-neutral timing calculations for generated TTS audio."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from math import isfinite

from local_dubbing.tts.models import GeneratedAudioSegment

from .models import (
    AlignmentAction,
    InvalidAlignmentSegmentError,
    TimingAlignmentConfig,
    TimingAlignmentInstruction,
    TimingAlignmentResult,
)


class TimingAlignmentEngine(ABC):
    """Stable interface for turning generated audio into alignment plans."""

    @abstractmethod
    def plan(
        self,
        segments: Iterable[GeneratedAudioSegment],
        config: TimingAlignmentConfig | None = None,
    ) -> TimingAlignmentResult:
        """Calculate non-destructive timing instructions for generated audio."""


class DefaultTimingAlignmentEngine(TimingAlignmentEngine):
    """Plan padding, bounded speed-up, and final trimming without processing files."""

    def plan(
        self,
        segments: Iterable[GeneratedAudioSegment],
        config: TimingAlignmentConfig | None = None,
    ) -> TimingAlignmentResult:
        active_config = config or TimingAlignmentConfig()
        materialized_segments = tuple(segments)
        self._validate_segments(materialized_segments)
        instructions = tuple(self._plan_segment(segment, active_config) for segment in materialized_segments)
        return TimingAlignmentResult(
            instructions=instructions,
            tolerance_seconds=active_config.tolerance_seconds,
            max_speed_up=active_config.max_speed_up,
        )

    @staticmethod
    def _validate_segments(segments: tuple[GeneratedAudioSegment, ...]) -> None:
        segment_ids = [segment.segment_id for segment in segments]
        if len(set(segment_ids)) != len(segment_ids):
            raise InvalidAlignmentSegmentError("Generated audio segment IDs must be unique.")
        for segment in segments:
            if not all(isfinite(value) for value in (segment.duration, segment.start, segment.end)):
                raise InvalidAlignmentSegmentError(
                    f"Timing values for segment '{segment.segment_id}' must be finite numbers."
                )
            if segment.duration <= 0:
                raise InvalidAlignmentSegmentError(
                    f"Generated audio duration for segment '{segment.segment_id}' must be greater than zero."
                )
            if segment.end <= segment.start:
                raise InvalidAlignmentSegmentError(
                    f"Target timestamps for segment '{segment.segment_id}' must define a positive slot."
                )

    @staticmethod
    def _plan_segment(
        segment: GeneratedAudioSegment,
        config: TimingAlignmentConfig,
    ) -> TimingAlignmentInstruction:
        target_duration = segment.end - segment.start
        difference = segment.duration - target_duration
        playback_rate = 1.0
        pad_end = 0.0
        trim_end = 0.0
        expected_duration = segment.duration

        if abs(difference) <= config.tolerance_seconds:
            action = AlignmentAction.KEEP
        elif difference < 0:
            action = AlignmentAction.PAD_END
            pad_end = -difference
            expected_duration = target_duration
        else:
            required_speed_up = segment.duration / target_duration
            playback_rate = min(required_speed_up, config.max_speed_up)
            stretched_duration = segment.duration / playback_rate
            if required_speed_up <= config.max_speed_up:
                action = AlignmentAction.SPEED_UP
                expected_duration = target_duration
            else:
                action = AlignmentAction.SPEED_UP_AND_TRIM
                trim_end = stretched_duration - target_duration
                expected_duration = target_duration

        return TimingAlignmentInstruction(
            segment_id=segment.segment_id,
            source_audio_path=segment.audio_path,
            timeline_start=segment.start,
            timeline_end=segment.end,
            source_duration=segment.duration,
            target_duration=target_duration,
            difference_seconds=difference,
            action=action,
            playback_rate=playback_rate,
            pad_end_seconds=pad_end,
            trim_end_seconds=trim_end,
            expected_duration=expected_duration,
            target_language=segment.target_language,
            source_metadata=segment.metadata,
        )
