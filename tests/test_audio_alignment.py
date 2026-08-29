"""Unit tests for Phase 8 backend-neutral timing alignment."""

from pathlib import Path

import pytest

from local_dubbing.audio import (
    AlignmentAction,
    DefaultTimingAlignmentEngine,
    InvalidAlignmentConfigurationError,
    InvalidAlignmentSegmentError,
    TimingAlignmentConfig,
)
from local_dubbing.tts.models import GeneratedAudioSegment


def _segment(
    duration: float,
    *,
    segment_id: str = "segment-0001",
    start: float = 2.0,
    end: float = 5.0,
) -> GeneratedAudioSegment:
    return GeneratedAudioSegment(
        segment_id=segment_id,
        audio_path=Path(f"{segment_id}.wav"),
        duration=duration,
        start=start,
        end=end,
        target_language="id",
        metadata={"sample_rate": 48_000},
    )


def test_alignment_configuration_validation() -> None:
    assert TimingAlignmentConfig().tolerance_seconds == 0.05
    with pytest.raises(InvalidAlignmentConfigurationError):
        TimingAlignmentConfig(tolerance_seconds=-0.01)
    with pytest.raises(InvalidAlignmentConfigurationError):
        TimingAlignmentConfig(max_speed_up=0.99)
    with pytest.raises(InvalidAlignmentConfigurationError):
        TimingAlignmentConfig(tolerance_seconds=float("nan"))
    with pytest.raises(InvalidAlignmentConfigurationError):
        TimingAlignmentConfig(max_speed_up=float("inf"))


def test_matching_duration_is_kept_without_processing() -> None:
    instruction = DefaultTimingAlignmentEngine().plan((_segment(3.0),)).instructions[0]
    assert instruction.action is AlignmentAction.KEEP
    assert instruction.target_duration == pytest.approx(3.0)
    assert instruction.difference_seconds == pytest.approx(0.0)
    assert instruction.playback_rate == pytest.approx(1.0)
    assert instruction.expected_duration == pytest.approx(3.0)


def test_difference_within_tolerance_is_kept() -> None:
    instruction = DefaultTimingAlignmentEngine().plan(
        (_segment(3.04),), TimingAlignmentConfig(tolerance_seconds=0.05)
    ).instructions[0]
    assert instruction.action is AlignmentAction.KEEP
    assert instruction.expected_duration == pytest.approx(3.04)


def test_short_audio_is_padded_at_the_end() -> None:
    instruction = DefaultTimingAlignmentEngine().plan((_segment(2.25),)).instructions[0]
    assert instruction.action is AlignmentAction.PAD_END
    assert instruction.pad_end_seconds == pytest.approx(0.75)
    assert instruction.trim_end_seconds == 0
    assert instruction.expected_duration == pytest.approx(3.0)


def test_long_audio_is_sped_up_to_fit_target_slot() -> None:
    instruction = DefaultTimingAlignmentEngine().plan((_segment(3.6),)).instructions[0]
    assert instruction.action is AlignmentAction.SPEED_UP
    assert instruction.playback_rate == pytest.approx(1.2)
    assert instruction.trim_end_seconds == 0
    assert instruction.expected_duration == pytest.approx(3.0)


def test_extreme_overrun_caps_speed_and_trims_remainder() -> None:
    instruction = DefaultTimingAlignmentEngine().plan(
        (_segment(6.0),), TimingAlignmentConfig(max_speed_up=1.5)
    ).instructions[0]
    assert instruction.action is AlignmentAction.SPEED_UP_AND_TRIM
    assert instruction.playback_rate == pytest.approx(1.5)
    assert instruction.trim_end_seconds == pytest.approx(1.0)
    assert instruction.expected_duration == pytest.approx(3.0)


def test_plan_preserves_timeline_path_language_and_metadata() -> None:
    result = DefaultTimingAlignmentEngine().plan(
        (_segment(2.5, segment_id="first", start=1.25, end=4.25), _segment(1.0, segment_id="second", start=8, end=9))
    )
    first = result.instructions[0]
    assert first.segment_id == "first"
    assert first.source_audio_path == Path("first.wav")
    assert (first.timeline_start, first.timeline_end) == (1.25, 4.25)
    assert first.target_language == "id"
    assert first.source_metadata["sample_rate"] == 48_000
    assert result.timeline_duration == pytest.approx(9.0)


def test_empty_plan_is_valid_and_has_zero_timeline_duration() -> None:
    result = DefaultTimingAlignmentEngine().plan(())
    assert result.instructions == ()
    assert result.timeline_duration == 0


def test_duplicate_ids_and_non_positive_durations_are_rejected() -> None:
    engine = DefaultTimingAlignmentEngine()
    with pytest.raises(InvalidAlignmentSegmentError, match="unique"):
        engine.plan((_segment(1.0, segment_id="same"), _segment(1.0, segment_id="same", start=6, end=7)))
    with pytest.raises(InvalidAlignmentSegmentError, match="duration"):
        engine.plan((_segment(0),))
    with pytest.raises(InvalidAlignmentSegmentError, match="positive slot"):
        engine.plan((_segment(1.0, start=2, end=2),))
    with pytest.raises(InvalidAlignmentSegmentError, match="finite"):
        engine.plan((_segment(float("nan")),))
    with pytest.raises(InvalidAlignmentSegmentError, match="finite"):
        engine.plan((_segment(1.0, end=float("inf")),))
