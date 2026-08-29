"""Unit tests for Phase 9 audio processing without models or real speech."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from local_dubbing.audio import (
    AlignmentAction,
    AudioProcessingFailedError,
    AudioStreamInfo,
    DefaultTimingAlignmentEngine,
    FFmpegAudioTimingProcessor,
    InvalidAlignmentSegmentError,
    InvalidAudioProcessingInputError,
    MissingAudioProcessingDependencyError,
    TimingAlignmentConfig,
)
from local_dubbing.tts.models import GeneratedAudioSegment


def _plan(
    source_path: Path,
    duration: float,
    *,
    target_duration: float = 2.0,
    tolerance: float = 0.05,
    max_speed_up: float = 1.5,
):
    source_path.write_bytes(b"source wav remains unchanged")
    segment = GeneratedAudioSegment(
        segment_id="line/one",
        audio_path=source_path,
        duration=duration,
        start=1.25,
        end=1.25 + target_duration,
        target_language="id",
        metadata={"backend": "fake-tts", "sample_rate": 48_000},
    )
    return DefaultTimingAlignmentEngine().plan(
        (segment,),
        TimingAlignmentConfig(tolerance_seconds=tolerance, max_speed_up=max_speed_up),
    )


def _processor(
    source_path: Path,
    source_duration: float,
    output_duration: float,
    *,
    source_info: AudioStreamInfo | None = None,
    output_info: AudioStreamInfo | None = None,
):
    commands: list[list[str]] = []
    source_stream = source_info or AudioStreamInfo(source_duration, 48_000, 2, "stereo")
    output_stream = output_info or AudioStreamInfo(output_duration, 48_000, 2, "stereo")

    def probe(path: Path) -> AudioStreamInfo:
        return source_stream if path == source_path else output_stream

    def run(command: list[str]) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"processed wav")

    return FFmpegAudioTimingProcessor(command_runner=run, audio_probe=probe), commands


def test_exact_duration_audio_is_copied_without_ffmpeg(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    plan = _plan(source, 2.0)
    processor, commands = _processor(source, 2.0, 2.0)

    segment = processor.process(plan, tmp_path / "processed").segments[0]

    assert plan.instructions[0].action is AlignmentAction.KEEP
    assert commands == []
    assert segment.source_audio_path == source
    assert segment.processed_audio_path != source
    assert segment.processed_audio_path.read_bytes() == source.read_bytes()
    assert segment.duration == pytest.approx(2.0)
    assert source.read_bytes() == b"source wav remains unchanged"


def test_short_audio_is_padded_to_target_duration(tmp_path: Path) -> None:
    source = tmp_path / "short.wav"
    plan = _plan(source, 1.25)
    processor, commands = _processor(source, 1.25, 2.0)

    segment = processor.process(plan, tmp_path / "processed").segments[0]
    filter_chain = commands[0][commands[0].index("-af") + 1]

    assert plan.instructions[0].action is AlignmentAction.PAD_END
    assert "apad=pad_dur=2" in filter_chain
    assert "atrim=duration=2" in filter_chain
    assert "atempo" not in filter_chain
    assert segment.duration == pytest.approx(2.0)
    assert segment.processing_metadata["pad_end_seconds"] == pytest.approx(0.75)


def test_moderately_long_audio_uses_pitch_preserving_atempo(tmp_path: Path) -> None:
    source = tmp_path / "moderate.wav"
    plan = _plan(source, 2.4)
    processor, commands = _processor(source, 2.4, 2.0)

    segment = processor.process(plan, tmp_path / "processed").segments[0]
    filter_chain = commands[0][commands[0].index("-af") + 1]

    assert plan.instructions[0].action is AlignmentAction.SPEED_UP
    assert filter_chain.startswith("atempo=1.2,")
    assert segment.processing_metadata["pitch_preserving"] is True
    assert segment.processing_metadata["playback_rate"] == pytest.approx(1.2)


def test_excessively_long_audio_obeys_speed_limit_and_trims(tmp_path: Path) -> None:
    source = tmp_path / "long.wav"
    plan = _plan(source, 4.0, max_speed_up=1.5)
    processor, commands = _processor(source, 4.0, 2.0)

    segment = processor.process(plan, tmp_path / "processed").segments[0]
    instruction = plan.instructions[0]
    filter_chain = commands[0][commands[0].index("-af") + 1]

    assert instruction.action is AlignmentAction.SPEED_UP_AND_TRIM
    assert instruction.playback_rate == pytest.approx(1.5)
    assert instruction.trim_end_seconds == pytest.approx(2 / 3)
    assert filter_chain.startswith("atempo=1.5,")
    assert "atrim=duration=2" in filter_chain
    assert segment.duration == pytest.approx(2.0)


def test_boundary_conditions_keep_tolerance_and_allow_exact_maximum_speed(tmp_path: Path) -> None:
    within_tolerance = tmp_path / "within.wav"
    keep_plan = _plan(within_tolerance, 2.04, tolerance=0.05)
    assert keep_plan.instructions[0].action is AlignmentAction.KEEP

    exact_limit = tmp_path / "limit.wav"
    speed_plan = _plan(exact_limit, 3.0, max_speed_up=1.5)
    assert speed_plan.instructions[0].action is AlignmentAction.SPEED_UP
    assert speed_plan.instructions[0].playback_rate == pytest.approx(1.5)
    assert speed_plan.instructions[0].trim_end_seconds == 0


def test_metadata_timeline_and_stream_information_are_preserved(tmp_path: Path) -> None:
    source = tmp_path / "metadata.wav"
    plan = _plan(source, 1.0)
    stream = AudioStreamInfo(1.0, 44_100, 2, "stereo")
    output_stream = AudioStreamInfo(2.0, 44_100, 2, "stereo")
    processor, commands = _processor(
        source,
        1.0,
        2.0,
        source_info=stream,
        output_info=output_stream,
    )

    result = processor.process(plan, tmp_path / "processed")
    segment = result.segments[0]

    assert segment.segment_id == "line/one"
    assert (segment.timeline_start, segment.timeline_end) == pytest.approx((1.25, 3.25))
    assert segment.target_language == "id"
    assert (segment.sample_rate, segment.channels, segment.channel_layout) == (44_100, 2, "stereo")
    assert segment.source_metadata == {"backend": "fake-tts", "sample_rate": 48_000}
    assert segment.processing_metadata["alignment_action"] == "pad_end"
    assert result.timeline_duration == pytest.approx(3.25)
    assert commands[0][commands[0].index("-ar") + 1] == "44100"
    assert commands[0][commands[0].index("-ac") + 1] == "2"
    assert commands[0][commands[0].index("-channel_layout") + 1] == "stereo"


def test_invalid_zero_duration_and_missing_inputs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidAlignmentSegmentError, match="greater than zero"):
        _plan(tmp_path / "zero.wav", 0)
    with pytest.raises(InvalidAudioProcessingInputError, match="greater than zero"):
        AudioStreamInfo(0, 48_000, 1)

    source = tmp_path / "missing.wav"
    plan = _plan(source, 1.0)
    source.unlink()
    processor, _ = _processor(source, 1.0, 2.0)
    with pytest.raises(InvalidAudioProcessingInputError, match="missing or empty"):
        processor.process(plan, tmp_path / "processed")


def test_stale_duration_and_changed_stream_information_fail_safely(tmp_path: Path) -> None:
    source = tmp_path / "stale.wav"
    plan = _plan(source, 1.0)
    stale_processor, _ = _processor(
        source,
        1.0,
        2.0,
        source_info=AudioStreamInfo(1.2, 48_000, 1, "mono"),
    )
    with pytest.raises(InvalidAudioProcessingInputError, match="no longer matches"):
        stale_processor.process(plan, tmp_path / "stale-output")

    changed_processor, _ = _processor(
        source,
        1.0,
        2.0,
        source_info=AudioStreamInfo(1.0, 48_000, 1, "mono"),
        output_info=AudioStreamInfo(2.0, 44_100, 2, "stereo"),
    )
    with pytest.raises(AudioProcessingFailedError, match="sample rate and channel"):
        changed_processor.process(plan, tmp_path / "changed-output")


def test_large_playback_rates_are_split_into_supported_atempo_factors() -> None:
    assert FFmpegAudioTimingProcessor._atempo_filters(4.5) == (
        "atempo=2",
        "atempo=2",
        "atempo=1.125",
    )


def test_missing_ffmpeg_is_reported_without_model_loading(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    plan = _plan(source, 1.0)

    def missing_command(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", missing_command)
    processor = FFmpegAudioTimingProcessor(
        audio_probe=lambda path: AudioStreamInfo(1.0, 48_000, 1, "mono")
    )
    with pytest.raises(MissingAudioProcessingDependencyError, match="FFmpeg is not available"):
        processor.process(plan, tmp_path / "processed")
