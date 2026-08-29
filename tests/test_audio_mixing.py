"""Unit tests for Phase 10 timeline mixing without models or real speech."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from local_dubbing.audio import (
    AudioMixingConfig,
    AudioMixingFailedError,
    AudioProcessingResult,
    AudioStreamInfo,
    FFmpegAudioTimelineMixer,
    InvalidAudioMixingConfigurationError,
    InvalidAudioMixingInputError,
    MissingAudioMixingDependencyError,
    ProcessedAudioSegment,
)


def _segment(
    tmp_path: Path,
    segment_id: str,
    start: float,
    end: float,
    *,
    sample_rate: int = 48_000,
    channels: int = 2,
) -> ProcessedAudioSegment:
    path = tmp_path / f"{segment_id}.wav"
    path.write_bytes(f"synthetic {segment_id}".encode())
    return ProcessedAudioSegment(
        segment_id=segment_id,
        source_audio_path=tmp_path / f"tts-{segment_id}.wav",
        processed_audio_path=path,
        duration=end - start,
        timeline_start=start,
        timeline_end=end,
        target_language="id",
        sample_rate=sample_rate,
        channels=channels,
        channel_layout="mono" if channels == 1 else "stereo",
        source_metadata={"backend": "fake-tts", "position": segment_id},
        processing_metadata={"alignment_action": "keep"},
    )


def _result(*segments: ProcessedAudioSegment) -> AudioProcessingResult:
    return AudioProcessingResult(tuple(segments), "fake Phase 9 processor")


def _fake_mixer(
    output_duration: float,
    *,
    source_path: Path | None = None,
    source_duration: float | None = None,
    sample_rate: int = 48_000,
    channels: int = 2,
):
    commands: list[list[str]] = []

    def probe(path: Path) -> AudioStreamInfo:
        if source_path is not None and path == source_path:
            assert source_duration is not None
            return AudioStreamInfo(source_duration, 44_100, 2, "stereo")
        return AudioStreamInfo(output_duration, sample_rate, channels, "mono" if channels == 1 else "stereo")

    def run(command: list[str]) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"mixed synthetic wav")

    return FFmpegAudioTimelineMixer(command_runner=run, audio_probe=probe), commands


def _filter_graph(command: list[str]) -> str:
    return command[command.index("-filter_complex") + 1]


def test_one_segment_at_timeline_beginning_is_placed_at_zero(tmp_path: Path) -> None:
    segment = _segment(tmp_path, "opening", 0.0, 1.5, channels=1)
    plan = FFmpegAudioTimelineMixer(audio_probe=lambda path: AudioStreamInfo(1.5, 48_000, 1)).plan(
        _result(segment)
    )

    assert plan.timeline_duration == pytest.approx(1.5)
    assert plan.output_channels == 1
    assert plan.placements[0].delay_milliseconds == 0
    assert plan.placements[0].timeline_end == pytest.approx(1.5)


def test_multiple_sequential_segments_preserve_input_order_and_indices(tmp_path: Path) -> None:
    segments = (
        _segment(tmp_path, "first", 0.0, 1.0),
        _segment(tmp_path, "second", 1.0, 2.5),
        _segment(tmp_path, "third", 2.5, 4.0),
    )
    plan = FFmpegAudioTimelineMixer().plan(_result(*segments))

    assert [placement.segment_id for placement in plan.placements] == ["first", "second", "third"]
    assert [placement.input_index for placement in plan.placements] == [0, 1, 2]
    assert [placement.source_order for placement in plan.placements] == [0, 1, 2]
    assert [placement.delay_milliseconds for placement in plan.placements] == [0, 1_000, 2_500]


def test_gaps_are_preserved_by_silent_base_and_timestamp_delays(tmp_path: Path) -> None:
    segments = (_segment(tmp_path, "early", 1.0, 2.0), _segment(tmp_path, "late", 5.0, 6.0))
    mixer, commands = _fake_mixer(6.0)

    result = mixer.mix(_result(*segments), tmp_path / "mixed")
    graph = _filter_graph(commands[0])

    assert "anullsrc=r=48000:cl=stereo" in graph
    assert "adelay=1000:all=1[speech0]" in graph
    assert "adelay=5000:all=1[speech1]" in graph
    assert result.timeline_duration == pytest.approx(6.0)
    assert result.metadata["silence_policy"] == "explicit_silent_base"


def test_overlapping_segments_are_summed_safely_and_deterministically(tmp_path: Path) -> None:
    segments = (_segment(tmp_path, "under", 1.0, 3.0), _segment(tmp_path, "over", 2.0, 4.0))
    mixer = FFmpegAudioTimelineMixer()

    first_plan = mixer.plan(_result(*segments))
    second_plan = mixer.plan(_result(*segments))
    first_graph = mixer._filter_graph(first_plan)
    second_graph = mixer._filter_graph(second_plan)

    assert first_plan == second_plan
    assert first_graph == second_graph
    assert "[silence][speech0][speech1]amix=inputs=3" in first_graph
    assert "normalize=0" in first_graph
    assert "alimiter=limit=0.95" in first_graph


def test_input_order_is_preserved_even_when_timestamps_are_not_sorted(tmp_path: Path) -> None:
    segments = (_segment(tmp_path, "later-first", 4.0, 5.0), _segment(tmp_path, "earlier-second", 0.0, 1.0))
    plan = FFmpegAudioTimelineMixer().plan(_result(*segments))

    assert [item.segment_id for item in plan.placements] == ["later-first", "earlier-second"]
    assert [item.delay_milliseconds for item in plan.placements] == [4_000, 0]
    assert plan.timeline_duration == pytest.approx(5.0)


def test_total_duration_uses_last_timeline_end_without_source_audio(tmp_path: Path) -> None:
    segments = (_segment(tmp_path, "first", 2.0, 3.0), _segment(tmp_path, "final", 8.5, 10.0))
    plan = FFmpegAudioTimelineMixer().plan(_result(*segments), AudioMixingConfig(include_source_audio=False))

    assert plan.timeline_duration == pytest.approx(10.0)
    assert plan.source_audio_path is None
    assert plan.source_audio_duration is None


def test_source_audio_disabled_does_not_probe_or_add_source_input(tmp_path: Path) -> None:
    segment = _segment(tmp_path, "speech", 0.0, 2.0)
    mixer, commands = _fake_mixer(2.0)
    unused_source = tmp_path / "does-not-exist.mp4"

    result = mixer.mix(_result(segment), tmp_path / "mixed", source_audio_path=unused_source)

    assert result.source_audio_path is None
    assert result.metadata["source_audio_included"] is False
    assert commands[0].count("-i") == 1
    assert commands[0][commands[0].index("-map") + 1] == "[dubbed]"


def test_source_audio_enabled_extends_timeline_and_applies_volume(tmp_path: Path) -> None:
    segment = _segment(tmp_path, "speech", 0.0, 2.0)
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original media remains unchanged")
    mixer, commands = _fake_mixer(5.0, source_path=source, source_duration=5.0)

    result = mixer.mix(
        _result(segment),
        tmp_path / "mixed",
        AudioMixingConfig(include_source_audio=True, source_volume=0.35),
        source,
    )
    graph = _filter_graph(commands[0])

    assert result.timeline_duration == pytest.approx(5.0)
    assert result.source_audio_path == source
    assert "volume=0.35" in graph
    assert "[source][dubbed]amix=inputs=2" in graph
    assert commands[0].count("-i") == 2
    assert source.read_bytes() == b"original media remains unchanged"


def test_source_ducking_uses_dubbed_timeline_as_sidechain(tmp_path: Path) -> None:
    segment = _segment(tmp_path, "speech", 1.0, 3.0)
    source = tmp_path / "original.wav"
    source.write_bytes(b"source audio")
    mixer, commands = _fake_mixer(4.0, source_path=source, source_duration=4.0)
    config = AudioMixingConfig(
        include_source_audio=True,
        source_volume=0.6,
        duck_source_audio=True,
        ducking_threshold=0.08,
        ducking_ratio=10.0,
        ducking_attack_ms=15.0,
        ducking_release_ms=300.0,
    )

    result = mixer.mix(_result(segment), tmp_path / "mixed", config, source)
    graph = _filter_graph(commands[0])

    assert "[dubbed]asplit=2[dubbed_mix][duck_key]" in graph
    assert "[source][duck_key]sidechaincompress=threshold=0.08:ratio=10:attack=15:release=300" in graph
    assert "[ducked_source][dubbed_mix]amix=inputs=2" in graph
    assert result.metadata["source_ducking"] is True
    assert result.metadata["ducking_ratio"] == pytest.approx(10.0)


def test_missing_empty_duplicate_and_stale_segments_are_rejected(tmp_path: Path) -> None:
    mixer = FFmpegAudioTimelineMixer()
    with pytest.raises(InvalidAudioMixingInputError, match="At least one"):
        mixer.plan(_result())

    missing = _segment(tmp_path, "missing", 0.0, 1.0)
    missing.processed_audio_path.unlink()
    with pytest.raises(InvalidAudioMixingInputError, match="missing or empty"):
        mixer.plan(_result(missing))

    duplicate_a = _segment(tmp_path, "duplicate", 0.0, 1.0)
    duplicate_b = ProcessedAudioSegment(
        segment_id="duplicate",
        source_audio_path=duplicate_a.source_audio_path,
        processed_audio_path=duplicate_a.processed_audio_path,
        duration=1.0,
        timeline_start=2.0,
        timeline_end=3.0,
        target_language="id",
        sample_rate=48_000,
        channels=2,
    )
    with pytest.raises(InvalidAudioMixingInputError, match="unique"):
        mixer.plan(_result(duplicate_a, duplicate_b))

    stale = ProcessedAudioSegment(
        segment_id="stale",
        source_audio_path=duplicate_a.source_audio_path,
        processed_audio_path=duplicate_a.processed_audio_path,
        duration=0.5,
        timeline_start=0.0,
        timeline_end=1.0,
        target_language="id",
        sample_rate=48_000,
        channels=2,
    )
    with pytest.raises(InvalidAudioMixingInputError, match="timeline slot"):
        mixer.plan(_result(stale))


def test_source_audio_configuration_and_presence_are_validated(tmp_path: Path) -> None:
    segment = _segment(tmp_path, "speech", 0.0, 1.0)
    with pytest.raises(InvalidAudioMixingConfigurationError, match="requires original audio"):
        AudioMixingConfig(duck_source_audio=True)
    with pytest.raises(InvalidAudioMixingConfigurationError, match="volume"):
        AudioMixingConfig(source_volume=-0.1)
    with pytest.raises(InvalidAudioMixingConfigurationError, match="ratio"):
        AudioMixingConfig(include_source_audio=True, duck_source_audio=True, ducking_ratio=21)
    with pytest.raises(InvalidAudioMixingInputError, match="Original media"):
        FFmpegAudioTimelineMixer().plan(_result(segment), AudioMixingConfig(include_source_audio=True))


def test_result_preserves_metadata_order_stream_and_output_contract(tmp_path: Path) -> None:
    segments = (_segment(tmp_path, "one", 0.0, 1.0), _segment(tmp_path, "two", 1.0, 3.0))
    mixer, commands = _fake_mixer(3.0)

    result = mixer.mix(_result(*segments), tmp_path / "mixed")

    assert result.output_audio_path.name == "mixed_dubbed_audio.wav"
    assert result.duration == pytest.approx(3.0)
    assert (result.sample_rate, result.channels) == (48_000, 2)
    assert [item.segment_id for item in result.placements] == ["one", "two"]
    assert result.placements[1].source_metadata["position"] == "two"
    assert result.metadata["segment_ids"] == ("one", "two")
    assert commands[0][-3:] == ["pcm_s16le", str(result.output_audio_path)] or commands[0][-2:] == ["pcm_s16le", str(result.output_audio_path)]


def test_unexpected_output_duration_or_stream_fails_safely(tmp_path: Path) -> None:
    segment = _segment(tmp_path, "speech", 0.0, 2.0)
    wrong_duration, _ = _fake_mixer(1.0)
    with pytest.raises(AudioMixingFailedError, match="unexpected total duration"):
        wrong_duration.mix(_result(segment), tmp_path / "wrong-duration")

    wrong_stream, _ = _fake_mixer(2.0, sample_rate=44_100, channels=1)
    with pytest.raises(AudioMixingFailedError, match="sample rate and channel"):
        wrong_stream.mix(_result(segment), tmp_path / "wrong-stream")


def test_missing_ffmpeg_is_reported_without_loading_models(monkeypatch, tmp_path: Path) -> None:
    segment = _segment(tmp_path, "speech", 0.0, 1.0)

    def missing_command(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", missing_command)
    mixer = FFmpegAudioTimelineMixer(audio_probe=lambda path: AudioStreamInfo(1.0, 48_000, 2, "stereo"))
    with pytest.raises(MissingAudioMixingDependencyError, match="FFmpeg is not available"):
        mixer.mix(_result(segment), tmp_path / "mixed")
