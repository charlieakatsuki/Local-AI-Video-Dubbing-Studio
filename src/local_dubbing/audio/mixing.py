"""Deterministic local timeline placement and mixing for Phase 9 audio."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import json
from pathlib import Path
import subprocess
from typing import Any

from .models import (
    AudioMixingConfig,
    AudioMixingFailedError,
    AudioMixingPlan,
    AudioProcessingResult,
    AudioStreamInfo,
    InvalidAudioMixingInputError,
    MissingAudioMixingDependencyError,
    MixedAudioResult,
    ProcessedAudioSegment,
    TimelinePlacement,
)

ProgressCallback = Callable[[str], None]
CommandRunner = Callable[[list[str]], None]
AudioProbe = Callable[[Path], AudioStreamInfo]


class AudioTimelineMixer(ABC):
    """Backend boundary for planning and creating a continuous soundtrack."""

    name: str

    @abstractmethod
    def plan(
        self,
        processed_audio: AudioProcessingResult,
        config: AudioMixingConfig | None = None,
        source_audio_path: Path | None = None,
    ) -> AudioMixingPlan:
        """Create a deterministic, non-destructive placement plan."""

    @abstractmethod
    def mix(
        self,
        processed_audio: AudioProcessingResult,
        output_dir: Path,
        config: AudioMixingConfig | None = None,
        source_audio_path: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> MixedAudioResult:
        """Render a new continuous WAV without changing any input file."""


class FFmpegAudioTimelineMixer(AudioTimelineMixer):
    """Place and combine aligned speech with an optional original soundtrack."""

    name = "FFmpeg timeline mixer (local)"

    def __init__(
        self,
        command_runner: CommandRunner | None = None,
        audio_probe: AudioProbe | None = None,
    ) -> None:
        self._command_runner = command_runner or self._run_command
        self._audio_probe = audio_probe or self._probe_audio

    def plan(
        self,
        processed_audio: AudioProcessingResult,
        config: AudioMixingConfig | None = None,
        source_audio_path: Path | None = None,
    ) -> AudioMixingPlan:
        active_config = config or AudioMixingConfig()
        segments = tuple(processed_audio.segments)
        self._validate_segments(segments)
        self._validate_source_request(active_config, source_audio_path)

        sample_rate = active_config.output_sample_rate or segments[0].sample_rate
        channels = active_config.output_channels or segments[0].channels
        if channels not in {1, 2}:
            raise InvalidAudioMixingInputError("Timeline output supports mono or stereo audio.")

        placements = tuple(
            TimelinePlacement(
                segment_id=segment.segment_id,
                input_index=index,
                source_order=index,
                audio_path=Path(segment.processed_audio_path),
                timeline_start=segment.timeline_start,
                timeline_end=segment.timeline_end,
                duration=segment.duration,
                delay_milliseconds=round(segment.timeline_start * 1000),
                source_metadata=dict(segment.source_metadata),
                processing_metadata=dict(segment.processing_metadata),
            )
            for index, segment in enumerate(segments)
        )
        speech_duration = max(segment.timeline_end for segment in segments)
        source_path: Path | None = None
        source_duration: float | None = None
        if active_config.include_source_audio:
            source_path = Path(source_audio_path)  # type: ignore[arg-type]
            source_info = self._audio_probe(source_path)
            source_duration = source_info.duration
        timeline_duration = max(speech_duration, source_duration or 0.0)

        return AudioMixingPlan(
            placements=placements,
            timeline_duration=timeline_duration,
            output_sample_rate=sample_rate,
            output_channels=channels,
            include_source_audio=active_config.include_source_audio,
            source_audio_path=source_path,
            source_audio_duration=source_duration,
            config=active_config,
        )

    def mix(
        self,
        processed_audio: AudioProcessingResult,
        output_dir: Path,
        config: AudioMixingConfig | None = None,
        source_audio_path: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> MixedAudioResult:
        if progress_callback:
            progress_callback("Planning deterministic timeline placement…")
        plan = self.plan(processed_audio, config, source_audio_path)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        output_path = destination / "mixed_dubbed_audio.wav"
        self._validate_output_path(plan, output_path)
        filter_graph = self._filter_graph(plan)
        command = self._ffmpeg_command(plan, output_path, filter_graph)

        if progress_callback:
            progress_callback("Mixing the continuous dubbed soundtrack with FFmpeg…")
        self._command_runner(command)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AudioMixingFailedError("Timeline mixing did not create a non-empty output WAV file.")
        output_info = self._audio_probe(output_path)
        self._validate_output(plan, output_info)
        if progress_callback:
            progress_callback("Timeline mixing complete.")

        metadata = {
            "mixer": "ffmpeg",
            "segment_count": len(plan.placements),
            "segment_ids": tuple(placement.segment_id for placement in plan.placements),
            "overlap_policy": "sum_in_source_order_then_limit",
            "silence_policy": "explicit_silent_base",
            "source_audio_included": plan.include_source_audio,
            "source_volume": plan.config.source_volume if plan.include_source_audio else None,
            "source_ducking": plan.config.duck_source_audio if plan.include_source_audio else False,
            "ducking_threshold": plan.config.ducking_threshold if plan.config.duck_source_audio else None,
            "ducking_ratio": plan.config.ducking_ratio if plan.config.duck_source_audio else None,
            "ducking_attack_ms": plan.config.ducking_attack_ms if plan.config.duck_source_audio else None,
            "ducking_release_ms": plan.config.ducking_release_ms if plan.config.duck_source_audio else None,
            "filter_graph": filter_graph,
        }
        return MixedAudioResult(
            output_audio_path=output_path,
            duration=output_info.duration,
            sample_rate=output_info.sample_rate,
            channels=output_info.channels,
            timeline_duration=plan.timeline_duration,
            placements=plan.placements,
            mixer_name=self.name,
            source_audio_path=plan.source_audio_path,
            metadata=metadata,
        )

    @staticmethod
    def _validate_segments(segments: tuple[ProcessedAudioSegment, ...]) -> None:
        if not segments:
            raise InvalidAudioMixingInputError("At least one processed audio segment is required for mixing.")
        segment_ids = [segment.segment_id for segment in segments]
        if len(set(segment_ids)) != len(segment_ids):
            raise InvalidAudioMixingInputError("Processed audio segment IDs must be unique.")
        for segment in segments:
            path = Path(segment.processed_audio_path)
            if not path.is_file() or path.stat().st_size == 0:
                raise InvalidAudioMixingInputError(
                    f"Processed WAV for segment '{segment.segment_id}' is missing or empty."
                )
            if path.suffix.lower() != ".wav":
                raise InvalidAudioMixingInputError(
                    f"Processed audio for segment '{segment.segment_id}' must be a WAV file."
                )
            slot_duration = segment.timeline_end - segment.timeline_start
            allowed_difference = max(0.05, 2 / segment.sample_rate)
            if abs(segment.duration - slot_duration) > allowed_difference:
                raise InvalidAudioMixingInputError(
                    f"Processed audio for segment '{segment.segment_id}' no longer matches its timeline slot."
                )

    @staticmethod
    def _validate_source_request(config: AudioMixingConfig, source_audio_path: Path | None) -> None:
        if not config.include_source_audio:
            return
        if source_audio_path is None:
            raise InvalidAudioMixingInputError("Original media is required when source-audio mixing is enabled.")
        source_path = Path(source_audio_path)
        if not source_path.is_file() or source_path.stat().st_size == 0:
            raise InvalidAudioMixingInputError("The original media file is missing or empty.")

    @staticmethod
    def _validate_output_path(plan: AudioMixingPlan, output_path: Path) -> None:
        resolved_output = output_path.resolve()
        input_paths = {placement.audio_path.resolve() for placement in plan.placements}
        if plan.source_audio_path is not None:
            input_paths.add(plan.source_audio_path.resolve())
        if resolved_output in input_paths:
            raise InvalidAudioMixingInputError("Mixed output must not overwrite any source audio file.")

    @classmethod
    def _filter_graph(cls, plan: AudioMixingPlan) -> str:
        duration = cls._number(plan.timeline_duration)
        layout = cls._channel_layout(plan.output_channels)
        filters = [
            f"anullsrc=r={plan.output_sample_rate}:cl={layout},"
            f"atrim=duration={duration},asetpts=PTS-STARTPTS[silence]"
        ]
        speech_labels = ["[silence]"]
        for placement in plan.placements:
            slot_duration = cls._number(placement.timeline_end - placement.timeline_start)
            label = f"speech{placement.source_order}"
            filters.append(
                f"[{placement.input_index}:a:0]"
                f"aformat=sample_rates={plan.output_sample_rate}:channel_layouts={layout},"
                f"apad=pad_dur={slot_duration},atrim=duration={slot_duration},asetpts=PTS-STARTPTS,"
                f"adelay={placement.delay_milliseconds}:all=1[{label}]"
            )
            speech_labels.append(f"[{label}]")
        filters.append(
            "".join(speech_labels)
            + f"amix=inputs={len(speech_labels)}:duration=first:dropout_transition=0:normalize=0,"
            f"alimiter=limit=0.95,atrim=duration={duration},asetpts=PTS-STARTPTS[dubbed]"
        )

        if not plan.include_source_audio:
            return ";".join(filters)

        source_index = len(plan.placements)
        filters.append(
            f"[{source_index}:a:0]"
            f"aformat=sample_rates={plan.output_sample_rate}:channel_layouts={layout},"
            f"volume={cls._number(plan.config.source_volume)},"
            f"apad=pad_dur={duration},atrim=duration={duration},asetpts=PTS-STARTPTS[source]"
        )
        if plan.config.duck_source_audio:
            filters.extend(
                (
                    "[dubbed]asplit=2[dubbed_mix][duck_key]",
                    "[source][duck_key]sidechaincompress="
                    f"threshold={cls._number(plan.config.ducking_threshold)}:"
                    f"ratio={cls._number(plan.config.ducking_ratio)}:"
                    f"attack={cls._number(plan.config.ducking_attack_ms)}:"
                    f"release={cls._number(plan.config.ducking_release_ms)}[ducked_source]",
                    "[ducked_source][dubbed_mix]"
                    f"amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                    f"alimiter=limit=0.95,atrim=duration={duration},asetpts=PTS-STARTPTS[mixed]",
                )
            )
        else:
            filters.append(
                "[source][dubbed]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
                f"alimiter=limit=0.95,atrim=duration={duration},asetpts=PTS-STARTPTS[mixed]"
            )
        return ";".join(filters)

    @classmethod
    def _ffmpeg_command(cls, plan: AudioMixingPlan, output_path: Path, filter_graph: str) -> list[str]:
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        for placement in plan.placements:
            command.extend(("-i", str(placement.audio_path)))
        if plan.source_audio_path is not None:
            command.extend(("-i", str(plan.source_audio_path)))
        output_label = "[mixed]" if plan.include_source_audio else "[dubbed]"
        command.extend(
            (
                "-filter_complex",
                filter_graph,
                "-map",
                output_label,
                "-vn",
                "-ar",
                str(plan.output_sample_rate),
                "-ac",
                str(plan.output_channels),
                "-c:a",
                "pcm_s16le",
                str(output_path),
            )
        )
        return command

    @staticmethod
    def _validate_output(plan: AudioMixingPlan, output_info: AudioStreamInfo) -> None:
        if output_info.sample_rate != plan.output_sample_rate or output_info.channels != plan.output_channels:
            raise AudioMixingFailedError("Mixed audio did not retain the planned sample rate and channel count.")
        allowed_difference = max(0.05, 2 / output_info.sample_rate)
        if abs(output_info.duration - plan.timeline_duration) > allowed_difference:
            raise AudioMixingFailedError("Mixed audio has an unexpected total duration.")

    @staticmethod
    def _channel_layout(channels: int) -> str:
        return "mono" if channels == 1 else "stereo"

    @staticmethod
    def _number(value: float) -> str:
        return format(value, ".12g")

    @staticmethod
    def _run_command(command: list[str]) -> None:
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise MissingAudioMixingDependencyError(
                "FFmpeg is not available. Install FFmpeg and ensure ffmpeg and ffprobe are on PATH."
            ) from error
        except OSError as error:
            raise AudioMixingFailedError("FFmpeg could not be started for timeline mixing.") from error
        if completed.returncode != 0:
            details = completed.stderr.strip().splitlines()
            reason = details[-1] if details else "unknown FFmpeg error"
            raise AudioMixingFailedError(f"FFmpeg timeline mixing failed: {reason}")

    @staticmethod
    def _probe_audio(path: Path) -> AudioStreamInfo:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,channel_layout:format=duration",
            "-of",
            "json",
            str(path),
        ]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise MissingAudioMixingDependencyError(
                "FFprobe is not available. Install FFmpeg and ensure ffmpeg and ffprobe are on PATH."
            ) from error
        except OSError as error:
            raise AudioMixingFailedError("FFprobe could not inspect the mixed audio.") from error
        if completed.returncode != 0:
            raise AudioMixingFailedError(f"FFprobe could not inspect audio from '{path.name}'.")
        try:
            payload: dict[str, Any] = json.loads(completed.stdout)
            stream = payload["streams"][0]
            return AudioStreamInfo(
                duration=float(payload["format"]["duration"]),
                sample_rate=int(stream["sample_rate"]),
                channels=int(stream["channels"]),
                channel_layout=str(stream["channel_layout"]) if stream.get("channel_layout") else None,
            )
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise AudioMixingFailedError(f"FFprobe returned invalid audio metadata for '{path.name}'.") from error
