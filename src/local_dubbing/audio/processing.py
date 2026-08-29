"""Non-destructive local processing of Phase 8 timing-alignment plans."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from .models import (
    AlignmentAction,
    AudioProcessingFailedError,
    AudioProcessingResult,
    AudioStreamInfo,
    InvalidAudioProcessingInputError,
    MissingAudioProcessingDependencyError,
    ProcessedAudioSegment,
    TimingAlignmentInstruction,
    TimingAlignmentResult,
)

ProgressCallback = Callable[[str], None]
CommandRunner = Callable[[list[str]], None]
AudioProbe = Callable[[Path], AudioStreamInfo]
FileCopier = Callable[[Path, Path], None]


class AudioTimingProcessor(ABC):
    """Stable boundary for materializing alignment plans as new WAV files."""

    name: str

    @abstractmethod
    def process(
        self,
        plan: TimingAlignmentResult,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> AudioProcessingResult:
        """Create processed clips without modifying source TTS files."""


class FFmpegAudioTimingProcessor(AudioTimingProcessor):
    """Apply alignment with FFmpeg's pitch-preserving ``atempo`` filter."""

    name = "FFmpeg audio timing (local)"

    def __init__(
        self,
        command_runner: CommandRunner | None = None,
        audio_probe: AudioProbe | None = None,
        file_copier: FileCopier | None = None,
    ) -> None:
        self._command_runner = command_runner or self._run_command
        self._audio_probe = audio_probe or self._probe_audio
        self._file_copier = file_copier or shutil.copyfile

    def process(
        self,
        plan: TimingAlignmentResult,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> AudioProcessingResult:
        instructions = tuple(plan.instructions)
        self._validate_plan(instructions)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        processed = []
        for index, instruction in enumerate(instructions, start=1):
            if progress_callback:
                progress_callback(f"Processing audio segment {index} of {len(instructions)}…")
            processed.append(self._process_segment(index, instruction, destination))
        return AudioProcessingResult(tuple(processed), self.name)

    @staticmethod
    def _validate_plan(instructions: tuple[TimingAlignmentInstruction, ...]) -> None:
        segment_ids = [instruction.segment_id for instruction in instructions]
        if len(set(segment_ids)) != len(segment_ids):
            raise InvalidAudioProcessingInputError("Alignment instruction IDs must be unique.")
        for instruction in instructions:
            source = Path(instruction.source_audio_path)
            if not source.is_file() or source.stat().st_size == 0:
                raise InvalidAudioProcessingInputError(
                    f"Source WAV for segment '{instruction.segment_id}' is missing or empty."
                )
            if source.suffix.lower() != ".wav":
                raise InvalidAudioProcessingInputError(
                    f"Source audio for segment '{instruction.segment_id}' must be a WAV file."
                )

    def _process_segment(
        self,
        index: int,
        instruction: TimingAlignmentInstruction,
        output_dir: Path,
    ) -> ProcessedAudioSegment:
        source_path = Path(instruction.source_audio_path)
        source_info = self._audio_probe(source_path)
        self._validate_source_duration(instruction, source_info)
        output_path = output_dir / self._filename(index, instruction.segment_id)
        if output_path.resolve() == source_path.resolve():
            raise InvalidAudioProcessingInputError("Processed audio path must differ from its source path.")

        filter_chain = self._filter_chain(instruction)
        if instruction.action is AlignmentAction.KEEP:
            self._copy_unchanged(source_path, output_path)
        else:
            command = self._ffmpeg_command(instruction, source_info, output_path, filter_chain)
            self._command_runner(command)

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AudioProcessingFailedError(
                f"Audio processing did not create output for segment '{instruction.segment_id}'."
            )
        output_info = self._audio_probe(output_path)
        self._validate_output(instruction, source_info, output_info)
        metadata = {
            "processor": "ffmpeg",
            "alignment_action": instruction.action.value,
            "pitch_preserving": instruction.playback_rate != 1.0,
            "playback_rate": instruction.playback_rate,
            "pad_end_seconds": instruction.pad_end_seconds,
            "trim_end_seconds": instruction.trim_end_seconds,
            "source_duration_reported": instruction.source_duration,
            "source_duration_measured": source_info.duration,
            "target_duration": instruction.target_duration,
            "output_duration_measured": output_info.duration,
            "filter_chain": filter_chain,
        }
        return ProcessedAudioSegment(
            segment_id=instruction.segment_id,
            source_audio_path=source_path,
            processed_audio_path=output_path,
            duration=output_info.duration,
            timeline_start=instruction.timeline_start,
            timeline_end=instruction.timeline_end,
            target_language=instruction.target_language,
            sample_rate=output_info.sample_rate,
            channels=output_info.channels,
            channel_layout=output_info.channel_layout,
            source_metadata=dict(instruction.source_metadata),
            processing_metadata=metadata,
        )

    def _copy_unchanged(self, source_path: Path, output_path: Path) -> None:
        try:
            self._file_copier(source_path, output_path)
        except OSError as error:
            raise AudioProcessingFailedError("The unchanged WAV could not be copied safely.") from error

    @staticmethod
    def _validate_source_duration(
        instruction: TimingAlignmentInstruction,
        source_info: AudioStreamInfo,
    ) -> None:
        allowed_difference = max(0.05, 2 / source_info.sample_rate)
        if abs(source_info.duration - instruction.source_duration) > allowed_difference:
            raise InvalidAudioProcessingInputError(
                f"Measured WAV duration for segment '{instruction.segment_id}' no longer matches its alignment plan."
            )

    @staticmethod
    def _validate_output(
        instruction: TimingAlignmentInstruction,
        source_info: AudioStreamInfo,
        output_info: AudioStreamInfo,
    ) -> None:
        stream_changed = (
            output_info.sample_rate != source_info.sample_rate
            or output_info.channels != source_info.channels
            or (
                source_info.channel_layout is not None
                and output_info.channel_layout is not None
                and output_info.channel_layout != source_info.channel_layout
            )
        )
        if stream_changed:
            raise AudioProcessingFailedError("Processed audio did not preserve sample rate and channel information.")
        expected_duration = instruction.expected_duration
        allowed_difference = max(0.05, 2 / output_info.sample_rate)
        if abs(output_info.duration - expected_duration) > allowed_difference:
            raise AudioProcessingFailedError(
                f"Processed audio for segment '{instruction.segment_id}' has an unexpected duration."
            )

    @classmethod
    def _ffmpeg_command(
        cls,
        instruction: TimingAlignmentInstruction,
        source_info: AudioStreamInfo,
        output_path: Path,
        filter_chain: str,
    ) -> list[str]:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(instruction.source_audio_path),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            filter_chain,
            "-ar",
            str(source_info.sample_rate),
            "-ac",
            str(source_info.channels),
        ]
        if source_info.channel_layout:
            command.extend(("-channel_layout", source_info.channel_layout))
        command.extend(("-c:a", "pcm_s16le", str(output_path)))
        return command

    @classmethod
    def _filter_chain(cls, instruction: TimingAlignmentInstruction) -> str:
        filters = []
        if instruction.playback_rate > 1.0:
            filters.extend(cls._atempo_filters(instruction.playback_rate))
        if instruction.action is not AlignmentAction.KEEP:
            filters.extend(
                (
                    f"apad=pad_dur={cls._number(instruction.target_duration)}",
                    f"atrim=duration={cls._number(instruction.target_duration)}",
                    "asetpts=PTS-STARTPTS",
                )
            )
        return ",".join(filters)

    @classmethod
    def _atempo_filters(cls, rate: float) -> tuple[str, ...]:
        """Split rates into FFmpeg's 0.5–2.0 atempo range for future policies."""
        factors = []
        remaining = rate
        while remaining > 2.0:
            factors.append(2.0)
            remaining /= 2.0
        factors.append(remaining)
        return tuple(f"atempo={cls._number(factor)}" for factor in factors)

    @staticmethod
    def _number(value: float) -> str:
        return format(value, ".12g")

    @staticmethod
    def _filename(index: int, segment_id: str) -> str:
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(segment_id)).strip("._") or "segment"
        return f"aligned_{index:04d}_{safe_id[:80]}.wav"

    @staticmethod
    def _run_command(command: list[str]) -> None:
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise MissingAudioProcessingDependencyError(
                "FFmpeg is not available. Install FFmpeg and ensure ffmpeg and ffprobe are on PATH."
            ) from error
        except OSError as error:
            raise AudioProcessingFailedError("FFmpeg could not be started.") from error
        if completed.returncode != 0:
            details = completed.stderr.strip().splitlines()
            reason = details[-1] if details else "unknown FFmpeg error"
            raise AudioProcessingFailedError(f"FFmpeg audio processing failed: {reason}")

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
            raise MissingAudioProcessingDependencyError(
                "FFprobe is not available. Install FFmpeg and ensure ffmpeg and ffprobe are on PATH."
            ) from error
        except OSError as error:
            raise AudioProcessingFailedError("FFprobe could not inspect the WAV file.") from error
        if completed.returncode != 0:
            raise AudioProcessingFailedError(f"FFprobe could not inspect '{path.name}'.")
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
            raise AudioProcessingFailedError(f"FFprobe returned invalid metadata for '{path.name}'.") from error
