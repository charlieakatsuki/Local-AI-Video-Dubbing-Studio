"""Pluggable local text-to-speech boundary and lazy VoxCPM adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path
import re
from typing import Any

from .models import (
    GeneratedAudioSegment,
    MissingTTSDependencyError,
    TTSConfig,
    TTSModelError,
    TTSSegment,
    TTSSynthesisError,
)

ProgressCallback = Callable[[str], None]
ModelFactory = Callable[[TTSConfig], Any]
AudioWriter = Callable[[Path, Any, int], None]


class TextToSpeechEngine(ABC):
    """Stable backend boundary for generating unsynchronized segment audio."""

    name: str

    @abstractmethod
    def synthesize(
        self,
        segments: Iterable[TTSSegment],
        output_dir: Path,
        target_language: str,
        config: TTSConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[GeneratedAudioSegment, ...]:
        """Generate one audio file per segment while retaining source timestamps."""


class VoxCPMEngine(TextToSpeechEngine):
    """Lazy VoxCPM backend; imports and model initialization happen on synthesis."""

    name = "VoxCPM (local)"

    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        audio_writer: AudioWriter | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._audio_writer = audio_writer
        self._model: Any | None = None
        self._model_key: tuple[Any, ...] | None = None

    @staticmethod
    def _default_dependencies() -> tuple[Any, Any]:
        try:
            from voxcpm import VoxCPM
            import soundfile
        except ImportError as error:
            raise MissingTTSDependencyError(
                "VoxCPM is not installed. Install the optional TTS dependencies with "
                "'python -m pip install -e .[tts]'."
            ) from error
        return VoxCPM, soundfile

    def _create_model(self, config: TTSConfig) -> Any:
        if self._model_factory is not None:
            return self._model_factory(config)
        VoxCPM, _ = self._default_dependencies()
        try:
            return VoxCPM.from_pretrained(
                config.model_name,
                load_denoiser=config.load_denoiser,
                cache_dir=str(config.model_cache_dir) if config.model_cache_dir else None,
                local_files_only=config.local_files_only,
                optimize=config.optimize,
                device=None if config.device == "auto" else config.device,
            )
        except Exception as error:
            raise TTSModelError(
                "The VoxCPM model could not be loaded. Check the model name, connection, cache, and device settings."
            ) from error

    def _get_model(self, config: TTSConfig) -> Any:
        key = (
            config.model_name,
            config.device,
            config.load_denoiser,
            config.optimize,
            config.local_files_only,
            config.model_cache_dir,
        )
        if self._model is None or self._model_key != key:
            self._model = self._create_model(config)
            self._model_key = key
        return self._model

    def _write_audio(self, path: Path, waveform: Any, sample_rate: int) -> None:
        if self._audio_writer is not None:
            self._audio_writer(path, waveform, sample_rate)
            return
        _, soundfile = self._default_dependencies()
        soundfile.write(str(path), waveform, sample_rate)

    @staticmethod
    def _filename(index: int, segment_id: str) -> str:
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(segment_id)).strip("._") or "segment"
        return f"segment_{index:04d}_{safe_id[:80]}.wav"

    def synthesize(
        self,
        segments: Iterable[TTSSegment],
        output_dir: Path,
        target_language: str,
        config: TTSConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[GeneratedAudioSegment, ...]:
        materialized_segments = tuple(segments)
        output_dir.mkdir(parents=True, exist_ok=True)
        if progress_callback:
            progress_callback(f"Loading {config.model_name}…")
        model = self._get_model(config)
        try:
            sample_rate = int(model.tts_model.sample_rate)
        except (AttributeError, TypeError, ValueError) as error:
            raise TTSModelError("VoxCPM did not provide a valid output sample rate.") from error
        if sample_rate <= 0:
            raise TTSModelError("VoxCPM did not provide a valid output sample rate.")

        generated = []
        for index, segment in enumerate(materialized_segments, start=1):
            if progress_callback:
                progress_callback(f"Generating speech for segment {index} of {len(materialized_segments)}…")
            spoken_text = segment.text.strip()
            if config.voice_description:
                spoken_text = f"({config.voice_description.strip()}){spoken_text}"
            try:
                waveform = model.generate(
                    text=spoken_text,
                    cfg_value=config.cfg_value,
                    inference_timesteps=config.inference_timesteps,
                    seed=config.seed,
                )
                sample_count = len(waveform)
                audio_path = output_dir / self._filename(index, segment.segment_id)
                self._write_audio(audio_path, waveform, sample_rate)
            except Exception as error:
                raise TTSSynthesisError(
                    f"VoxCPM could not generate audio for segment '{segment.segment_id}'."
                ) from error
            generated.append(
                GeneratedAudioSegment(
                    segment_id=segment.segment_id,
                    audio_path=audio_path,
                    duration=sample_count / sample_rate,
                    start=segment.start,
                    end=segment.end,
                    target_language=target_language,
                    metadata={
                        "backend": "voxcpm",
                        "model": config.model_name,
                        "sample_rate": sample_rate,
                        "sample_count": sample_count,
                        "device": config.device,
                        "cfg_value": config.cfg_value,
                        "inference_timesteps": config.inference_timesteps,
                        "seed": config.seed,
                    },
                )
            )
        return tuple(generated)
