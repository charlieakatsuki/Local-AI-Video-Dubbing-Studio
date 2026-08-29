"""Unit tests for Phase 7 local TTS without loading or downloading models."""

from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_dubbing.tts.engine import TextToSpeechEngine, VoxCPMEngine
from local_dubbing.tts.manager import TTSManager
from local_dubbing.tts.models import (
    EmptyTTSTextError,
    GeneratedAudioSegment,
    InvalidTTSConfigurationError,
    InvalidTTSSegmentError,
    MissingTTSDependencyError,
    TTSConfig,
    TTSSegment,
    UnsupportedTTSEngineError,
)


class FakeWaveform:
    def __len__(self) -> int:
        return 24_000


class FakeVoxCPM:
    tts_model = SimpleNamespace(sample_rate=24_000)

    def generate(self, **kwargs):
        assert kwargs["text"]
        return FakeWaveform()


class FakeEngine(TextToSpeechEngine):
    name = "Fake TTS"

    def synthesize(self, segments, output_dir, target_language, config, progress_callback=None):
        return tuple(
            GeneratedAudioSegment(
                segment.segment_id,
                Path(output_dir) / f"{segment.segment_id}.wav",
                1.0,
                segment.start,
                segment.end,
                target_language,
                {"backend": "fake"},
            )
            for segment in segments
        )


def _segments() -> tuple[TTSSegment, ...]:
    return (
        TTSSegment("line-1", 1.25, 2.5, "Halo"),
        TTSSegment("line-2", 2.5, 5.75, "dunia"),
    )


def test_tts_configuration_defaults_and_validation() -> None:
    assert TTSConfig().model_name == "openbmb/VoxCPM2"
    assert TTSConfig(device="cuda:1", inference_timesteps=12).device == "cuda:1"
    with pytest.raises(InvalidTTSConfigurationError):
        TTSConfig(model_name=" ")
    with pytest.raises(InvalidTTSConfigurationError):
        TTSConfig(device="tpu")
    with pytest.raises(InvalidTTSConfigurationError):
        TTSConfig(cfg_value=0)
    with pytest.raises(InvalidTTSConfigurationError):
        TTSConfig(inference_timesteps=0)


def test_empty_text_and_empty_batch_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(EmptyTTSTextError):
        TTSSegment("line-1", 0, 1, "  ")
    with pytest.raises(EmptyTTSTextError):
        TTSManager((FakeEngine(),)).synthesize_segments([], "id", tmp_path, FakeEngine.name, TTSConfig())


def test_segment_validation() -> None:
    with pytest.raises(InvalidTTSSegmentError):
        TTSSegment("", 0, 1, "Hello")
    with pytest.raises(InvalidTTSSegmentError):
        TTSSegment("line-1", -1, 1, "Hello")
    with pytest.raises(InvalidTTSSegmentError):
        TTSSegment("line-1", 2, 1, "Hello")


def test_tts_package_import_is_lazy() -> None:
    from local_dubbing.tts import TTSConfig as ImportedConfig, VoxCPMEngine as ImportedEngine

    assert ImportedConfig is TTSConfig
    assert ImportedEngine is VoxCPMEngine
    assert ImportedEngine()._model is None


def test_engine_selection() -> None:
    manager = TTSManager((FakeEngine(),))
    assert manager.engine_names == (FakeEngine.name,)
    with pytest.raises(UnsupportedTTSEngineError):
        manager.synthesize_segments(_segments(), "id", Path("unused"), "missing", TTSConfig())


def test_structured_voxcpm_output_preserves_ids_and_timestamps(tmp_path: Path) -> None:
    written = []
    engine = VoxCPMEngine(
        model_factory=lambda config: FakeVoxCPM(),
        audio_writer=lambda path, waveform, sample_rate: written.append((path, sample_rate)),
    )
    result = TTSManager((engine,)).synthesize_segments(
        _segments(), "id", tmp_path, engine.name, TTSConfig(),
    )
    assert [(item.segment_id, item.start, item.end, item.duration) for item in result.segments] == [
        ("line-1", 1.25, 2.5, 1.0),
        ("line-2", 2.5, 5.75, 1.0),
    ]
    assert all(item.target_language == "id" for item in result.segments)
    assert all(item.metadata["sample_rate"] == 24_000 for item in result.segments)
    assert [sample_rate for _, sample_rate in written] == [24_000, 24_000]


def test_missing_voxcpm_dependency_is_user_facing(monkeypatch, tmp_path: Path) -> None:
    original_import = builtins.__import__

    def missing_voxcpm(name, *args, **kwargs):
        if name == "voxcpm":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_voxcpm)
    with pytest.raises(MissingTTSDependencyError, match="VoxCPM is not installed"):
        VoxCPMEngine().synthesize(_segments(), tmp_path, "id", TTSConfig())
