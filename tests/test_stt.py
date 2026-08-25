"""Unit tests for local speech-to-text models and formatters."""

import pytest

from local_dubbing.stt.formatter import format_srt, format_srt_timestamp, format_transcript
from local_dubbing.stt.models import InvalidSTTConfigurationError, STTConfig, TranscriptionResult, TranscriptionSegment


def test_srt_timestamp_conversion() -> None:
    assert format_srt_timestamp(1.234) == "00:00:01,234"
    assert format_srt_timestamp(3_661.9996) == "01:01:02,000"
    assert format_srt_timestamp(-2) == "00:00:00,000"


def test_srt_segment_formatting() -> None:
    segments = (
        TranscriptionSegment(start=1.0, end=4.5, text=" Hello world. "),
        TranscriptionSegment(start=4.5, end=8.0, text="Welcome to this video."),
    )
    assert format_srt(segments) == (
        "1\n00:00:01,000 --> 00:00:04,500\nHello world.\n\n"
        "2\n00:00:04,500 --> 00:00:08,000\nWelcome to this video.\n"
    )


def test_transcript_formatting() -> None:
    result = TranscriptionResult(
        segments=(TranscriptionSegment(start=0, end=1, text=" Hello"), TranscriptionSegment(start=1, end=2, text="world. ")),
        detected_language="en",
    )
    assert format_transcript(result) == "Hello world.\n"


def test_stt_configuration_defaults_and_validation() -> None:
    assert STTConfig().model_name == "tiny"
    assert STTConfig(language="ja", device="cpu").language == "ja"
    with pytest.raises(InvalidSTTConfigurationError):
        STTConfig(model_name="large")
    with pytest.raises(InvalidSTTConfigurationError):
        STTConfig(language="invalid")


def test_stt_package_imports_without_model_download() -> None:
    from local_dubbing.stt import FasterWhisperEngine, STTConfig

    assert FasterWhisperEngine is not None
    assert STTConfig is not None
