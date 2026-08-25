"""Pluggable local speech-to-text interfaces and faster-whisper adapter."""

from .engine import FasterWhisperEngine, SpeechToTextEngine
from .models import STTConfig, TranscriptionResult, TranscriptionSegment

__all__ = [
    "FasterWhisperEngine",
    "SpeechToTextEngine",
    "STTConfig",
    "TranscriptionResult",
    "TranscriptionSegment",
]
