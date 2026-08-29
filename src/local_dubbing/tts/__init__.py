"""Pluggable local text-to-speech interfaces and VoxCPM adapter."""

from .engine import TextToSpeechEngine, VoxCPMEngine
from .manager import TTSManager
from .models import (
    GeneratedAudioSegment,
    TTSConfig,
    TTSError,
    TTSResult,
    TTSSegment,
)

__all__ = [
    "GeneratedAudioSegment",
    "TextToSpeechEngine",
    "TTSConfig",
    "TTSError",
    "TTSManager",
    "TTSResult",
    "TTSSegment",
    "VoxCPMEngine",
]
