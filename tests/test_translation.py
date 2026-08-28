"""Unit tests for local translation without downloading Argos models."""

from types import SimpleNamespace

import pytest

from local_dubbing.stt.models import TranscriptionResult, TranscriptionSegment
from local_dubbing.translation.engine import ArgosPackageManager, ArgosTranslateEngine, TranslationEngine
from local_dubbing.translation.manager import TranslationManager, format_translated_srt, format_translated_txt
from local_dubbing.translation.models import (
    EmptyTranscriptError,
    LanguagePair,
    MissingTranslationPackageError,
    TranslationResult,
    TranslationSegment,
    UnsupportedLanguagePairError,
)


class FakeTranslation:
    def translate(self, text: str) -> str:
        return f"id:{text}"


class FakeLanguage:
    def __init__(self, code: str) -> None:
        self.code = code

    def get_translation(self, target: "FakeLanguage") -> FakeTranslation:
        assert target.code == "id"
        return FakeTranslation()


class FakeEngine(TranslationEngine):
    name = "Fake local engine"

    def available_language_pairs(self) -> frozenset[LanguagePair]:
        return frozenset({LanguagePair("en", "id")})

    def translate_segments(self, segments, source_language, target_language, progress_callback=None):
        return tuple(TranslationSegment(segment.start, segment.end, f"id:{segment.text.strip()}") for segment in segments)


def _argos_engine() -> ArgosTranslateEngine:
    packages = SimpleNamespace(get_installed_packages=lambda: [SimpleNamespace(from_code="en", to_code="id")])
    translator = SimpleNamespace(get_installed_languages=lambda: [FakeLanguage("en"), FakeLanguage("id")])
    return ArgosTranslateEngine(ArgosPackageManager(packages), translator)


def _transcription() -> TranscriptionResult:
    return TranscriptionResult(
        (TranscriptionSegment(1.25, 2.5, " Hello "), TranscriptionSegment(2.5, 5.75, "world")), "en"
    )


def test_argos_translates_structured_segments_and_preserves_timestamps() -> None:
    translated = _argos_engine().translate_segments(_transcription().segments, "en", "id")
    assert [(segment.start, segment.end, segment.text) for segment in translated] == [
        (1.25, 2.5, "id:Hello"), (2.5, 5.75, "id:world")
    ]


def test_installed_argos_language_pairs_are_reported() -> None:
    packages = SimpleNamespace(get_installed_packages=lambda: [SimpleNamespace(from_code="en", to_code="id")])
    assert ArgosPackageManager(packages).installed_language_pairs() == frozenset({LanguagePair("en", "id")})


def test_missing_argos_language_package_is_user_facing() -> None:
    engine = ArgosTranslateEngine(ArgosPackageManager(SimpleNamespace(get_installed_packages=lambda: [])), SimpleNamespace())
    with pytest.raises(MissingTranslationPackageError, match="en → id"):
        engine.translate_segments(_transcription().segments, "en", "id")


def test_language_pair_validation_and_empty_transcript_handling() -> None:
    manager = TranslationManager((FakeEngine(),))
    with pytest.raises(UnsupportedLanguagePairError):
        manager.translate_transcription(_transcription(), "en", "en", FakeEngine.name)
    with pytest.raises(UnsupportedLanguagePairError):
        manager.translate_transcription(_transcription(), "invalid", "id", FakeEngine.name)
    empty = TranscriptionResult((TranscriptionSegment(0, 1, " "),), "en")
    with pytest.raises(EmptyTranscriptError):
        manager.translate_transcription(empty, "en", "id", FakeEngine.name)


def test_translated_txt_and_srt_formatting_preserve_timestamps() -> None:
    result = TranslationResult(
        (TranslationSegment(1.25, 2.5, "Halo"), TranslationSegment(2.5, 5.75, "dunia")),
        "en", "id", "Fake local engine",
    )
    assert format_translated_txt(result) == "Halo dunia\n"
    assert format_translated_srt(result) == (
        "1\n00:00:01,250 --> 00:00:02,500\nHalo\n\n2\n00:00:02,500 --> 00:00:05,750\ndunia\n"
    )
