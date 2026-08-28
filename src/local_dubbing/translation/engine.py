"""Translation engine abstraction and lazy Argos Translate adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

from .models import (
    LanguagePair,
    MissingTranslationPackageError,
    TranslationFailedError,
    TranslationSegment,
)


class TranslationEngine(ABC):
    """Abstract boundary for local translation backends."""

    name: str

    @abstractmethod
    def available_language_pairs(self) -> frozenset[LanguagePair]:
        """Return direct language pairs currently available to this engine."""

    @abstractmethod
    def translate_segments(
        self,
        segments: Iterable[Any],
        source_language: str,
        target_language: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[TranslationSegment, ...]:
        """Translate timestamped STT segments without changing timestamps."""


class ArgosPackageManager:
    """Read-only discovery of locally installed Argos language packages."""

    def __init__(self, package_module: Any | None = None) -> None:
        self._package_module = package_module

    def _packages(self) -> Any:
        if self._package_module is not None:
            return self._package_module
        try:
            import argostranslate.package as package_module
        except ImportError as error:
            raise MissingTranslationPackageError(
                "Argos Translate is not installed. Install the project dependencies and try again."
            ) from error
        return package_module

    def installed_language_pairs(self) -> frozenset[LanguagePair]:
        """Return direct pairs supplied by installed Argos model packages."""
        try:
            packages = self._packages().get_installed_packages()
        except MissingTranslationPackageError:
            raise
        except Exception as error:
            raise MissingTranslationPackageError(
                "Installed Argos language packages could not be inspected."
            ) from error
        return frozenset(
            LanguagePair(str(package.from_code), str(package.to_code))
            for package in packages
            if getattr(package, "from_code", None) and getattr(package, "to_code", None)
        )

    def has_language_pair(self, source_language: str, target_language: str) -> bool:
        """Return whether a direct installed package supports this pair."""
        return LanguagePair(source_language, target_language) in self.installed_language_pairs()


class ArgosTranslateEngine(TranslationEngine):
    """Argos Translate backend that loads models only when translation is requested."""

    name = "Argos Translate (local)"

    def __init__(self, package_manager: ArgosPackageManager | None = None, translate_module: Any | None = None) -> None:
        self.package_manager = package_manager or ArgosPackageManager()
        self._translate_module = translate_module

    def _translator(self) -> Any:
        if self._translate_module is not None:
            return self._translate_module
        try:
            import argostranslate.translate as translate_module
        except ImportError as error:
            raise MissingTranslationPackageError(
                "Argos Translate is not installed. Install the project dependencies and try again."
            ) from error
        return translate_module

    def available_language_pairs(self) -> frozenset[LanguagePair]:
        return self.package_manager.installed_language_pairs()

    def translate_segments(
        self,
        segments: Iterable[Any],
        source_language: str,
        target_language: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[TranslationSegment, ...]:
        if not self.package_manager.has_language_pair(source_language, target_language):
            raise MissingTranslationPackageError(
                f"The Argos language package for {source_language} → {target_language} is not installed. "
                "Install that package, then try again."
            )
        try:
            installed_languages = self._translator().get_installed_languages()
            source = next(language for language in installed_languages if language.code == source_language)
            target = next(language for language in installed_languages if language.code == target_language)
            translation = source.get_translation(target)
        except Exception as error:
            raise MissingTranslationPackageError(
                f"The installed Argos package for {source_language} → {target_language} is not ready to use."
            ) from error

        translated = []
        materialized_segments = tuple(segments)
        for index, segment in enumerate(materialized_segments, start=1):
            text = str(segment.text).strip()
            if progress_callback:
                progress_callback(f"Translating segment {index} of {len(materialized_segments)}…")
            try:
                translated_text = translation.translate(text) if text else ""
            except Exception as error:
                raise TranslationFailedError("Argos Translate could not translate the transcript.") from error
            translated.append(TranslationSegment(start=float(segment.start), end=float(segment.end), text=translated_text.strip()))
        return tuple(translated)
