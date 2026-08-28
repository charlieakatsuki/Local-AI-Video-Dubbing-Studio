"""Streamlit interface for local faster-whisper speech-to-text."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import streamlit as st

from local_dubbing.stt.engine import FasterWhisperEngine, cuda_available
from local_dubbing.stt.formatter import format_srt, format_transcript
from local_dubbing.stt.models import STTError, STTConfig, SUPPORTED_LANGUAGES, SUPPORTED_MODELS, TranscriptionResult
from local_dubbing.translation.manager import TranslationManager, format_translated_srt, format_translated_txt
from local_dubbing.translation.models import SUPPORTED_LANGUAGES as TRANSLATION_LANGUAGES
from local_dubbing.translation.models import TranslationError, TranslationResult

SUPPORTED_MEDIA_EXTENSIONS = ("mp4", "mov", "mkv", "avi", "mp3", "wav", "m4a")


def _save_upload_temporarily(uploaded_file: Any, directory: Path) -> Path:
    """Copy an uploaded file to an OS temporary directory for local processing."""
    extension = Path(uploaded_file.name).suffix.lower().lstrip(".")
    if extension not in SUPPORTED_MEDIA_EXTENSIONS:
        raise ValueError("Unsupported media format. Please upload a common video or audio file.")
    content = uploaded_file.getvalue()
    if not content:
        raise ValueError("The uploaded file is empty.")
    media_path = directory / f"uploaded_media.{extension}"
    media_path.write_bytes(content)
    return media_path


def _render_results(result: TranscriptionResult) -> None:
    """Render transcription output kept only in the browser session."""
    language_label = next(
        (name for name, code in SUPPORTED_LANGUAGES.items() if code == result.detected_language),
        result.detected_language or "Unknown",
    )
    st.success(f"Transcription complete. Detected language: {language_label}")
    if result.language_probability is not None:
        st.caption(f"Language confidence: {result.language_probability:.0%}")
    st.subheader("Transcription segments")
    for segment in result.segments:
        st.markdown(f"**{segment.start:0.2f}s – {segment.end:0.2f}s**  \n{segment.text}")
    st.subheader("Full transcript")
    st.text_area("Transcript", value=result.full_text, height=220, disabled=True, label_visibility="collapsed")
    st.download_button("Download transcript.txt", format_transcript(result), "transcript.txt", "text/plain")
    st.download_button("Download subtitles.srt", format_srt(result.segments), "subtitles.srt", "application/x-subrip")


def _render_translation(result: TranslationResult) -> None:
    """Render translated, timestamp-preserving output stored in this session."""
    st.success(f"Translation complete with {result.engine_name}.")
    st.subheader("Translated segments")
    for segment in result.segments:
        st.markdown(f"**{segment.start:0.2f}s – {segment.end:0.2f}s**  \n{segment.text}")
    st.subheader("Translated full transcript")
    st.text_area("Translated transcript", value=result.full_text, height=220, disabled=True, label_visibility="collapsed")
    st.download_button("Download translated.txt", format_translated_txt(result), "translated.txt", "text/plain")
    st.download_button(
        "Download translated.srt", format_translated_srt(result), "translated.srt", "application/x-subrip"
    )


def main() -> None:
    """Render the local-first speech-to-text application."""
    st.set_page_config(page_title="Local AI Video Dubbing Studio", page_icon="🎙️", layout="wide")
    st.title("Local AI Video Dubbing Studio")
    st.caption("Local-first speech-to-text. Media is processed on this machine; no cloud API is used.")
    st.header("Speech-to-Text")
    uploaded_media = st.file_uploader(
        "Upload video or audio", type=list(SUPPORTED_MEDIA_EXTENSIONS),
        help="Supported: MP4, MOV, MKV, AVI, MP3, WAV, and M4A. Files are temporary while transcribing.",
    )
    left, right = st.columns(2)
    with left:
        model_name = st.selectbox("Whisper model", SUPPORTED_MODELS, index=0)
        auto_detect = st.checkbox("Automatically detect source language", value=True)
        language_name = st.selectbox("Source language", list(SUPPORTED_LANGUAGES), disabled=auto_detect)
    with right:
        device_options = ["auto", "cpu"] + (["cuda"] if cuda_available() else [])
        device = st.selectbox("Processing device", device_options)
        compute_choice = st.selectbox("Compute type", ["Automatic", "int8", "float16", "int8_float16"])
        if not cuda_available():
            st.caption("CUDA was not detected. CPU mode uses efficient int8 inference by default.")
    if st.button("Transcribe", type="primary", disabled=uploaded_media is None):
        config = STTConfig(
            model_name=model_name, language=None if auto_detect else SUPPORTED_LANGUAGES[language_name],
            device=device, compute_type=None if compute_choice == "Automatic" else compute_choice,
        )
        status = st.status("Preparing transcription…", expanded=True)
        try:
            with TemporaryDirectory(prefix="local_dubbing_stt_") as temp_dir:
                media_path = _save_upload_temporarily(uploaded_media, Path(temp_dir))
                result = FasterWhisperEngine().transcribe(media_path, config, progress_callback=status.write)
            st.session_state["transcription_result"] = result
            st.session_state.pop("translation_result", None)
            status.update(label="Transcription complete", state="complete")
        except (ValueError, STTError) as error:
            status.update(label="Transcription could not be completed", state="error")
            st.error(str(error))
        except Exception:
            status.update(label="Transcription could not be completed", state="error")
            st.error("An unexpected error occurred while preparing the transcription. Please try again.")
    result = st.session_state.get("transcription_result")
    if isinstance(result, TranscriptionResult):
        _render_results(result)
        st.divider()
        st.header("Translation")
        st.caption("Translate the completed local transcription. Argos language packages are installed separately.")
        translation_manager = TranslationManager()
        engine_name = st.selectbox("Translation engine", translation_manager.engine_names)
        detected_language = result.detected_language if result.detected_language in TRANSLATION_LANGUAGES.values() else "en"
        language_names = list(TRANSLATION_LANGUAGES)
        source_index = next(
            index for index, language_name in enumerate(language_names)
            if TRANSLATION_LANGUAGES[language_name] == detected_language
        )
        source_name = st.selectbox("Translation source language", language_names, index=source_index)
        target_name = st.selectbox("Target language", language_names, index=1)
        try:
            available_pairs = translation_manager.available_language_pairs(engine_name)
            if available_pairs:
                pairs_text = ", ".join(
                    f"{pair.source_language} → {pair.target_language}"
                    for pair in sorted(available_pairs, key=lambda pair: (pair.source_language, pair.target_language))
                )
                st.caption(f"Installed Argos language packages: {pairs_text}")
            else:
                st.warning("No Argos language packages are installed. Install a source → target package before translating.")
        except TranslationError as error:
            st.warning(str(error))
        if st.button("Translate", type="primary"):
            status = st.status("Preparing local translation…", expanded=True)
            try:
                translated_result = translation_manager.translate_transcription(
                    result,
                    TRANSLATION_LANGUAGES[source_name],
                    TRANSLATION_LANGUAGES[target_name],
                    engine_name,
                    progress_callback=status.write,
                )
                st.session_state["translation_result"] = translated_result
                status.update(label="Translation complete", state="complete")
            except TranslationError as error:
                status.update(label="Translation could not be completed", state="error")
                st.error(str(error))
            except Exception:
                status.update(label="Translation could not be completed", state="error")
                st.error("An unexpected error occurred while translating. Please try again.")
        translated_result = st.session_state.get("translation_result")
        if isinstance(translated_result, TranslationResult):
            _render_translation(translated_result)
    st.divider()
    st.caption("Voice generation, dubbing, synchronization, and rendering are planned future phases.")


if __name__ == "__main__":
    main()
