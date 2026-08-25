"""Streamlit interface for local faster-whisper speech-to-text."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import streamlit as st

from local_dubbing.stt.engine import FasterWhisperEngine, cuda_available
from local_dubbing.stt.formatter import format_srt, format_transcript
from local_dubbing.stt.models import STTError, STTConfig, SUPPORTED_LANGUAGES, SUPPORTED_MODELS, TranscriptionResult

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
        st.markdown(f"**{segment.start:0.2f}s – {segment.end:0.2f}s**  \\n+{segment.text}")
    st.subheader("Full transcript")
    st.text_area("Transcript", value=result.full_text, height=220, disabled=True, label_visibility="collapsed")
    st.download_button("Download transcript.txt", format_transcript(result), "transcript.txt", "text/plain")
    st.download_button("Download subtitles.srt", format_srt(result.segments), "subtitles.srt", "application/x-subrip")


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
    st.caption("Translation, voice generation, dubbing, synchronization, and rendering are planned future phases.")


if __name__ == "__main__":
    main()
