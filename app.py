"""Streamlit interface for local faster-whisper speech-to-text."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import streamlit as st

from local_dubbing.audio import (
    AlignmentAction,
    AudioMixingConfig,
    AudioMixingError,
    AudioProcessingError,
    AudioProcessingResult,
    DefaultTimingAlignmentEngine,
    FFmpegAudioTimelineMixer,
    FFmpegAudioTimingProcessor,
    MixedAudioResult,
    TimingAlignmentConfig,
    TimingAlignmentError,
    TimingAlignmentResult,
)
from local_dubbing.config import AppConfig
from local_dubbing.stt.engine import FasterWhisperEngine, cuda_available
from local_dubbing.stt.formatter import format_srt, format_transcript
from local_dubbing.stt.models import STTError, STTConfig, SUPPORTED_LANGUAGES, SUPPORTED_MODELS, TranscriptionResult
from local_dubbing.translation.manager import TranslationManager, format_translated_srt, format_translated_txt
from local_dubbing.translation.models import SUPPORTED_LANGUAGES as TRANSLATION_LANGUAGES
from local_dubbing.translation.models import TranslationError, TranslationResult
from local_dubbing.tts.manager import TTSManager
from local_dubbing.tts.models import TTSConfig, TTSError, TTSResult, TTSSegment

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


def _tts_segments(result: TranslationResult) -> tuple[TTSSegment, ...]:
    """Add stable IDs to translated segments at the TTS boundary."""
    return tuple(
        TTSSegment(f"segment-{index:04d}", segment.start, segment.end, segment.text)
        for index, segment in enumerate(result.segments, start=1)
        if segment.text.strip()
    )


@st.cache_resource
def _tts_manager() -> TTSManager:
    """Keep the lazy engine instance across Streamlit reruns."""
    return TTSManager()


def _render_tts(result: TTSResult) -> None:
    """Render generated segment audio without aligning or mixing it."""
    st.success(f"Generated {len(result.segments)} audio segments with {result.engine_name}.")
    for segment in result.segments:
        st.markdown(
            f"**{segment.segment_id} · original {segment.start:0.2f}s–{segment.end:0.2f}s · "
            f"generated {segment.duration:0.2f}s**"
        )
        if segment.audio_path.is_file():
            st.audio(str(segment.audio_path), format="audio/wav")
            st.caption(str(segment.audio_path))
        else:
            st.warning(f"Generated audio is no longer available: {segment.audio_path}")


def _render_alignment(result: TimingAlignmentResult) -> None:
    """Render a non-destructive alignment plan for later audio processing."""
    st.success(f"Timing plan calculated for {len(result.instructions)} audio segments.")
    action_labels = {
        AlignmentAction.KEEP: "Keep unchanged",
        AlignmentAction.PAD_END: "Pad end with silence",
        AlignmentAction.SPEED_UP: "Speed up",
        AlignmentAction.SPEED_UP_AND_TRIM: "Speed up, then trim end",
    }
    rows = []
    for instruction in result.instructions:
        rows.append(
            {
                "Segment": instruction.segment_id,
                "Timeline": f"{instruction.timeline_start:.2f}s–{instruction.timeline_end:.2f}s",
                "Generated": f"{instruction.source_duration:.3f}s",
                "Target": f"{instruction.target_duration:.3f}s",
                "Difference": f"{instruction.difference_seconds:+.3f}s",
                "Plan": action_labels[instruction.action],
                "Speed": f"{instruction.playback_rate:.3f}×",
                "End padding": f"{instruction.pad_end_seconds:.3f}s",
                "End trim": f"{instruction.trim_end_seconds:.3f}s",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "This is a calculation only. Source WAV files are unchanged; silence insertion, speed adjustment, "
        "trimming, mixing, and FFmpeg rendering are not performed in Phase 8."
    )


def _render_processed_audio(result: AudioProcessingResult) -> None:
    """Render processed clips that are ready for timeline placement."""
    st.success(f"Prepared {len(result.segments)} aligned audio segments with {result.processor_name}.")
    for segment in result.segments:
        action = segment.processing_metadata.get("alignment_action", "unknown")
        st.markdown(
            f"**{segment.segment_id} · timeline {segment.timeline_start:.2f}s–{segment.timeline_end:.2f}s · "
            f"processed {segment.duration:.2f}s · {segment.sample_rate} Hz · "
            f"{segment.channels} channel(s) · {action}**"
        )
        if segment.processed_audio_path.is_file():
            st.audio(str(segment.processed_audio_path), format="audio/wav")
            st.caption(str(segment.processed_audio_path))
        else:
            st.warning(f"Processed audio is no longer available: {segment.processed_audio_path}")


def _render_mixed_audio(result: MixedAudioResult) -> None:
    """Render the Phase 10 soundtrack ready for later video attachment."""
    st.success(
        f"Mixed {len(result.placements)} segments into a {result.duration:.2f}s soundtrack "
        f"with {result.mixer_name}."
    )
    st.caption(
        f"{result.sample_rate} Hz · {result.channels} channel(s) · "
        f"source audio {'included' if result.metadata.get('source_audio_included') else 'not included'}"
    )
    if result.output_audio_path.is_file():
        st.audio(str(result.output_audio_path), format="audio/wav")
        st.caption(str(result.output_audio_path))
    else:
        st.warning(f"Mixed audio is no longer available: {result.output_audio_path}")


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
            st.session_state.pop("tts_result", None)
            st.session_state.pop("timing_alignment_result", None)
            st.session_state.pop("audio_processing_result", None)
            st.session_state.pop("audio_mixing_result", None)
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
                st.session_state.pop("tts_result", None)
                st.session_state.pop("timing_alignment_result", None)
                st.session_state.pop("audio_processing_result", None)
                st.session_state.pop("audio_mixing_result", None)
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
            st.header("Text-to-Speech")
            st.caption(
                "Generate one local WAV file per translated segment. Phase 7 does not align, mix, or render audio."
            )
            manager = _tts_manager()
            tts_engine_name = st.selectbox("Text-to-speech engine", manager.engine_names)
            tts_model_name = st.text_input("VoxCPM model", value="openbmb/VoxCPM2")
            tts_left, tts_right = st.columns(2)
            with tts_left:
                tts_device = st.selectbox("TTS device", ["auto", "cpu", "mps", "cuda"])
                voice_description = st.text_input(
                    "Voice description (optional)",
                    help="VoxCPM2 voice-design instruction, for example: A warm, calm narrator.",
                )
            with tts_right:
                inference_timesteps = st.number_input("Inference timesteps", min_value=1, value=10, step=1)
                seed = st.number_input("Generation seed", min_value=0, value=42, step=1)
                local_files_only = st.checkbox(
                    "Use cached/local model files only",
                    help="Prevents VoxCPM from downloading model files during generation.",
                )
            if st.button("Generate segment audio", type="primary"):
                status = st.status("Preparing local speech generation…", expanded=True)
                try:
                    config = TTSConfig(
                        model_name=tts_model_name,
                        device=tts_device,
                        inference_timesteps=int(inference_timesteps),
                        seed=int(seed),
                        local_files_only=local_files_only,
                        voice_description=voice_description or None,
                    )
                    project_config = AppConfig.from_project_root(Path(__file__).resolve().parent)
                    tts_result = manager.synthesize_segments(
                        _tts_segments(translated_result),
                        translated_result.target_language,
                        project_config.outputs_dir / "tts",
                        tts_engine_name,
                        config,
                        progress_callback=status.write,
                    )
                    st.session_state["tts_result"] = tts_result
                    st.session_state.pop("timing_alignment_result", None)
                    st.session_state.pop("audio_processing_result", None)
                    st.session_state.pop("audio_mixing_result", None)
                    status.update(label="Speech generation complete", state="complete")
                except TTSError as error:
                    status.update(label="Speech generation could not be completed", state="error")
                    st.error(str(error))
                except Exception:
                    status.update(label="Speech generation could not be completed", state="error")
                    st.error("An unexpected error occurred while generating speech. Please try again.")
            tts_result = st.session_state.get("tts_result")
            if isinstance(tts_result, TTSResult):
                _render_tts(tts_result)
                st.divider()
                st.header("Timing Alignment")
                st.caption(
                    "Calculate how each generated clip should fit its original timestamp slot. "
                    "This creates instructions only and does not process audio."
                )
                alignment_left, alignment_right = st.columns(2)
                with alignment_left:
                    tolerance_seconds = st.number_input(
                        "Duration tolerance (seconds)", min_value=0.0, value=0.05, step=0.01, format="%.2f"
                    )
                with alignment_right:
                    max_speed_up = st.number_input(
                        "Maximum speed-up", min_value=1.0, value=1.5, step=0.05, format="%.2f"
                    )
                if st.button("Calculate timing plan", type="primary"):
                    try:
                        alignment_result = DefaultTimingAlignmentEngine().plan(
                            tts_result.segments,
                            TimingAlignmentConfig(
                                tolerance_seconds=float(tolerance_seconds),
                                max_speed_up=float(max_speed_up),
                            ),
                        )
                        st.session_state["timing_alignment_result"] = alignment_result
                        st.session_state.pop("audio_processing_result", None)
                        st.session_state.pop("audio_mixing_result", None)
                    except TimingAlignmentError as error:
                        st.error(str(error))
                alignment_result = st.session_state.get("timing_alignment_result")
                if isinstance(alignment_result, TimingAlignmentResult):
                    _render_alignment(alignment_result)
                    st.subheader("Prepare aligned WAV files")
                    st.caption(
                        "Create new local WAV files from this plan. FFmpeg preserves pitch during speed changes; "
                        "the original VoxCPM files remain unchanged."
                    )
                    if st.button("Process aligned WAV files", type="primary"):
                        status = st.status("Preparing aligned audio…", expanded=True)
                        try:
                            project_config = AppConfig.from_project_root(Path(__file__).resolve().parent)
                            processing_result = FFmpegAudioTimingProcessor().process(
                                alignment_result,
                                project_config.outputs_dir / "aligned",
                                progress_callback=status.write,
                            )
                            st.session_state["audio_processing_result"] = processing_result
                            st.session_state.pop("audio_mixing_result", None)
                            status.update(label="Aligned audio preparation complete", state="complete")
                        except AudioProcessingError as error:
                            st.session_state.pop("audio_processing_result", None)
                            status.update(label="Aligned audio could not be prepared", state="error")
                            st.error(str(error))
                        except Exception:
                            st.session_state.pop("audio_processing_result", None)
                            status.update(label="Aligned audio could not be prepared", state="error")
                            st.error("An unexpected error occurred while processing audio. Please try again.")
                    processing_result = st.session_state.get("audio_processing_result")
                    if isinstance(processing_result, AudioProcessingResult):
                        _render_processed_audio(processing_result)
                        st.subheader("Timeline placement and audio mixing")
                        st.caption(
                            "Place aligned clips at their original timestamps and create one continuous PCM WAV. "
                            "Optionally retain the uploaded media's original soundtrack."
                        )
                        include_source_audio = st.checkbox(
                            "Mix with original uploaded audio",
                            value=False,
                            disabled=uploaded_media is None,
                        )
                        mix_left, mix_right = st.columns(2)
                        with mix_left:
                            source_volume = st.slider(
                                "Original-audio volume",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.35,
                                step=0.05,
                                disabled=not include_source_audio,
                            )
                            duck_source_audio = st.checkbox(
                                "Duck original audio while dubbed speech is active",
                                value=True,
                                disabled=not include_source_audio,
                            )
                        with mix_right:
                            ducking_ratio = st.slider(
                                "Ducking ratio",
                                min_value=1.0,
                                max_value=20.0,
                                value=8.0,
                                step=0.5,
                                disabled=not include_source_audio or not duck_source_audio,
                            )
                            ducking_release = st.number_input(
                                "Ducking release (milliseconds)",
                                min_value=10.0,
                                max_value=9000.0,
                                value=250.0,
                                step=10.0,
                                disabled=not include_source_audio or not duck_source_audio,
                            )
                        if st.button("Mix dubbed audio timeline", type="primary"):
                            status = st.status("Preparing the mixed soundtrack…", expanded=True)
                            try:
                                project_config = AppConfig.from_project_root(Path(__file__).resolve().parent)
                                source_path = None
                                if include_source_audio:
                                    source_directory = project_config.outputs_dir / "source_media"
                                    source_directory.mkdir(parents=True, exist_ok=True)
                                    source_path = _save_upload_temporarily(uploaded_media, source_directory)
                                mixing_result = FFmpegAudioTimelineMixer().mix(
                                    processing_result,
                                    project_config.outputs_dir / "mixed",
                                    AudioMixingConfig(
                                        include_source_audio=include_source_audio,
                                        source_volume=float(source_volume),
                                        duck_source_audio=include_source_audio and duck_source_audio,
                                        ducking_ratio=float(ducking_ratio),
                                        ducking_release_ms=float(ducking_release),
                                    ),
                                    source_audio_path=source_path,
                                    progress_callback=status.write,
                                )
                                st.session_state["audio_mixing_result"] = mixing_result
                                status.update(label="Audio timeline mixing complete", state="complete")
                            except (ValueError, AudioMixingError) as error:
                                st.session_state.pop("audio_mixing_result", None)
                                status.update(label="Audio timeline could not be mixed", state="error")
                                st.error(str(error))
                            except Exception:
                                st.session_state.pop("audio_mixing_result", None)
                                status.update(label="Audio timeline could not be mixed", state="error")
                                st.error("An unexpected error occurred while mixing audio. Please try again.")
                        mixing_result = st.session_state.get("audio_mixing_result")
                        if isinstance(mixing_result, MixedAudioResult):
                            _render_mixed_audio(mixing_result)
    st.divider()
    st.caption("Phase 10 creates a complete audio soundtrack. Video attachment/rendering remains Phase 11.")


if __name__ == "__main__":
    main()
