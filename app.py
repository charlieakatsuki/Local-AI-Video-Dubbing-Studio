"""Initial Streamlit interface for Local AI Video Dubbing Studio."""

from __future__ import annotations

import streamlit as st


LANGUAGES: dict[str, str] = {
    "English": "en",
    "Indonesian": "id",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
}


def main() -> None:
    """Render the initial, non-processing user interface."""
    st.set_page_config(page_title="Local AI Video Dubbing Studio", page_icon="🎙️")
    st.title("Local AI Video Dubbing Studio")
    st.write(
        "A local-first workspace for preparing AI-assisted video dubbing workflows. "
        "No cloud API is required by this project foundation."
    )
    st.info("The AI dubbing pipeline is not implemented yet. This screen is a UI foundation only.")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "mkv", "mov", "avi"],
        help="Your file stays in the browser session for now.",
    )
    source_language = st.selectbox("Source language", options=list(LANGUAGES), index=0)
    target_language = st.selectbox("Target language", options=list(LANGUAGES), index=1)

    if source_language == target_language:
        st.warning("Choose different source and target languages for a future dubbing job.")
    if uploaded_video is not None:
        st.success(f"Selected video: {uploaded_video.name}")

    st.subheader("Planned processing stages")
    stages = {
        "Speech-to-Text": "Transcribe spoken dialogue with timestamps.",
        "Translation": "Translate transcript segments with an offline engine.",
        "Text-to-Speech": "Generate target-language speech with a local voice engine.",
        "Audio synchronization": "Align generated dialogue to the original timing.",
        "Video rendering": "Use FFmpeg to combine synchronized audio and video.",
    }
    for stage, description in stages.items():
        with st.expander(stage, expanded=False):
            st.caption(f"Planned: {description}")

    st.button("Start dubbing", disabled=True, help="Available after the pipeline is implemented.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        st.error(f"The application could not start: {error}")
