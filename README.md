# Local AI Video Dubbing Studio

Local AI Video Dubbing Studio is a Windows-first, local-first application for creating dubbed versions of videos with AI. The project is designed to keep future speech, translation, voice, and rendering workflows on the user's machine rather than depending on paid cloud APIs.

## Current status

This repository currently provides the project foundation and a Streamlit interface prototype. The AI dubbing pipeline is **not implemented yet**: no models are downloaded, no media is processed, and uploaded videos are not persisted by the application.

## Planned architecture

The Python package is split by responsibility so engines can be added or replaced independently:

- `stt/` — speech-to-text adapters and transcript models.
- `translation/` — offline translation adapters.
- `tts/` — local text-to-speech adapters.
- `audio/` — timing, mixing, and synchronization utilities.
- `video/` — FFmpeg-backed video and audio rendering utilities.
- `pipeline.py` — orchestration layer joining the components.
- `config.py` — application settings and paths without user-specific hard-coding.

## Planned features

- Local video ingestion and media inspection with FFmpeg.
- Speech transcription with timestamps.
- Offline translation between selected languages.
- Local voice generation and timing alignment.
- Audio synchronization, mixing, and rendered dubbed video output.
- Progress reporting and recoverable error messages in the UI.

## Installation on Windows

Install Python 3.11 or newer and ensure it is available as `python`. FFmpeg will be required once video processing is implemented; it is not required for this initial UI.

```powershell
git clone https://github.com/your-username/Local-AI-Video-Dubbing-Studio.git
cd Local-AI-Video-Dubbing-Studio
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No AI models are installed by these commands. Future model downloads and generated media should remain in ignored directories such as `models/` and `outputs/`.

## Run the Streamlit app

With the virtual environment activated:

```powershell
python -m streamlit run app.py
```

## Development

Install the package in editable mode when working on the source package:

```powershell
python -m pip install -e .
python -m pytest
```

## Roadmap

1. Define shared data models and configuration for media jobs.
2. Add optional local STT, translation, and TTS engine adapters.
3. Add FFmpeg media inspection, synchronization, and rendering.
4. Build an end-to-end local dubbing workflow with progress and error recovery.
5. Add broader automated tests, documentation, and packaging support.

## License

This project is released under the [MIT License](LICENSE).
