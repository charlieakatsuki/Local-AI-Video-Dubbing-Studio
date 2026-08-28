# Local AI Video Dubbing Studio

Local AI Video Dubbing Studio is a Windows-first, **local-first** application for preparing dubbed versions of media with AI. It provides fully local speech-to-text, translation, and per-segment speech generation with no cloud API or credentials.

## Current feature: local speech-to-text

The Streamlit app uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) to transcribe media and generate timestamped segments, a full plain-text transcript (`transcript.txt`), and properly formatted SubRip subtitles (`subtitles.srt`).

Supported input formats are MP4, MOV, MKV, AVI, MP3, WAV, and M4A. Uploads are copied to an operating-system temporary directory only while a transcription runs; media is not stored in this Git repository.

### Models and languages

The UI supports the `tiny`, `base`, and `small` Whisper models; `tiny` is the default. It offers automatic language detection or manual selection for English, Indonesian, Japanese, Korean, Chinese, Spanish, French, and German.

No model is downloaded when the app starts. faster-whisper downloads the selected model only after you click **Transcribe** for the first time. Models are cached by faster-whisper outside the repository.

### CPU and GPU support

CPU is the primary supported path and uses efficient `int8` inference by default. CUDA is detected when available and can be selected in the UI; its default compute type is `float16`. An NVIDIA GPU is optional, and this project does not install the full CUDA toolkit.

## Current feature: local translation

After transcription, the **Translation** section translates its completed timestamped segments with [Argos Translate](https://www.argosopentech.com/). It preserves every segment's `start` and `end` values, and provides `translated.txt` plus `translated.srt`; the latter keeps the original SRT timings exactly.

Initial language choices are English, Indonesian, Japanese, Korean, Chinese, Spanish, French, and German. Translation runs locally/offline after the required Argos source-to-target package is installed. Packages are direct: English → Indonesian does not also provide Indonesian → English. The app reports installed pairs and clearly explains when a selected pair is missing. It does not download models at startup.

Argos language packages are model assets, are never stored in Git, and should remain outside the repository (or in ignored `argos-packages/`). To install a pair after activating `.venv`, refresh the package index and install only the pair you need:

```powershell
python -c "import argostranslate.package as p; p.update_package_index(); package = next(x for x in p.get_available_packages() if x.from_code == 'en' and x.to_code == 'id'); p.install_from_path(package.download())"
```

Replace `en` and `id` with the needed source and target codes. The package download needs internet access once; subsequent translation is local.

## Current feature: local text-to-speech

After translation, the **Text-to-Speech** section sends structured translated segments to a pluggable local TTS interface. Each segment has a stable ID, translated text, and the original start/end timestamps. The engine returns one WAV path per segment together with its measured duration, original timestamps, target language, and generation metadata.

[VoxCPM2](https://github.com/OpenBMB/VoxCPM) is the first backend. It is lazy-loaded: importing this project, opening the app, and running unit tests do not initialize VoxCPM or download weights. The model is loaded only after **Generate segment audio** is clicked. Generated WAV files are written to the ignored `outputs/tts/` directory; model assets remain in the external Hugging Face cache unless a local model path or cache is configured.

This phase deliberately produces independent, unsynchronized WAV files. It does not stretch clips to timestamps, mix audio, or render video.

## Installation on Windows

Install Python 3.11 or 3.12. FFmpeg must be available on your system PATH so audio can be decoded from video and audio containers. VoxCPM currently requires Python below 3.13 and PyTorch 2.5 or newer; an NVIDIA GPU with CUDA 12 or newer is recommended for practical VoxCPM2 generation, although the adapter also exposes CPU and MPS device choices.

```powershell
git clone https://github.com/charlieakatsuki/Local-AI-Video-Dubbing-Studio.git
cd Local-AI-Video-Dubbing-Studio
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies are intended to be installed only in `.venv`. `faster-whisper` brings its inference runtime; it does not require PyTorch. The Windows requirements install VoxCPM on supported Python versions. Developers who install the package in editable mode can request the backend explicitly:

```powershell
python -m pip install -e ".[tts]"
```

VoxCPM and its large model weights are not imported or downloaded during application startup. The first generation using `openbmb/VoxCPM2` downloads the model to the normal Hugging Face cache. To prevent network access, enable **Use cached/local model files only** and provide a cached model ID or local model directory.

## Usage

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

1. Open the local URL printed by Streamlit.
2. In **Speech-to-Text**, upload a supported video or audio file.
3. Choose `tiny`, `base`, or `small`; leave automatic language detection enabled, or select the source language manually.
4. Click **Transcribe**. On the first run for that model, wait for its download to finish.
5. Review timestamped segments and download `transcript.txt` or `subtitles.srt`.
6. In **Translation**, select the source and target language, then click **Translate**.
7. Review translated segments and download `translated.txt` or `translated.srt`.
8. In **Text-to-Speech**, choose VoxCPM, its model and device, and optionally enter a voice description.
9. Click **Generate segment audio**. Review each independent WAV clip and its original timestamps.

## Known limitations

- Accurate decoding requires FFmpeg on the system PATH.
- The first transcription for each selected model needs an internet connection to download that model.
- Larger models improve accuracy but use more RAM, disk space, and processing time.
- Argos package coverage and translation quality vary by language pair; only installed direct pairs are available.
- Translation requires a completed non-empty transcription and different source/target languages.
- VoxCPM2 is a large 2B-parameter model; generation can be slow or memory-intensive without a compatible GPU.
- Basic VoxCPM2 voice design is supported, but reference-audio voice cloning is not exposed yet.
- Generated clips retain original timestamps as metadata but are not duration-aligned, mixed, or rendered into video.

## Project layout

- `src/local_dubbing/stt/` — modular STT data models, formatter, engine boundary, and faster-whisper adapter.
- `src/local_dubbing/translation/` — modular models, engine abstraction, Argos adapter, package discovery, and export helpers.
- `src/local_dubbing/tts/` — structured TTS models, backend manager, engine abstraction, and lazy VoxCPM adapter.
- `app.py` — Streamlit speech-to-text, translation, and per-segment TTS interface.
- `tests/` — fast unit tests that do not download Whisper, Argos, or VoxCPM models.

## Development

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
```

## Roadmap

1. Improve local speech-to-text robustness and media inspection.
2. Add optional offline translation adapters and improved package-management UX.
3. Extend local TTS with optional voice-cloning controls and additional backends.
4. Add timing alignment and local audio mixing (Phase 8).
5. Render a dubbed output video with FFmpeg in a later phase.

## License

This project is released under the [MIT License](LICENSE).
