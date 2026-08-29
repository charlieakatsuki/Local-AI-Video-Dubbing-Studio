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

VoxCPM deliberately produces independent WAV files. The timing layer described below analyzes them separately, so speech generation remains backend-neutral and unchanged.

## Current feature: timing alignment planning

After TTS generation, the **Timing Alignment** section compares each measured WAV duration with the duration of its original timestamp slot (`end - start`). Phase 8 creates a structured, non-destructive processing plan:

- clips within the configured tolerance are kept unchanged;
- shorter clips retain their natural speech rate and receive planned silence padding at the end;
- moderately longer clips receive the exact playback-rate increase needed to fit;
- severe overruns use a configurable maximum speed-up and then trim only the remaining excess from the end.

Each instruction retains the segment ID, source WAV path, timeline start/end, target language, and TTS metadata. It also reports the target duration, signed duration difference, playback rate, end padding, end trim, and expected duration. This contract is independent of VoxCPM and is designed for a later audio-processing and mixing backend.

Phase 8 performs calculations only. It does not modify WAV files, insert silence, time-stretch speech, trim audio, mix a timeline, or invoke FFmpeg.

## Current feature: aligned audio preparation

Phase 9 materializes a validated Phase 8 plan as new WAV files under `outputs/aligned/`. Original VoxCPM WAV files are never overwritten. Exact-duration clips are copied, short clips are padded with silence, moderate overruns are accelerated, and severe overruns use the Phase 8 speed limit before trimming the remaining excess.

Processing uses local FFmpeg filters. The `atempo` filter changes speech duration while preserving pitch as much as practical; longer rates are split into supported filter stages. Padding and trimming are followed by an exact target-duration boundary to account for sample rounding. Output is uncompressed PCM WAV.

Every processed result retains the segment ID, source and processed paths, original timeline start/end, target language, sample rate, channel count/layout, original TTS metadata, and deterministic processing metadata. FFprobe verifies the source against the plan and confirms output duration and stream properties. These independent files are ready for the Phase 10 mixer to place at their original timeline starts.

## Current feature: timeline placement and audio mixing

Phase 10 consumes only the backend-neutral Phase 9 `ProcessedAudioSegment` contract. It places each aligned WAV at its original `timeline_start`, preserves explicit silence before and between clips, and safely sums overlaps in stable Phase 9 input order. A final limiter reduces clipping risk without changing the deterministic placement plan.

The output is one continuous PCM 16-bit WAV under `outputs/mixed/`. Its duration is the last dubbed segment end in dubbed-only mode. When original audio is enabled, the duration is the later of the dubbed timeline end and original-media duration, producing a complete soundtrack ready to attach to video in Phase 11.

Three local modes are supported:

- dubbed speech only;
- dubbed speech plus original media audio at a configurable volume;
- dubbed speech plus original audio with sidechain compression that ducks the original while dubbed speech is active.

FFmpeg resamples and remaps inputs to one deterministic output stream, while FFprobe validates the final duration, sample rate, and channel count. Processed segment WAVs and original media are never overwritten. Phase 10 creates audio only; it does not attach the soundtrack to a video or render video frames.

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
10. In **Timing Alignment**, choose a duration tolerance and maximum speed-up, then click **Calculate timing plan** to review the proposed operations.
11. Click **Process aligned WAV files** to create non-destructive, duration-adjusted PCM WAV copies.
12. In **Timeline placement and audio mixing**, choose dubbed-only output or include the uploaded media's original audio. Configure source volume and optional ducking, then click **Mix dubbed audio timeline**.
13. Review the continuous WAV under `outputs/mixed/`. It is ready for Phase 11 video attachment.

## Known limitations

- Accurate decoding requires FFmpeg on the system PATH.
- The first transcription for each selected model needs an internet connection to download that model.
- Larger models improve accuracy but use more RAM, disk space, and processing time.
- Argos package coverage and translation quality vary by language pair; only installed direct pairs are available.
- Translation requires a completed non-empty transcription and different source/target languages.
- VoxCPM2 is a large 2B-parameter model; generation can be slow or memory-intensive without a compatible GPU.
- Basic VoxCPM2 voice design is supported, but reference-audio voice cloning is not exposed yet.
- Phases 9 and 10 require local `ffmpeg` and `ffprobe` executables on PATH.
- Phase 10 produces a complete WAV soundtrack but does not attach it to the original video or render video; that remains Phase 11.
- Overlapping dubbed clips are summed and limited deterministically. Dense overlaps may still sound crowded and can trigger audible limiting.
- Ducking uses the complete dubbed timeline as a sidechain key. It does not perform stem separation, so it lowers all original audio—including dialogue, music, and effects—during dubbed speech.
- The mixer currently emits mono or stereo PCM 16-bit WAV and derives its default format from the first processed segment.
- Severe overruns are trimmed from the end after the configured maximum speed-up; review the Phase 8 plan before processing.
- FFmpeg `atempo` is pitch-preserving, but aggressive speed changes can still reduce perceived speech quality.

## Project layout

- `src/local_dubbing/stt/` — modular STT data models, formatter, engine boundary, and faster-whisper adapter.
- `src/local_dubbing/translation/` — modular models, engine abstraction, Argos adapter, package discovery, and export helpers.
- `src/local_dubbing/tts/` — structured TTS models, backend manager, engine abstraction, and lazy VoxCPM adapter.
- `src/local_dubbing/audio/` — backend-neutral timing plans, non-destructive WAV processing, and deterministic FFmpeg timeline mixing.
- `app.py` — Streamlit speech-to-text, translation, TTS, timing, aligned-audio, and soundtrack-mixing interface.
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
4. Improve timeline-mixing controls with optional per-track automation and richer loudness metering.
5. Attach the validated Phase 10 soundtrack to the original video and render the dubbed output with FFmpeg.

## License

This project is released under the [MIT License](LICENSE).
