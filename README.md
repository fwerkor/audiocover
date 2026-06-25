# AudioCover

AudioCover is a desktop GUI and CLI for local audio-cover workflows. It prepares authorized voice data, builds a model package, renders a cover from a source song, and writes intermediate files plus JSON reports.

## Features

- Desktop GUI for training data, model package, song input, output folder, and rights confirmation.
- Automatic backend runtime selection without a user-facing backend picker.
- Isolated backend workers that communicate with the main app through JSON stdin/stdout.
- CLI commands for dataset preparation, training, rendering, quality checks, and diagnostics.
- Reproducible output folders with manifests, quality-control reports, and intermediate audio files.

## GUI

```bash
audiocover-gui
```

The GUI expects:

- **Training data:** a folder of authorized `.wav`, `.flac`, `.mp3`, `.m4a`, `.aac`, or `.ogg` recordings of one target voice.
- **Model package:** an AudioCover `model.yaml` produced by training.
- **Song input:** an authorized audio file to render.
- **Output folder:** a directory for generated audio, intermediates, and reports.

For training data, prefer clean single-source recordings, stable microphone placement, minimal echo/noise, no backing music, and no clipping. 48 kHz WAV is preferred.

## CLI

```bash
audiocover train data/my_recordings models/my_profile --consent
audiocover render track.mp3 --model models/my_profile/model.yaml --out runs/track --consent
audiocover doctor
```

## Backend runtimes

AudioCover uses a runtime manager in the main process and isolated worker executables under `backend-runtimes/` in desktop builds. Workers are invoked with JSON requests over stdin/stdout, which keeps backend dependency sets separated from the GUI process.

Desktop release artifacts include the runtime directory produced by the build workflow. Source-tree development can also run workers as Python modules for tests and local debugging.

## Responsible use

Only train on recordings and render songs that you own or are authorized to use. The GUI and CLI both require explicit confirmation before training or rendering.