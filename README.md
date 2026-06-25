# AudioCover

AudioCover is a desktop GUI and CLI for local audio projects.

## Features

- Prepare a profile package from a data folder.
- Render from a source file and a profile package.
- Save intermediate files, final output, and JSON reports.

## GUI

```bash
audiocover-gui
```

## CLI

```bash
audiocover train data/my_recordings models/my_profile --consent
audiocover render track.mp3 --model models/my_profile/model.yaml --out runs/track --consent
audiocover doctor
```

## Training data

Use a folder containing `.wav`, `.flac`, `.mp3`, `.m4a`, `.aac`, or `.ogg` files. Prefer clean single-source recordings, stable capture conditions, minimal echo/noise, no backing music, and no clipping. 48 kHz WAV is preferred.
