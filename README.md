# AudioCover

<p align="center">
  <strong>Desktop GUI and CLI for training a voice profile and rendering audio covers.</strong>
</p>

<p align="center">
  <a href="https://github.com/fwerkor/audiocover/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/fwerkor/audiocover/actions/workflows/ci.yml/badge.svg"></a>
</p>

AudioCover packages a practical cover-rendering workflow:

1. prepare authorized voice recordings;
2. train or register a target voice profile;
3. split a song into vocal and instrumental stems;
4. convert the vocal through a selected backend;
5. polish, mix, and export `final_mix.wav` with reports and intermediate files.

It includes a Tkinter desktop GUI, a command-line interface, a model-package format, automated tests, and a PyInstaller build path. The built-in `simple-timbre` backend is a lightweight local fallback for testing the full pipeline without GPU-only dependencies. For higher quality results, connect an external RVC, Seed-VC, So-VITS, or similar backend with command templates.

No songs, private datasets, model weights, or third-party model repositories are included.

## Features

- Desktop GUI: `audiocover-gui`
- CLI entry point: `audiocover`
- Dataset preparation with audio QC reports
- Reproducible `model.yaml` profile packages
- Built-in CPU-testable fallback backend
- External backend adapters for serious training and inference stacks
- PyInstaller packaging for desktop bundles

## Install for development

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Install FFmpeg and ensure `ffmpeg` is available in `PATH`.

## Launch the GUI

```bash
audiocover-gui
```

or:

```bash
python -m audiocover.gui
```

## GUI workflow

### Train

1. Select a folder containing authorized dry voice recordings.
2. Select an output model directory.
3. Choose a backend:
   - `simple-timbre`: built-in fallback, CPU-only, useful for local tests and smoke runs.
   - `external`: runs configured commands for RVC, Seed-VC, So-VITS, or another backend.
4. Confirm that you have permission to use the recordings.
5. Click **Start training**.

Output:

```text
model.yaml
simple_timbre.json       # simple-timbre backend only
dataset/
  wavs/
  report.json
  manifest.jsonl
```

### Generate cover

1. Select a song file.
2. Select a `model.yaml` profile.
3. Select an output directory.
4. Confirm that you have permission to use the song and model.
5. Click **Generate cover**.

Output:

```text
input/input.wav
stems/vocals.wav
stems/instrumental.wav
converted/converted_vocal.wav
mix/polished_vocal.wav
mix/final_mix.wav
reports/qc.json
manifest.json
```

## CLI examples

Train the built-in fallback profile:

```bash
audiocover train data/my_recordings models/my_profile --backend simple-timbre --consent
```

Render a cover:

```bash
audiocover render song.mp3 --model models/my_profile/model.yaml --out runs/song --overwrite --consent
```

Run an environment check:

```bash
audiocover doctor
```

## External backend configuration

External backends are configured with command templates. This keeps AudioCover independent from any specific voice-conversion framework while still supporting serious training and inference implementations.

Example conversion section in `model.yaml`:

```yaml
conversion:
  backend: external
  command_template: >
    python path/to/rvc_cli/infer.py
    --input {input}
    --output {output}
    --model {model}
    --index {index}
    --f0_method {f0_method}
    --transpose {transpose}
```

Common placeholders:

| Placeholder | Meaning |
| --- | --- |
| `{input}` | separated vocal path |
| `{output}` | converted vocal output path |
| `{model}` | model file path |
| `{index}` | index file path |
| `{f0_method}` | pitch method, for example `rmvpe` |
| `{transpose}` | semitone shift |
| `{workdir}` | run directory |
| `{protect}` / `{index_rate}` / `{rms_mix_rate}` | optional conversion parameters |

Example external training config:

```yaml
training:
  backend: external
  commands:
    - python path/to/rvc_cli/preprocess.py --dataset {dataset} --out {workdir}
    - python path/to/rvc_cli/train.py --dataset {dataset} --out {workdir} --epochs {epochs}
    - python path/to/rvc_cli/build_index.py --model {model} --out {index}
```

Training placeholders:

| Placeholder | Meaning |
| --- | --- |
| `{raw}` | raw data directory |
| `{dataset}` | prepared dataset WAV directory |
| `{workdir}` | output model directory |
| `{model}` | expected model path |
| `{index}` | expected index path |
| `{epochs}` / `{batch_size}` / `{sample_rate}` | training settings |

## Build

Build Python distributions:

```bash
python -m pip install -e ".[build]"
python -m build
```

Build a desktop bundle:

```bash
python -m pip install -e ".[build]"
python scripts/build_desktop.py
```

Output is written under `dist/`.

## Recording recommendations

- WAV, FLAC, MP3, M4A, AAC, and OGG inputs are supported through FFmpeg
- dry voice without background music
- 48 kHz preferred
- avoid clipping and heavy room echo
- for external high-quality training, collect enough speech and singing samples to cover the target pitch range and style

## Responsible use

Use only recordings and target profiles you own or are authorized to use. Do not use this project to impersonate people without permission, mislead listeners, or distribute copyrighted songs without rights.
