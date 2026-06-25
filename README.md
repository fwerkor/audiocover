# AudioCover Lab

AudioCover Lab is a Windows-friendly desktop and CLI project for **path-A audio cover rendering**:

1. train or register a personal target-singing profile;
2. separate an existing song into vocal and instrumental stems;
3. convert the separated vocal through a selected model/backend;
4. polish and mix the result;
5. export `final_mix.wav` with intermediate stems and QC reports.

The project includes:

- `audiocover` CLI
- `audiocover-gui` Tkinter desktop GUI
- dataset preparation and model-package format
- built-in lightweight `simple-timbre` trainer/converter for local testing and fallback
- external RVC/Seed-VC/So-VITS-style backend adapters via command templates
- PyInstaller build specs
- GitHub Actions CI
- GitHub Actions release workflow that builds a Windows `.exe`

## Scope

The built-in `simple-timbre` backend is intentionally lightweight and exists so the full training/rendering/GUI/release pipeline can run in CI without GPU or third-party model weights. For highest quality, use the external backend adapter to connect a serious SVC/RVC training and inference implementation.

No songs, private datasets, model weights, or third-party model repositories are included.

## Install for development

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Install FFmpeg and ensure it is available in `PATH`.

## Launch Windows GUI from source

```bash
audiocover-gui
```

Or:

```bash
python -m audiocover_lab.gui
```

## GUI workflow

### Train tab

1. Select a folder containing your own dry recordings.
2. Select an output model directory.
3. Choose backend:
   - `simple-timbre`: built-in fallback model, no GPU required.
   - `external`: calls configured training commands for RVC/Seed-VC/other tools.
4. Tick the consent checkbox.
5. Click **Start training**.

The output directory will contain:

```text
model.yaml
simple_timbre.json       # for simple-timbre backend
dataset/
  wavs/
  report.json
  manifest.jsonl
```

### Render tab

1. Select a song file.
2. Select an existing `model.yaml`.
3. Select output directory.
4. Tick the consent checkbox.
5. Click **Generate cover**.

Outputs:

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

Train built-in model:

```bash
audiocover train data/my_recordings models/my_profile --backend simple-timbre --consent
```

Render using a trained model package:

```bash
audiocover render song.mp3 --model models/my_profile/model.yaml --out runs/song --overwrite --consent
```

Launch GUI:

```bash
audiocover-gui
```

Run environment check:

```bash
audiocover doctor
```

## External backend configuration

For high-quality production use, create a profile/training config with external command templates.

Example `model.yaml` conversion section:

```yaml
conversion:
  backend: external
  command_template: >
    python C:/tools/rvc_cli/infer.py
    --input {input}
    --output {output}
    --model {model}
    --index {index}
    --f0_method {f0_method}
    --transpose {transpose}
```

Placeholders:

- `{input}`: separated vocal path
- `{output}`: converted vocal output path
- `{model}`: model file path
- `{index}`: index file path
- `{f0_method}`: pitch method, for example `rmvpe`
- `{transpose}`: semitone shift
- `{workdir}`: run directory
- `{protect}`, `{index_rate}`, `{rms_mix_rate}`: optional conversion parameters

External training supports command stages:

```yaml
training:
  backend: external
  commands:
    - python C:/tools/rvc_cli/preprocess.py --dataset {dataset} --out {workdir}
    - python C:/tools/rvc_cli/train.py --dataset {dataset} --out {workdir} --epochs {epochs}
    - python C:/tools/rvc_cli/build_index.py --model {model} --out {index}
```

Placeholders:

- `{raw}`: raw data directory
- `{dataset}`: prepared dataset wav directory
- `{workdir}`: output model directory
- `{model}`: expected model path
- `{index}`: expected index path
- `{epochs}`, `{batch_size}`, `{sample_rate}`

## Build local package

```bash
python -m pip install -e ".[build]"
python -m build
```

## Build Windows GUI executable

On Windows:

```powershell
python -m pip install -e ".[build]"
pyinstaller packaging/audiocover-gui.spec --clean --noconfirm
```

Output:

```text
dist/AudioCoverLab/AudioCoverLab.exe
```

## GitHub release

Push a tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The release workflow will:

1. run tests;
2. build wheel and source distribution;
3. build Windows GUI executable with PyInstaller;
4. upload artifacts;
5. create a GitHub Release.

## Recording format

Recommended input data:

- WAV/FLAC/MP3/M4A accepted by FFmpeg
- dry voice, no background music
- 48 kHz preferred
- no clipping
- 30-60 min speech + 1-2 h singing minimum for serious external training
- enough high notes/falsetto/breathy samples for the songs you want to cover

## Responsible use

Use only recordings and target profiles you own or are authorized to use. Do not use the tool to impersonate people without permission, mislead listeners, or distribute copyrighted songs without rights.
