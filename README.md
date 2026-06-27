# AudioCover

AudioCover is a desktop GUI and CLI for local audio-cover workflows. It prepares authorized voice data, builds a model package, renders a cover from a source song, and writes intermediate files plus JSON reports.

## Features

- Desktop GUI for training data, model package, song input, output folder, and rights confirmation.
- CPU-only release binaries for Windows, Linux, and macOS.
- Single-binary desktop delivery: no external `_internal` folder and no separate `backend-runtimes` pack for normal desktop use.
- Automatic backend selection without a user-facing backend picker.
- Automatic octave-level pitch range adaptation from the trained voice profile.
- CLI commands for dataset preparation, training, rendering, quality checks, and diagnostics.
- Reproducible output folders with manifests, quality-control reports, and intermediate audio files.

## Download desktop builds

Download the single binary for your platform from the release page.

| Platform | Artifact | Notes |
| --- | --- | --- |
| Windows x64 | `audiocover-windows-x64.exe` | CPU-only |
| Linux x64 | `audiocover-linux-x64` | CPU-only; make executable before running |
| Linux arm64 | `audiocover-linux-arm64` | CPU-only; make executable before running |
| macOS arm64 | `audiocover-macos-arm64` | CPU-only; make executable before running |

The Python wheel and source archive are for package/developer use. Desktop users normally only need the platform binary.

## Run desktop builds

### Windows x64

```powershell
mkdir C:\AudioCover
Move-Item .\audiocover-windows-x64.exe C:\AudioCover\AudioCover.exe
C:\AudioCover\AudioCover.exe
```

If SmartScreen appears, choose **More info** and **Run anyway**.

### Linux x64 / Linux arm64

```bash
mkdir -p ~/AudioCover
cp ~/Downloads/audiocover-linux-x64 ~/AudioCover/AudioCover
chmod +x ~/AudioCover/AudioCover
~/AudioCover/AudioCover
```

For Linux arm64, replace `audiocover-linux-x64` with `audiocover-linux-arm64`.

### macOS arm64

```bash
mkdir -p ~/AudioCover
cp ~/Downloads/audiocover-macos-arm64 ~/AudioCover/AudioCover
chmod +x ~/AudioCover/AudioCover
~/AudioCover/AudioCover
```

If macOS Gatekeeper blocks the binary, open **System Settings → Privacy & Security** and allow it, or run:

```bash
xattr -d com.apple.quarantine ~/AudioCover/AudioCover
~/AudioCover/AudioCover
```

## Single-file behavior

Release binaries are PyInstaller one-file executables. Backend code and pinned So-VITS-SVC model assets are embedded in the binary. At launch, the executable extracts its runtime into the operating system temporary directory. Embedded worker subprocesses are launched through the same executable and reuse the extracted application home while the parent process is alive.

Practical implications:

- Keep enough free space in the system temporary directory. Several GB of temporary space is recommended for training or rendering.
- First launch can be slower because native libraries and model assets need to be unpacked.
- Antivirus or endpoint protection can slow the first launch on Windows.
- Do not delete the temporary extraction directory while AudioCover is running.
- Release binaries force CPU execution. CUDA/MPS GPU execution is intentionally only supported from source.

## GUI workflow

The GUI expects:

- **Training data:** a folder of authorized `.wav`, `.flac`, `.mp3`, `.m4a`, `.aac`, or `.ogg` recordings of one target voice.
- **Model package:** an AudioCover `model.yaml` produced by training or prepared for an existing backend model.
- **Song input:** an authorized audio file to render.
- **Output folder:** a directory for generated audio, intermediates, and reports.

For training data, prefer clean single-source recordings, stable microphone placement, minimal echo/noise, no backing music, and no clipping. 48 kHz WAV is preferred.

Recommended dataset layout:

```text
my_voice_dataset/
  001.wav
  002.wav
  003.wav
```

Training produces a model package and pitch profile such as:

```text
models/my_voice/model.yaml
models/my_voice/voice_profile.json
```

`voice_profile.json` stores the target voice F0 range. During rendering, AudioCover analyzes the separated input vocal and automatically chooses an octave-level pitch shift from `-12`, `0`, or `+12` semitones when the selected backend supports transpose. The render report records the decision in `reports/auto_pitch.json`.

Rendering uses that `model.yaml` plus a song file and writes the final audio, intermediates, and JSON reports into the selected output folder.

## CLI

```bash
audiocover train data/my_recordings models/my_profile --consent
audiocover render track.mp3 --model models/my_profile/model.yaml --out runs/track --consent
audiocover doctor
```

## Run from source

Source runs are intended for development and for users who want GPU acceleration. Use Python 3.10-3.12.

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/fwerkor/audiocover.git
cd audiocover
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

### 2. Install PyTorch

For CPU source runs:

```bash
python -m pip install -U pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch torchaudio
```

For NVIDIA GPU source runs, install the CUDA wheel matching your driver and CUDA runtime. Example for CUDA 13.0 wheels:

```bash
python -m pip install -U pip
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchaudio
```

For macOS Apple Silicon GPU/MPS source runs, install the normal PyPI wheels:

```bash
python -m pip install -U pip
python -m pip install torch torchaudio
```

Verify the device:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda build:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
print('mps available:', getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available())
PY
```

### 3. Install AudioCover dependencies

```bash
python -m pip install -e ".[full,so-vits-svc-backend]"
```

For development checks, use:

```bash
python -m pip install -e ".[dev,full,so-vits-svc-backend]"
```

### 4. Prepare pinned model assets

So-VITS-SVC uses pinned ContentVec and initialization checkpoint assets. Download them once:

```bash
python scripts/build_desktop.py --prepare-assets-only
```

This stores assets under:

```text
build/audiocover-bundle-assets/
  content-vec-best/
  so-vits-svc-init/
```

AudioCover source workers look there automatically. You can also store the same layout elsewhere and set:

```bash
# Windows PowerShell
$env:AUDIOCOVER_ASSETS_DIR = "D:\AudioCoverAssets"

# Linux/macOS
export AUDIOCOVER_ASSETS_DIR=/opt/audiocover-assets
```

### 5. Run

GUI:

```bash
python -m audiocover.gui
```

CLI:

```bash
audiocover train data/my_recordings models/my_profile --consent
audiocover render track.mp3 --model models/my_profile/model.yaml --out runs/track --consent
audiocover doctor
```

By default, source runs select the best available PyTorch device. To force CPU or CUDA for training/config files, set `device: cpu`, `device: cuda`, or `device: cuda:0` in the relevant config.

## Build release-style binaries locally

CPU-only single-file build:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch torchaudio
python -m pip install -e ".[build,demucs-backend,so-vits-svc-backend]"
python scripts/build_desktop.py
```

The output is written to `dist/audiocover-<platform>` or `dist/audiocover-<platform>.exe`.

## Responsible use

Only train on recordings and render songs that you own or are authorized to use. The GUI and CLI both require explicit confirmation before training or rendering.
