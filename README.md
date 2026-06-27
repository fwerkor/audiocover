# AudioCover

AudioCover is a desktop GUI and CLI for local audio-cover workflows. It prepares authorized voice data, builds a model package, renders a cover from a source song, and writes intermediate files plus JSON reports.

## Features

- Desktop GUI for training data, model package, song input, output folder, and rights confirmation.
- Automatic backend runtime selection without a user-facing backend picker.
- Automatic octave-level pitch range adaptation from the trained voice profile.
- Isolated backend workers that communicate with the main app through JSON stdin/stdout.
- Split backend runtime packs for large or conflicting engines.
- CLI commands for dataset preparation, training, rendering, quality checks, and diagnostics.
- Reproducible output folders with manifests, quality-control reports, and intermediate audio files.

## Download

Use the desktop artifact for your platform, then add the matching backend runtime packs.

| Platform | Desktop artifact | Runtime packs |
| --- | --- | --- |
| Windows x64 | `audiocover-windows-x64.zip` | `audiocover-backend-runtimes-demucs-windows-x64.zip`, `audiocover-backend-runtimes-so-vits-svc-windows-x64.zip` |
| Linux x64 | `audiocover-linux-x64.tar.gz` | `audiocover-backend-runtimes-demucs-linux-x64.tar.gz`, `audiocover-backend-runtimes-so-vits-svc-linux-x64.tar.gz` |
| Linux arm64 | `audiocover-linux-arm64.tar.gz` | `audiocover-backend-runtimes-demucs-linux-arm64.tar.gz`, `audiocover-backend-runtimes-so-vits-svc-linux-arm64.tar.gz` |
| macOS arm64 | `audiocover-macos-arm64.tar.gz` | `audiocover-backend-runtimes-demucs-macos-arm64.tar.gz`, `audiocover-backend-runtimes-so-vits-svc-macos-arm64.tar.gz` |

The Python wheel and source archive are for package/developer use. Desktop users normally do not need them.

## Install desktop builds

Create one install directory, extract the desktop artifact into it, then extract runtime packs into the same install directory.

### Windows x64

```powershell
mkdir C:\AudioCover
Expand-Archive .\audiocover-windows-x64.zip C:\AudioCover
Expand-Archive .\audiocover-backend-runtimes-demucs-windows-x64.zip C:\AudioCover
Expand-Archive .\audiocover-backend-runtimes-so-vits-svc-windows-x64.zip C:\AudioCover
C:\AudioCover\AudioCover\AudioCover.exe
```

Expected layout:

```text
C:\AudioCover\AudioCover\AudioCover.exe
C:\AudioCover\AudioCover\_internal\...
C:\AudioCover\backend-runtimes\demucs-separator\demucs-separator.exe
C:\AudioCover\backend-runtimes\so-vits-svc\so-vits-svc.exe
```

If SmartScreen appears, choose **More info** and **Run anyway**.

### Linux x64 / Linux arm64

```bash
mkdir -p ~/AudioCover
cd ~/AudioCover
tar -xzf ~/Downloads/audiocover-linux-x64.tar.gz
tar -xzf ~/Downloads/audiocover-backend-runtimes-demucs-linux-x64.tar.gz
tar -xzf ~/Downloads/audiocover-backend-runtimes-so-vits-svc-linux-x64.tar.gz
./AudioCover/AudioCover
```

For Linux arm64, use the `linux-arm64` desktop and runtime pack names instead.

Expected layout:

```text
~/AudioCover/AudioCover/AudioCover
~/AudioCover/AudioCover/_internal/...
~/AudioCover/backend-runtimes/demucs-separator/demucs-separator
~/AudioCover/backend-runtimes/so-vits-svc/so-vits-svc
```

### macOS arm64

```bash
mkdir -p ~/AudioCover
cd ~/AudioCover
tar -xzf ~/Downloads/audiocover-macos-arm64.tar.gz
tar -xzf ~/Downloads/audiocover-backend-runtimes-demucs-macos-arm64.tar.gz
tar -xzf ~/Downloads/audiocover-backend-runtimes-so-vits-svc-macos-arm64.tar.gz
open ./AudioCover.app
```

Expected layout:

```text
~/AudioCover/AudioCover.app
~/AudioCover/backend-runtimes/demucs-separator/demucs-separator
~/AudioCover/backend-runtimes/so-vits-svc/so-vits-svc
```

If macOS Gatekeeper blocks the app, open **System Settings → Privacy & Security** and allow the app, or run it once from Terminal:

```bash
xattr -dr com.apple.quarantine ~/AudioCover/AudioCover.app
open ~/AudioCover/AudioCover.app
```

## Split runtime pack files

Large Linux runtime packs may be published as parts, for example:

```text
audiocover-backend-runtimes-so-vits-svc-linux-x64.tar.gz.part001
audiocover-backend-runtimes-so-vits-svc-linux-x64.tar.gz.part002
audiocover-backend-runtimes-so-vits-svc-linux-x64.tar.gz.sha256
```

Download all parts for that runtime pack into the same directory, then join them before extraction.

Windows PowerShell:

```powershell
cmd /c copy /b "audiocover-backend-runtimes-so-vits-svc-linux-x64.tar.gz.part001"+"audiocover-backend-runtimes-so-vits-svc-linux-x64.tar.gz.part002" "audiocover-backend-runtimes-so-vits-svc-linux-x64.tar.gz"
```

Linux/macOS:

```bash
cat audiocover-backend-runtimes-so-vits-svc-linux-x64.tar.gz.part* > audiocover-backend-runtimes-so-vits-svc-linux-x64.tar.gz
sha256sum -c audiocover-backend-runtimes-so-vits-svc-linux-x64.tar.gz.sha256  # Linux
shasum -a 256 -c audiocover-backend-runtimes-so-vits-svc-linux-x64.tar.gz.sha256  # macOS
```

Then extract the joined `.tar.gz` file into the install directory.

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

## Backend runtimes

AudioCover uses a runtime manager in the main process and isolated worker executables under `backend-runtimes/`. Workers are invoked with JSON requests over stdin/stdout, which keeps backend dependency sets separated from the GUI process.

Desktop artifacts include the lightweight built-in worker. Runtime packs provide larger backend workers and any packaged model assets required by those workers. The runtime manager discovers `backend-runtimes/` in the install directory, beside the executable, or through `AUDIOCOVER_BACKEND_RUNTIMES`. Python, pip, and backend command-line tools do not need to be installed by desktop users.

Backend workers do not silently download missing model assets during normal desktop use. If a required asset is missing, install the matching runtime pack. Developers who intentionally want legacy runtime downloads can set `AUDIOCOVER_ALLOW_RUNTIME_DOWNLOADS=1`.

## Responsible use

Only train on recordings and render songs that you own or are authorized to use. The GUI and CLI both require explicit confirmation before training or rendering.
