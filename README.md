# AudioCover

Desktop GUI and CLI.

The GUI now keeps normal choices small:

- data folder
- output profile folder
- source file
- profile YAML
- output folder
- permission confirmation

Implementation details are selected internally. Normal users do not choose backend, device, epoch count, batch size, or render config.

## Usage

```bash
audiocover-gui
audiocover train data/my_recordings models/my_profile --consent
audiocover render track.mp3 --model models/my_profile/model.yaml --out runs/track --consent
audiocover doctor
```

## Data requirements

Use a folder containing `.wav`, `.flac`, `.mp3`, `.m4a`, `.aac`, or `.ogg` files. Prefer clean authorized data, stable capture conditions, minimal echo/noise, no backing music, and no clipping. 48 kHz WAV is preferred.
