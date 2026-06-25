# Architecture

AudioCover Lab has two user-facing entry points:

- CLI: `audiocover`
- Windows GUI: `audiocover-gui`

The pipeline has six stages:

```text
recording folder -> dataset preparation -> model package
song file -> normalization -> separation -> conversion -> mixing -> reports
```

## Model packages

A model package is a directory containing `model.yaml`. For built-in training it also contains `simple_timbre.json`. For external training it may contain `.pth`, `.index`, or backend-specific assets.

## Training backends

### simple-timbre

A CI-testable fallback backend. It learns a spectral target profile from the supplied data and applies that profile to an input vocal stem. It is not intended to match a serious RVC/SVC model, but it verifies the complete product path.

### external

Runs user-configured commands. This is the intended production path for RVC, Seed-VC, So-VITS-SVC, or future SVC engines.

## Render backends

- `demucs`: default source separation if Demucs is installed.
- `external`: custom source separation command.
- `none`: debug mode.

Conversion supports:

- `external`
- `simple-timbre`
- `passthrough`
