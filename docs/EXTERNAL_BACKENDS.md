# External backends

AudioCover Lab does not vendor third-party SVC/RVC engines. It invokes them through command templates.

## External training

Example:

```yaml
backend: external
sample_rate: 48000
segment_seconds: 12
epochs: 300
batch_size: 8
commands:
  - python C:/rvc/preprocess.py --input {dataset} --output {workdir}
  - python C:/rvc/train.py --workdir {workdir} --epochs {epochs} --batch-size {batch_size}
  - python C:/rvc/index.py --workdir {workdir} --output {index}
```

## External inference

In `model.yaml`:

```yaml
conversion:
  backend: external
  command_template: >
    python C:/rvc/infer.py
    --input {input}
    --output {output}
    --model {model}
    --index {index}
    --transpose {transpose}
    --f0_method {f0_method}
```

This keeps AudioCover Lab stable even when third-party CLIs change.
