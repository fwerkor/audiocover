# Advanced engine hooks

Normal GUI users do not choose an engine.

Advanced automation can add command templates to YAML files. With `backend: auto`, AudioCover uses package-specific commands when present and otherwise uses the local CPU path.

Placeholders include `{raw}`, `{dataset}`, `{workdir}`, `{model}`, `{index}`, `{input}`, `{output}`, `{epochs}`, `{batch_size}`, `{sample_rate}`, and `{transpose}`.
