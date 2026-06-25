from __future__ import annotations

import json
from pathlib import Path

from .config import ConversionConfig, ModelPackage, TrainingConfig
from .dataset import prepare_dataset
from .external import run_template
from .simple_timbre import train_simple_timbre


def train_model(
    raw_data_dir: Path,
    output_dir: Path,
    *,
    display_name: str,
    config: TrainingConfig,
    consent: bool,
) -> ModelPackage:
    if not consent:
        raise PermissionError("training requires explicit consent confirmation")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"
    report = prepare_dataset(
        raw_data_dir,
        dataset_dir,
        segment_seconds=config.segment_seconds,
        sample_rate=config.sample_rate,
    )
    if not report["items"]:
        raise RuntimeError("dataset preparation produced no usable audio segments")

    if config.backend == "simple-timbre":
        simple_profile = train_simple_timbre(dataset_dir / "wavs", output_dir / "simple_timbre.json", config.sample_rate)
        package = ModelPackage(
            display_name=display_name,
            training=config,
            conversion=ConversionConfig(
                backend="simple-timbre",
                simple_profile_path=simple_profile.name,
                f0_method=config.f0_method,
            ),
            simple_profile_path=simple_profile.name,
            f0_method=config.f0_method,
            notes="Built-in lightweight model. Use external RVC/SVC training for best quality.",
        )
    elif config.backend == "external":
        model_path = output_dir / "model.pth"
        index_path = output_dir / "model.index"
        for i, command in enumerate(config.commands):
            run_template(
                command,
                log_file=output_dir / f"external_train_{i:02d}.log",
                raw=raw_data_dir,
                dataset=dataset_dir / "wavs",
                workdir=output_dir,
                model=model_path,
                index=index_path,
                epochs=config.epochs,
                batch_size=config.batch_size,
                sample_rate=config.sample_rate,
            )
        package = ModelPackage(
            display_name=display_name,
            training=config,
            conversion=ConversionConfig(
                backend="external",
                model_path=model_path.name if model_path.exists() else None,
                index_path=index_path.name if index_path.exists() else None,
                f0_method=config.f0_method,
            ),
            model_path=model_path.name if model_path.exists() else None,
            index_path=index_path.name if index_path.exists() else None,
            f0_method=config.f0_method,
            notes="External backend model package. Fill conversion.command_template if the training tool did not write it.",
        )
    else:
        raise ValueError(f"unknown training backend: {config.backend}")

    model_yaml = output_dir / "model.yaml"
    package.write_yaml(model_yaml)
    (output_dir / "training_report.json").write_text(
        json.dumps({"model_yaml": str(model_yaml), "dataset_report": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ModelPackage.from_yaml(model_yaml)
