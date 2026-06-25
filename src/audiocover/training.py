from __future__ import annotations

import json
from pathlib import Path

from .config import ConversionConfig, ModelPackage, TrainingConfig, resolve_training_config
from .dataset import prepare_dataset
from .external import run_template
from .runtime import BackendRuntimeManager
from .simple_timbre import train_simple_timbre


def _relative_to_output(path: str | None, output_dir: Path) -> Path | None:
    if not path:
        return None
    value = Path(path)
    if not value.is_absolute():
        return value
    try:
        return value.relative_to(output_dir)
    except ValueError:
        return value


def _package_from_runtime(
    *,
    display_name: str,
    config: TrainingConfig,
    output_dir: Path,
    runtime_backend: str,
    result: dict,
) -> ModelPackage:
    simple_profile = _relative_to_output(result.get("simple_profile_path"), output_dir)
    model_path = _relative_to_output(result.get("model_path"), output_dir)
    index_path = _relative_to_output(result.get("index_path"), output_dir)
    conversion_backend = str(result.get("conversion_backend") or "managed")
    if conversion_backend == "simple-timbre":
        conversion_backend = "managed"
    conversion = ConversionConfig(
        backend=conversion_backend,
        runtime_backend=runtime_backend,
        model_path=model_path,
        index_path=index_path,
        simple_profile_path=simple_profile,
        f0_method=config.f0_method,
    )
    return ModelPackage(
        display_name=display_name,
        training=config,
        conversion=conversion,
        runtime_backend=runtime_backend,
        model_path=model_path,
        index_path=index_path,
        simple_profile_path=simple_profile,
        f0_method=config.f0_method,
        notes=result.get("notes"),
    )


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
    config = resolve_training_config(config)
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

    if config.backend == "managed":
        manager = BackendRuntimeManager()
        runtime_backend = (
            manager.require_training_backend(config.runtime_backend)
            if config.runtime_backend
            else manager.select_training_backend()
        )
        result = manager.invoke(
            runtime_backend,
            "train",
            {
                "raw_data_dir": str(raw_data_dir),
                "dataset_dir": str(dataset_dir),
                "dataset_wavs": str(dataset_dir / "wavs"),
                "output_dir": str(output_dir),
                "display_name": display_name,
                "sample_rate": config.sample_rate,
                "segment_seconds": config.segment_seconds,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "f0_method": config.f0_method,
            },
        )
        package = _package_from_runtime(
            display_name=display_name,
            config=config,
            output_dir=output_dir,
            runtime_backend=runtime_backend,
            result=result,
        )
    elif config.backend == "simple-timbre":
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
            notes="Built-in lightweight model. Use a managed runtime package for backend-specific models.",
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
            notes="External backend model package.",
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
