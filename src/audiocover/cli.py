from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import RenderConfig, TrainingConfig, default_config_path
from .dataset import prepare_dataset as prepare_dataset_impl
from .pipeline import render_cover
from .qc import analyze_audio
from .runtime import WORKER_MODULES, BackendRuntimeManager
from .training import train_model

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
DEFAULT_RENDER_CONFIG_PATH = default_config_path()


@app.command()
def version() -> None:
    console.print(__version__)


@app.command()
def doctor() -> None:
    table = Table("component", "status", "detail")
    for binary in ["ffmpeg", "python"]:
        found = shutil.which(binary)
        table.add_row(binary, "ok" if found else "missing", found or "not found in PATH")
    for module in ["numpy", "scipy", "soundfile", "librosa", "pyloudnorm", "tkinter", "demucs"]:
        try:
            __import__(module)
            table.add_row(module, "ok", "importable")
        except Exception as exc:
            table.add_row(module, "missing", str(exc))
    manager = BackendRuntimeManager()
    for worker_name in WORKER_MODULES:
        cap = manager.capabilities(worker_name)
        status = "ok" if cap.available else "inactive"
        detail = ",".join(cap.actions) if cap.available else (cap.reason or "not available")
        table.add_row(f"runtime:{worker_name}", status, detail)
    console.print(table)


@app.command("prepare-dataset")
def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    segment_seconds: Annotated[float, typer.Option(min=2.0, max=30.0)] = 12.0,
    sample_rate: int = 48000,
) -> None:
    report = prepare_dataset_impl(input_dir, output_dir, segment_seconds=segment_seconds, sample_rate=sample_rate)
    console.print(f"accepted={len(report['items'])} rejected={len(report['rejected'])}")
    console.print(f"report={output_dir / 'report.json'}")


@app.command()
def train(
    raw_data_dir: Path,
    output_dir: Path,
    display_name: Annotated[str, typer.Option("--name")] = "my_profile",
    backend: Annotated[str, typer.Option("--backend", hidden=True)] = "auto",
    sample_rate: Annotated[int, typer.Option(hidden=True)] = 48000,
    segment_seconds: Annotated[float, typer.Option(hidden=True)] = 12.0,
    epochs: Annotated[int, typer.Option(hidden=True)] = 200,
    batch_size: Annotated[int, typer.Option(hidden=True)] = 8,
    command: Annotated[
        list[str] | None, typer.Option("--command", help="External training command. May be repeated.", hidden=True)
    ] = None,
    consent: Annotated[bool, typer.Option("--consent", help="Confirm that you own or are authorized to use the data.")] = False,
) -> None:
    cfg = TrainingConfig(
        backend=backend,
        sample_rate=sample_rate,
        segment_seconds=segment_seconds,
        epochs=epochs,
        batch_size=batch_size,
        commands=command or [],
    )
    package = train_model(raw_data_dir, output_dir, display_name=display_name, config=cfg, consent=consent)
    console.print(f"model package: {output_dir / 'model.yaml'}")
    console.print(package.model_dump_json(indent=2))


@app.command()
def render(
    input_song: Path,
    model: Annotated[Path, typer.Option("--model", "-m", help="Path to model.yaml.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output run directory.")],
    config: Annotated[Path, typer.Option("--config", "-c", hidden=True)] = DEFAULT_RENDER_CONFIG_PATH,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    consent: Annotated[bool, typer.Option("--consent", help="Confirm that you have rights to use the song/model.")] = False,
) -> None:
    cfg = RenderConfig.from_yaml(config)
    cfg.overwrite = cfg.overwrite or overwrite
    manifest = render_cover(input_song, model, out, config=cfg, consent=consent)
    console.print(f"final mix: {manifest['outputs']['final_mix']}")
    console.print(f"manifest: {out / 'manifest.json'}")


@app.command()
def qc(path: Path, json_out: Annotated[Path | None, typer.Option("--json-out")] = None) -> None:
    report = analyze_audio(path)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print_json(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    app()
