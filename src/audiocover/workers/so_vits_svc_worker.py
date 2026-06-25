from __future__ import annotations

import importlib.util
from typing import Any

from audiocover.workers.json_worker import serve


def _available() -> tuple[bool, str | None]:
    if importlib.util.find_spec("so_vits_svc_fork") is None:
        return False, "so-vits-svc-fork is not present in this isolated worker runtime"
    return True, None


def capabilities(_: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    return {
        "available": available,
        "actions": ["train", "convert"] if available else [],
        "description": "So-VITS-SVC isolated training and inference worker",
        "reason": reason,
    }


def train(payload: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    if not available:
        raise RuntimeError(reason)

    import json
    import shutil
    from pathlib import Path

    from so_vits_svc_fork.__main__ import pre_config, pre_hubert, pre_resample
    from so_vits_svc_fork.__main__ import train as svc_train

    dataset_wavs = Path(payload["dataset_wavs"])
    output_dir = Path(payload["output_dir"])
    speaker = str(payload.get("display_name") or "speaker")
    workdir = output_dir / "so-vits-svc"
    raw_speaker = workdir / "dataset_raw" / speaker
    raw_speaker.mkdir(parents=True, exist_ok=True)
    for item in dataset_wavs.glob("*.wav"):
        shutil.copyfile(item, raw_speaker / item.name)

    dataset_dir = workdir / "dataset" / "44k"
    filelist_dir = workdir / "filelists" / "44k"
    config_path = workdir / "configs" / "44k" / "config.json"
    model_dir = workdir / "logs" / "44k"
    sample_rate = int(payload.get("sample_rate") or 44100)
    f0_method = str(payload.get("f0_method") or "dio")
    if f0_method not in {"crepe", "crepe-tiny", "parselmouth", "dio", "harvest"}:
        f0_method = "dio"

    pre_resample.callback(
        input_dir=workdir / "dataset_raw",
        output_dir=dataset_dir,
        sampling_rate=sample_rate,
        n_jobs=1,
        top_db=30,
        frame_seconds=1,
        hop_seconds=0.3,
    )
    pre_config.callback(
        input_dir=dataset_dir,
        filelist_path=filelist_dir,
        config_path=config_path,
        config_type="so-vits-svc-4.0v1",
    )
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        train_cfg = data.setdefault("train", {})
        train_cfg["epochs"] = int(payload.get("epochs") or train_cfg.get("epochs") or 200)
        train_cfg["batch_size"] = int(payload.get("batch_size") or train_cfg.get("batch_size") or 8)
        config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    pre_hubert.callback(
        input_dir=dataset_dir,
        config_path=config_path,
        n_jobs=1,
        force_rebuild=True,
        f0_method=f0_method,
    )
    svc_train.callback(
        config_path=config_path,
        model_path=model_dir,
        tensorboard=False,
        reset_optimizer=False,
    )
    candidates = sorted(model_dir.glob("G_*.pth"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"So-VITS-SVC training did not produce a G_*.pth model in {model_dir}")
    model_path = candidates[-1]
    return {
        "backend": "so-vits-svc",
        "conversion_backend": "managed",
        "model_path": str(model_path),
        "config_path": str(config_path),
        "speaker": speaker,
        "notes": "So-VITS-SVC model produced by the packaged isolated runtime.",
    }


def convert(payload: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    if not available:
        raise RuntimeError(reason)

    from pathlib import Path

    from so_vits_svc_fork.inference.main import infer

    model_path = payload.get("model_path")
    config_path = payload.get("config_path")
    speaker = payload.get("speaker") or "speaker"
    if not model_path or not config_path:
        raise RuntimeError("So-VITS-SVC conversion requires model_path and config_path in the model package")
    output_path = Path(payload["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f0_method = str(payload.get("f0_method") or "dio")
    if f0_method not in {"crepe", "crepe-tiny", "parselmouth", "dio", "harvest"}:
        f0_method = "dio"
    infer(
        input_path=Path(payload["input"]),
        output_path=output_path,
        model_path=Path(model_path),
        config_path=Path(config_path),
        speaker=speaker,
        cluster_model_path=Path(payload["cluster_model_path"]) if payload.get("cluster_model_path") else None,
        transpose=int(payload.get("transpose") or 0),
        auto_predict_f0=True,
        cluster_infer_ratio=float(payload.get("cluster_infer_ratio") or 0),
        noise_scale=float(payload.get("noise_scale") or 0.4),
        f0_method=f0_method,
        db_thresh=int(payload.get("db_thresh") or -40),
        pad_seconds=float(payload.get("pad_seconds") or 0.5),
        chunk_seconds=float(payload.get("chunk_seconds") or 0.5),
        device=payload.get("device") or "cpu",
    )
    return {"output": str(output_path)}


def main() -> None:
    serve({"capabilities": capabilities, "train": train, "convert": convert})


if __name__ == "__main__":
    main()
