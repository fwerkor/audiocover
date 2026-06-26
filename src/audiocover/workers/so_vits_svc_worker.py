from __future__ import annotations

import importlib
import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

from audiocover.workers.json_worker import serve

_REQUIRED_IMPORTS: tuple[tuple[str, str], ...] = (
    ("so_vits_svc_fork", "so-vits-svc-fork"),
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("soundfile", "soundfile"),
    ("librosa", "librosa"),
    ("sklearn", "scikit-learn"),
    ("transformers", "transformers"),
    ("joblib", "joblib"),
)


def _asset_dir(name: str, required_files: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    env_var = f"AUDIOCOVER_{name.upper().replace('-', '_')}_DIR"
    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value).expanduser())

    executable_dir = Path(sys.executable).resolve().parent
    candidates.extend(
        (
            executable_dir / "assets" / name,
            executable_dir.parent / "assets" / name,
            Path.cwd() / "backend-runtimes" / "so-vits-svc" / "assets" / name,
        )
    )

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "assets" / name)

    source_path = Path(__file__).resolve()
    candidates.extend((source_path.parent / "assets" / name, source_path.parents[3] / "backend-runtimes" / "so-vits-svc" / "assets" / name))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_dir() and all((candidate / item).is_file() for item in required_files):
            return candidate
    return None


def _contentvec_dir() -> Path | None:
    return _asset_dir("content-vec-best", ("config.json", "pytorch_model.bin"))


def _runtime_downloads_allowed() -> bool:
    return os.environ.get("AUDIOCOVER_ALLOW_RUNTIME_DOWNLOADS", "").lower() in {"1", "true", "yes"}


def _patch_contentvec_loader() -> None:
    model_dir = _contentvec_dir()
    if model_dir is None:
        if _runtime_downloads_allowed():
            print("so-vits-svc: bundled ContentVec model not found; runtime downloads are enabled", flush=True)
            return
        raise RuntimeError(
            "So-VITS-SVC ContentVec assets are missing. Install the matching "
            "audiocover-backend-runtimes-so-vits-svc pack, or set "
            "AUDIOCOVER_ALLOW_RUNTIME_DOWNLOADS=1 to allow explicit runtime downloads."
        )

    import torch.nn as nn
    from so_vits_svc_fork import utils as svc_utils
    from transformers import HubertModel

    offline_env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }
    for key, value in offline_env.items():
        os.environ.setdefault(key, value)

    def get_bundled_hubert_model(device: str, final_proj: bool = True):
        model_class = svc_utils.HubertModelWithFinalProj if final_proj else HubertModel
        model = model_class.from_pretrained(
            str(model_dir),
            local_files_only=True,
            use_safetensors=False,
        )
        model.config.name_or_path = str(model_dir)
        model.config._name_or_path = str(model_dir)
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                svc_utils.remove_weight_norm_if_exists(module)
        return model.to(device)

    svc_utils.get_hubert_model = get_bundled_hubert_model
    print(f"so-vits-svc: using bundled ContentVec model at {model_dir}", flush=True)


def _copy_bundled_init_checkpoints(model_dir: Path) -> None:
    init_dir = _asset_dir("so-vits-svc-init", ("D_0.pth", "G_0.pth"))
    if init_dir is None:
        if _runtime_downloads_allowed():
            print("so-vits-svc: bundled initialization checkpoints not found; runtime downloads are enabled", flush=True)
            return
        raise RuntimeError(
            "So-VITS-SVC initialization checkpoints are missing. Install the matching "
            "audiocover-backend-runtimes-so-vits-svc pack, or set "
            "AUDIOCOVER_ALLOW_RUNTIME_DOWNLOADS=1 to allow explicit runtime downloads."
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    copied = False
    for name in ("D_0.pth", "G_0.pth"):
        source = init_dir / name
        target = model_dir / name
        if not target.exists():
            import shutil

            shutil.copy2(source, target)
            copied = True
    if copied:
        print(f"so-vits-svc: copied bundled initialization checkpoints from {init_dir}", flush=True)
    else:
        print("so-vits-svc: initialization checkpoints already present", flush=True)


def _check_required_dependencies() -> str | None:
    failures: list[str] = []
    for module_name, package_name in _REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{package_name}: {type(exc).__name__}: {exc}")
    if failures:
        return "missing or broken So-VITS-SVC runtime dependencies: " + "; ".join(failures)
    return None


def _write_probe_wav(path: Path, *, sample_rate: int = 16000) -> None:
    frame_count = sample_rate // 20
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            value = int(32767 * 0.05 * math.sin(2 * math.pi * 440 * index / sample_rate))
            frames.extend(struct.pack("<h", value))
        handle.writeframes(bytes(frames))


def _patch_torchaudio_wav_loader() -> None:
    import soundfile as sf
    import torch
    import torchaudio

    if getattr(torchaudio.load, "_audiocover_wav_patch", False):
        return
    original_load = torchaudio.load

    def load_with_soundfile_wav(path, *args, **kwargs):
        path_string = os.fspath(path) if isinstance(path, (str, os.PathLike)) else ""
        if path_string.lower().endswith(".wav"):
            frame_offset = int(kwargs.get("frame_offset", 0) or 0)
            num_frames = int(kwargs.get("num_frames", -1) or -1)
            channels_first = bool(kwargs.get("channels_first", True))
            frames = -1 if num_frames < 0 else num_frames
            data, sample_rate = sf.read(
                path_string,
                start=frame_offset,
                frames=frames,
                dtype="float32",
                always_2d=True,
            )
            audio = torch.from_numpy(data)
            if channels_first:
                audio = audio.transpose(0, 1).contiguous()
            return audio, sample_rate
        return original_load(path, *args, **kwargs)

    load_with_soundfile_wav._audiocover_wav_patch = True  # type: ignore[attr-defined]
    torchaudio.load = load_with_soundfile_wav


def _torchaudio_decode_self_test() -> str | None:
    try:
        import torchaudio

        _patch_torchaudio_wav_loader()
        with tempfile.TemporaryDirectory(prefix="audiocover-svc-probe-") as temp_dir:
            probe = Path(temp_dir) / "probe.wav"
            _write_probe_wav(probe)
            audio, sample_rate = torchaudio.load(str(probe))
        if sample_rate != 16000 or getattr(audio, "numel", lambda: 0)() <= 0:
            return "torchaudio WAV decoder returned an invalid probe result"
    except Exception as exc:
        return f"torchaudio WAV decoder self-test failed: {type(exc).__name__}: {exc}"
    return None


def _available() -> tuple[bool, str | None]:
    dependency_error = _check_required_dependencies()
    if dependency_error:
        return False, dependency_error
    decode_error = _torchaudio_decode_self_test()
    if decode_error:
        return False, decode_error
    return True, None


def capabilities(_: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    return {
        "available": available,
        "actions": ["train", "convert"] if available else [],
        "description": "So-VITS-SVC isolated training and inference worker",
        "reason": reason,
    }


def self_test(_: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    if not available:
        raise RuntimeError(reason)
    return {"checks": ["required imports", "torchaudio WAV decoder"], "ok": True}


def train(payload: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    if not available:
        raise RuntimeError(reason)

    import json
    import shutil
    from pathlib import Path

    from so_vits_svc_fork.__main__ import pre_config, pre_hubert, pre_resample
    from so_vits_svc_fork.__main__ import train as svc_train

    _patch_contentvec_loader()
    _patch_torchaudio_wav_loader()

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

    print("so-vits-svc: preprocessing audio", flush=True)
    pre_resample.callback(
        input_dir=workdir / "dataset_raw",
        output_dir=dataset_dir,
        sampling_rate=sample_rate,
        n_jobs=1,
        top_db=30,
        frame_seconds=1,
        hop_seconds=0.3,
    )
    print("so-vits-svc: writing filelists and config", flush=True)
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
    print(f"so-vits-svc: extracting features and f0 with {f0_method}", flush=True)
    pre_hubert.callback(
        input_dir=dataset_dir,
        config_path=config_path,
        n_jobs=1,
        force_rebuild=True,
        f0_method=f0_method,
    )
    print("so-vits-svc: starting model training", flush=True)
    _copy_bundled_init_checkpoints(model_dir)
    svc_train.callback(
        config_path=config_path,
        model_path=model_dir,
        tensorboard=False,
        reset_optimizer=False,
    )
    print("so-vits-svc: locating trained checkpoint", flush=True)
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

    _patch_contentvec_loader()
    _patch_torchaudio_wav_loader()

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
    serve({"capabilities": capabilities, "self_test": self_test, "train": train, "convert": convert})


if __name__ == "__main__":
    main()
