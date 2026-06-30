from __future__ import annotations

import importlib
import logging
import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

from audiocover.workers.json_worker import serve


def _configure_backend_process() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")
    os.environ.setdefault("NO_COLOR", "1")
    os.environ.setdefault("RICH_FORCE_TERMINAL", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


_configure_backend_process()

_REQUIRED_IMPORTS: tuple[tuple[str, str], ...] = (
    ("so_vits_svc_fork", "so-vits-svc-fork"),
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("tensorboard", "tensorboard"),
    ("torch.utils.tensorboard", "torch.utils.tensorboard"),
    ("soundfile", "soundfile"),
    ("librosa", "librosa"),
    ("sklearn", "scikit-learn"),
    ("transformers", "transformers"),
    ("joblib", "joblib"),
)

_SO_VITS_SVC_SAMPLE_RATE = 44100
_SO_VITS_SVC_F0_METHODS = {"crepe", "crepe-tiny", "parselmouth", "dio", "harvest"}
_SO_VITS_SVC_CHECKPOINT_INTERVAL_EPOCHS = 10


def _count_filelist_items(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _configure_so_vits_checkpoint_retention(train_cfg: dict[str, Any], filelist_dir: Path) -> None:
    epochs = max(1, int(train_cfg.get("epochs") or 200))
    batch_size = max(1, int(train_cfg.get("batch_size") or 8))
    train_items = _count_filelist_items(filelist_dir / "train.txt")
    batches_per_epoch = max(1, math.ceil(train_items / batch_size))

    train_cfg["eval_interval"] = batches_per_epoch * _SO_VITS_SVC_CHECKPOINT_INTERVAL_EPOCHS
    expected_checkpoints = math.ceil(epochs / _SO_VITS_SVC_CHECKPOINT_INTERVAL_EPOCHS) + 2
    train_cfg["keep_ckpts"] = max(expected_checkpoints, int(train_cfg.get("keep_ckpts") or 0))
    train_cfg["ckpt_name_by_step"] = False


def _resolve_so_vits_sample_rate(requested: int | str | None) -> tuple[int, str | None]:
    try:
        requested_rate = int(requested) if requested is not None else _SO_VITS_SVC_SAMPLE_RATE
    except (TypeError, ValueError):
        requested_rate = _SO_VITS_SVC_SAMPLE_RATE
    if requested_rate == _SO_VITS_SVC_SAMPLE_RATE:
        return _SO_VITS_SVC_SAMPLE_RATE, None
    return (
        _SO_VITS_SVC_SAMPLE_RATE,
        f"using the 44k preset; resampling training audio to {_SO_VITS_SVC_SAMPLE_RATE} Hz instead of {requested_rate} Hz",
    )


def _resolve_so_vits_f0_method(requested: object) -> tuple[str, str | None]:
    value = str(requested or "").strip().lower()
    if value in _SO_VITS_SVC_F0_METHODS:
        return value, None
    if value in {"rmvpe", "rmvpe+", "fcpe"}:
        return "harvest", f"so-vits-svc-fork does not support {value}; using harvest f0 extraction"
    return "dio", f"unsupported so-vits-svc f0 method {value or '<empty>'}; using dio"


def _asset_dir(name: str, required_files: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    env_var = f"AUDIOCOVER_{name.upper().replace('-', '_')}_DIR"
    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    common_assets = os.environ.get("AUDIOCOVER_ASSETS_DIR")
    if common_assets:
        candidates.append(Path(common_assets).expanduser() / name)

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
    candidates.append(source_path.parent / "assets" / name)
    if len(source_path.parents) > 3:
        source_root = source_path.parents[3]
        candidates.append(source_root / "build" / "audiocover-bundle-assets" / name)
        candidates.append(source_root / "backend-runtimes" / "so-vits-svc" / "assets" / name)

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
            "So-VITS-SVC ContentVec assets are missing. Run "
            "python scripts/build_desktop.py --prepare-assets-only for source use, "
            "set AUDIOCOVER_ASSETS_DIR to a prepared asset directory, or set "
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
            "So-VITS-SVC initialization checkpoints are missing. Run "
            "python scripts/build_desktop.py --prepare-assets-only for source use, "
            "set AUDIOCOVER_ASSETS_DIR to a prepared asset directory, or set "
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


def _install_pretrained_model_compat(
    model_dir: Path,
    svc_utils: Any,
    svc_train_module: Any | None = None,
) -> None:
    original = getattr(svc_utils, "ensure_pretrained_model", None)
    if original is None or getattr(original, "_audiocover_patch", False):
        return

    def ensure_pretrained_model_compat(*args, **kwargs):
        checkpoint_dir = model_dir
        for value in (*args, kwargs.get("model_dir"), kwargs.get("model_path"), kwargs.get("path")):
            if isinstance(value, (str, os.PathLike)):
                checkpoint_dir = Path(value)
                break
        if (checkpoint_dir / "G_0.pth").is_file() and (checkpoint_dir / "D_0.pth").is_file():
            print(
                "so-vits-svc: using local initialization checkpoints; skipping pretrained lookup",
                flush=True,
            )
            return None
        try:
            return original(*args, **kwargs)
        except TypeError as exc:
            if "unhashable type" not in str(exc):
                raise
            raise RuntimeError(
                "So-VITS-SVC pretrained checkpoint lookup failed before local initialization "
                "checkpoints were available. This usually means the packaged "
                "so-vits-svc-init assets are missing or were not copied into the model log "
                "directory."
            ) from exc

    ensure_pretrained_model_compat._audiocover_patch = True  # type: ignore[attr-defined]
    svc_utils.ensure_pretrained_model = ensure_pretrained_model_compat
    if svc_train_module is not None and hasattr(svc_train_module, "ensure_pretrained_model"):
        svc_train_module.ensure_pretrained_model = ensure_pretrained_model_compat


def _patch_so_vits_pretrained_model_lookup(model_dir: Path) -> None:
    from so_vits_svc_fork import train as svc_train_module
    from so_vits_svc_fork import utils as svc_utils

    _install_pretrained_model_compat(model_dir, svc_utils, svc_train_module)


def _format_torch_device_report() -> str:
    try:
        import torch
    except Exception as exc:
        return f"torch import failed: {type(exc).__name__}: {exc}"

    parts = [f"torch={getattr(torch, '__version__', 'unknown')}"]
    version = getattr(torch, "version", None)
    parts.append(f"cuda_build={getattr(version, 'cuda', None) or 'none'}")
    try:
        available = torch.cuda.is_available()
    except Exception as exc:
        return "; ".join(parts + [f"cuda_available_check_failed={type(exc).__name__}: {exc}"])
    parts.append(f"cuda_available={available}")
    if available:
        try:
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(index) for index in range(count)]
            parts.append(f"cuda_devices={count} ({', '.join(names)})")
        except Exception as exc:
            parts.append(f"cuda_device_query_failed={type(exc).__name__}: {exc}")
    return "; ".join(parts)


def _resolve_torch_device(preferred: str | None) -> tuple[str, str | None]:
    requested = (preferred or "auto").strip().lower()
    if os.environ.get("AUDIOCOVER_BINARY_CPU_ONLY", "").lower() in {"1", "true", "yes"}:
        if requested in {"gpu", "cuda"} or requested.startswith("cuda:"):
            return "cpu", "AudioCover release binaries are CPU-only; run from source with a CUDA/MPS PyTorch build for GPU execution"
        return "cpu", None
    if requested in {"", "auto"}:
        candidates = ["cuda", "cpu"]
    elif requested in {"gpu", "cuda"} or requested.startswith("cuda:"):
        candidates = [requested if requested.startswith("cuda:") else "cuda", "cpu"]
    else:
        candidates = [requested, "cpu"] if requested != "cpu" else ["cpu"]

    try:
        import torch
    except Exception as exc:
        return "cpu", f"torch import failed: {type(exc).__name__}: {exc}"

    failures: list[str] = []
    for candidate in candidates:
        if candidate.startswith("cuda"):
            cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
            if not cuda_build:
                failures.append("CUDA PyTorch build is not installed")
                continue
            try:
                if not torch.cuda.is_available():
                    failures.append("torch.cuda.is_available() is false")
                    continue
                if torch.cuda.device_count() <= 0:
                    failures.append("torch.cuda.device_count() is 0")
                    continue
                device = torch.device(candidate)
                probe = torch.empty((1,), device=device)
                probe += 1
                matmul_probe = torch.eye(8, device=device) @ torch.eye(8, device=device)
                _ = float(matmul_probe[0, 0].detach().cpu())
                torch.cuda.synchronize(device)
                return str(device), None
            except Exception as exc:
                failures.append(f"{candidate} initialization failed: {type(exc).__name__}: {exc}")
                continue
        if candidate == "cpu":
            return "cpu", "; ".join(failures) if failures else None
        try:
            device = torch.device(candidate)
            _ = torch.empty((1,), device=device)
            return str(device), None
        except Exception as exc:
            failures.append(f"{candidate} initialization failed: {type(exc).__name__}: {exc}")
    return "cpu", "; ".join(failures) if failures else "no usable training device found"


def _should_fail_training_device_fallback(preferred: str | None, fallback_reason: str | None) -> bool:
    if not fallback_reason:
        return False
    requested = (preferred or "auto").strip().lower()
    if requested in {"gpu", "cuda"} or requested.startswith("cuda:"):
        return True
    return requested in {"", "auto"} and "initialization failed" in fallback_reason


def _patch_so_vits_device_selection(device: str) -> None:
    import torch
    from so_vits_svc_fork import utils as svc_utils

    selected = torch.device(device)

    def get_selected_device(index: int = 0) -> torch.device:
        if selected.type == "cuda" and selected.index is None and torch.cuda.device_count() > 0:
            return torch.device(f"cuda:{index % torch.cuda.device_count()}")
        return selected

    svc_utils.get_optimal_device = get_selected_device


def _install_lightning_trainer_device_defaults(device: str):
    import lightning.pytorch as pl

    original = getattr(pl.Trainer, "_audiocover_original", pl.Trainer)

    class AudioCoverProgressCallback(pl.Callback):
        _audiocover_progress = True

        def on_train_epoch_start(self, trainer, pl_module) -> None:
            total = getattr(trainer, "max_epochs", None)
            current = int(getattr(trainer, "current_epoch", 0)) + 1
            if isinstance(total, int) and total > 0:
                print(f"so-vits-svc: training epoch {current}/{total} started", flush=True)
            else:
                print(f"so-vits-svc: training epoch {current} started", flush=True)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
            total = getattr(trainer, "num_training_batches", None)
            if not isinstance(total, int) or total <= 0:
                return
            interval = max(1, total // 10)
            current = batch_idx + 1
            if current != total and current % interval != 0:
                return
            epoch = int(getattr(trainer, "current_epoch", 0)) + 1
            max_epochs = getattr(trainer, "max_epochs", None)
            prefix = f"so-vits-svc: epoch {epoch}/{max_epochs}" if isinstance(max_epochs, int) else f"so-vits-svc: epoch {epoch}"
            metrics = []
            for key, value in sorted(getattr(trainer, "progress_bar_metrics", {}).items()):
                try:
                    number = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
                except Exception:
                    continue
                metrics.append(f"{key}={number:.4g}")
                if len(metrics) >= 4:
                    break
            suffix = f"; {', '.join(metrics)}" if metrics else ""
            print(f"{prefix} batch {current}/{total}{suffix}", flush=True)

        def on_train_epoch_end(self, trainer, pl_module) -> None:
            epoch = int(getattr(trainer, "current_epoch", 0)) + 1
            print(f"so-vits-svc: training epoch {epoch} finished", flush=True)

    def trainer_with_device_defaults(*args, **kwargs):
        if device.startswith("cuda"):
            kwargs["accelerator"] = "gpu"
            if ":" in device:
                kwargs["devices"] = [int(device.rsplit(":", 1)[1])]
            else:
                kwargs["devices"] = 1
        elif device == "cpu":
            kwargs["accelerator"] = "cpu"
            kwargs["devices"] = 1
        elif device == "mps":
            kwargs["accelerator"] = "mps"
            kwargs["devices"] = 1
        callbacks = list(kwargs.get("callbacks") or [])
        callbacks = [callback for callback in callbacks if callback.__class__.__name__ != "RichProgressBar"]
        if not any(getattr(callback, "_audiocover_progress", False) for callback in callbacks):
            callbacks.append(AudioCoverProgressCallback())
        kwargs["callbacks"] = callbacks
        return original(*args, **kwargs)

    trainer_with_device_defaults._audiocover_original = original  # type: ignore[attr-defined]
    pl.Trainer = trainer_with_device_defaults
    return original


def _restore_lightning_trainer(original) -> None:
    if original is None:
        return
    try:
        import lightning.pytorch as pl
    except Exception:
        return
    pl.Trainer = original


def _patch_so_vits_numpy_plotting() -> None:
    from so_vits_svc_fork import utils as svc_utils

    if getattr(svc_utils.plot_spectrogram_to_numpy, "_audiocover_patch", False):
        return

    def _canvas_argb_to_numpy(fig):
        import numpy as np

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        return data.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    def plot_spectrogram_to_numpy(spectrogram):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pylab as plt

        fig, ax = plt.subplots(figsize=(10, 2))
        try:
            im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
            plt.colorbar(im, ax=ax)
            plt.xlabel("Frames")
            plt.ylabel("Channels")
            plt.tight_layout()
            return _canvas_argb_to_numpy(fig)
        finally:
            plt.close(fig)

    def plot_data_to_numpy(x, y):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pylab as plt

        fig, _ax = plt.subplots(figsize=(10, 2))
        try:
            plt.plot(x)
            plt.plot(y)
            plt.tight_layout()
            return _canvas_argb_to_numpy(fig)
        finally:
            plt.close(fig)

    plot_spectrogram_to_numpy._audiocover_patch = True  # type: ignore[attr-defined]
    plot_data_to_numpy._audiocover_patch = True  # type: ignore[attr-defined]
    svc_utils.plot_spectrogram_to_numpy = plot_spectrogram_to_numpy
    svc_utils.plot_data_to_numpy = plot_data_to_numpy


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


def _hubert_import_self_test() -> str | None:
    try:
        from transformers import HubertModel
    except Exception as exc:
        return f"transformers HuBERT import self-test failed: {type(exc).__name__}: {exc}"
    if HubertModel is None:
        return "transformers HuBERT import self-test failed: HubertModel is unavailable"
    return None


def _asset_self_test() -> str | None:
    if _runtime_downloads_allowed():
        return None
    missing: list[str] = []
    if _contentvec_dir() is None:
        missing.append("content-vec-best/config.json and pytorch_model.bin")
    if _asset_dir("so-vits-svc-init", ("D_0.pth", "G_0.pth")) is None:
        missing.append("so-vits-svc-init/D_0.pth and G_0.pth")
    if missing:
        return "missing bundled So-VITS-SVC runtime assets: " + "; ".join(missing)
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


def _train_entrypoint_self_test() -> str | None:
    try:
        from so_vits_svc_fork import train as svc_train_module
        from torch.utils.tensorboard.writer import SummaryWriter
    except Exception as exc:
        return f"So-VITS-SVC training entrypoint self-test failed: {type(exc).__name__}: {exc}"
    if not hasattr(svc_train_module, "train"):
        return "So-VITS-SVC training entrypoint self-test failed: train command is unavailable"
    if SummaryWriter is None:
        return "So-VITS-SVC training entrypoint self-test failed: SummaryWriter is unavailable"
    return None


def _patch_lightning_rich_summary() -> None:
    try:
        from lightning.pytorch.callbacks.rich_model_summary import RichModelSummary
    except Exception:
        return

    if getattr(RichModelSummary.on_fit_start, "_audiocover_patch", False):
        return

    def on_fit_start_noop(self, trainer, pl_module) -> None:
        return None

    on_fit_start_noop._audiocover_patch = True  # type: ignore[attr-defined]
    RichModelSummary.on_fit_start = on_fit_start_noop


def _available() -> tuple[bool, str | None]:
    dependency_error = _check_required_dependencies()
    if dependency_error:
        return False, dependency_error
    hubert_error = _hubert_import_self_test()
    if hubert_error:
        return False, hubert_error
    decode_error = _torchaudio_decode_self_test()
    if decode_error:
        return False, decode_error
    train_error = _train_entrypoint_self_test()
    if train_error:
        return False, train_error
    asset_error = _asset_self_test()
    if asset_error:
        return False, asset_error
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
    return {
        "checks": [
            "required imports",
            "transformers HuBERT import",
            "torchaudio WAV decoder",
            "training entrypoint and tensorboard writer",
            "bundled assets",
        ],
        "ok": True,
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

    _patch_contentvec_loader()
    _patch_torchaudio_wav_loader()
    _patch_so_vits_numpy_plotting()
    print(f"so-vits-svc: {_format_torch_device_report()}", flush=True)
    requested_device = str(payload.get("device") or "auto")
    training_device, fallback_reason = _resolve_torch_device(requested_device)
    if fallback_reason and training_device == "cpu":
        if _should_fail_training_device_fallback(requested_device, fallback_reason):
            print(f"so-vits-svc: CUDA unusable for training ({fallback_reason}); refusing CPU fallback", flush=True)
            raise RuntimeError(
                "So-VITS-SVC training found a CUDA device but could not run CUDA kernels. "
                f"Install a runtime pack built with a compatible CUDA PyTorch wheel. Details: {fallback_reason}"
            )
        print(f"so-vits-svc: CUDA unavailable for training ({fallback_reason}); falling back to CPU", flush=True)
    print(f"so-vits-svc: selected training device: {training_device}", flush=True)
    _patch_so_vits_device_selection(training_device)

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
    sample_rate, sample_rate_note = _resolve_so_vits_sample_rate(payload.get("sample_rate"))
    if sample_rate_note:
        print(f"so-vits-svc: {sample_rate_note}", flush=True)
    f0_method, f0_note = _resolve_so_vits_f0_method(payload.get("f0_method"))
    if f0_note:
        print(f"so-vits-svc: {f0_note}", flush=True)

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
        _configure_so_vits_checkpoint_retention(train_cfg, filelist_dir)
        print(
            "so-vits-svc: checkpointing every "
            f"{_SO_VITS_SVC_CHECKPOINT_INTERVAL_EPOCHS} epoch(s); "
            f"keeping up to {train_cfg['keep_ckpts']} checkpoint(s)",
            flush=True,
        )
        data_cfg = data.setdefault("data", {})
        data_cfg["sampling_rate"] = sample_rate
        model_cfg = data.setdefault("model", {})
        model_cfg.pop("pretrained", None)
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
    _patch_so_vits_pretrained_model_lookup(model_dir)
    _patch_lightning_rich_summary()
    logging.getLogger("lightning.pytorch.utilities.rank_zero").disabled = True
    original_trainer = _install_lightning_trainer_device_defaults(training_device)
    try:
        svc_train.callback(
            config_path=config_path,
            model_path=model_dir,
            tensorboard=False,
            reset_optimizer=False,
        )
    finally:
        _restore_lightning_trainer(original_trainer)
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
    f0_method, f0_note = _resolve_so_vits_f0_method(payload.get("f0_method"))
    if f0_note:
        print(f"so-vits-svc: {f0_note}", flush=True)
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
        noise_scale=float(payload.get("noise_scale") or 0.12),
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
