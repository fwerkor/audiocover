from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from audiocover.config import TrainingConfig
from audiocover.runtime import BackendRuntimeError, BackendRuntimeManager
from audiocover.training import train_model


def _write_tone(path: Path, freq: float, seconds: float = 2.2, sr: int = 48000) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    x = (0.08 * np.sin(2 * np.pi * freq * t))[:, None].astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, sr)


def _write_fake_runtime(runtime_root: Path, body: str) -> None:
    runtime = runtime_root / "simple-timbre"
    runtime.mkdir(parents=True)
    if os.name == "nt":
        script = runtime / "worker.py"
        script.write_text(body, encoding="utf-8")
        worker = runtime / "simple-timbre.cmd"
        worker.write_text('@echo off\r\npython "%~dp0worker.py"\r\n', encoding="utf-8")
    else:
        worker = runtime / "simple-timbre"
        worker.write_text(f"#!/usr/bin/env python3\n{body}\n", encoding="utf-8")
        worker.chmod(0o755)


def test_runtime_manager_source_worker_capabilities() -> None:
    manager = BackendRuntimeManager(runtime_roots=[])
    cap = manager.capabilities("simple-timbre")
    assert cap.available
    assert cap.supports("train")
    assert cap.supports("convert")


def test_managed_training_uses_runtime_worker(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_tone(raw / "a.wav", 220)
    _write_tone(raw / "b.wav", 330)

    model_dir = tmp_path / "model"
    logs: list[str] = []
    package = train_model(
        raw,
        model_dir,
        display_name="runtime-test",
        config=TrainingConfig(backend="managed", segment_seconds=2.0),
        consent=True,
        log=logs.append,
    )

    assert package.runtime_backend == "simple-timbre"
    assert package.conversion.backend == "managed"
    assert package.conversion.runtime_backend == "simple-timbre"
    assert package.simple_profile_path is not None
    assert package.simple_profile_path.exists()
    assert any("dataset preparation complete" in line for line in logs)
    assert any("selected backend" in line for line in logs)
    assert any("starting simple-timbre runtime action: train" in line for line in logs)


def test_frozen_worker_protocol_when_runtime_dir_is_supplied(tmp_path: Path) -> None:
    runtime_root = tmp_path / "backend-runtimes"
    _write_fake_runtime(
        runtime_root,
        "import json, sys\n"
        "req=json.loads(sys.stdin.read())\n"
        "print(json.dumps({'id': req['id'], 'ok': True, 'result': {'available': True, 'actions': ['train'], 'description': 'fake'}}))\n",
    )

    manager = BackendRuntimeManager(runtime_roots=[runtime_root])
    cap = manager.capabilities("simple-timbre")
    assert cap.available
    assert cap.supports("train")


def test_runtime_manager_streams_worker_logs(tmp_path: Path) -> None:
    runtime_root = tmp_path / "backend-runtimes"
    _write_fake_runtime(
        runtime_root,
        "import json, sys\n"
        "req=json.loads(sys.stdin.read())\n"
        "print('step 1', flush=True)\n"
        "print('warn 1', file=sys.stderr, flush=True)\n"
        "print(json.dumps({'id': req['id'], 'ok': True, 'result': {'value': 7}}), flush=True)\n",
    )

    logs: list[str] = []
    manager = BackendRuntimeManager(runtime_roots=[runtime_root])
    result = manager.invoke("simple-timbre", "train", {}, log=logs.append)

    assert result == {"value": 7}
    assert any("simple-timbre stdout: step 1" in line for line in logs)
    assert any("simple-timbre stderr: warn 1" in line for line in logs)
    assert not any("'ok': True" in line or '"ok": true' in line for line in logs)


def test_preferred_runtime_must_be_available() -> None:
    manager = BackendRuntimeManager(runtime_roots=[])
    try:
        manager.require_training_backend("rvc")
    except BackendRuntimeError as exc:
        assert "preferred backend runtime is not available" in str(exc)
    else:
        raise AssertionError("expected unavailable preferred runtime to fail")


def test_so_vits_worker_prefers_env_contentvec_asset_dir(tmp_path: Path, monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    asset_dir = tmp_path / "content-vec-best"
    asset_dir.mkdir()
    (asset_dir / "config.json").write_text("{}", encoding="utf-8")
    (asset_dir / "pytorch_model.bin").write_bytes(b"weights")

    monkeypatch.setenv("AUDIOCOVER_CONTENT_VEC_BEST_DIR", str(asset_dir))

    assert so_vits_svc_worker._contentvec_dir() == asset_dir


def test_so_vits_worker_asset_dir_handles_shallow_frozen_file_path(monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    monkeypatch.setattr(so_vits_svc_worker, "__file__", "/tmp/worker.py")

    assert so_vits_svc_worker._asset_dir("missing-assets", ("model.bin",)) is None


def test_so_vits_worker_copies_bundled_init_checkpoints(tmp_path: Path, monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    asset_dir = tmp_path / "so-vits-svc-init"
    asset_dir.mkdir()
    (asset_dir / "D_0.pth").write_bytes(b"d")
    (asset_dir / "G_0.pth").write_bytes(b"g")
    model_dir = tmp_path / "model"

    monkeypatch.setenv("AUDIOCOVER_SO_VITS_SVC_INIT_DIR", str(asset_dir))
    so_vits_svc_worker._copy_bundled_init_checkpoints(model_dir)

    assert (model_dir / "D_0.pth").read_bytes() == b"d"
    assert (model_dir / "G_0.pth").read_bytes() == b"g"


def test_so_vits_worker_skips_pretrained_lookup_when_local_checkpoints_exist(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from audiocover.workers import so_vits_svc_worker

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "D_0.pth").write_bytes(b"d")
    (model_dir / "G_0.pth").write_bytes(b"g")
    calls = []

    def broken_lookup(*args, **kwargs):
        calls.append((args, kwargs))
        raise TypeError("unhashable type: 'dict'")

    svc_utils = SimpleNamespace(ensure_pretrained_model=broken_lookup)
    svc_train_module = SimpleNamespace(ensure_pretrained_model=broken_lookup)

    so_vits_svc_worker._install_pretrained_model_compat(model_dir, svc_utils, svc_train_module)
    svc_train_module.ensure_pretrained_model({"nested": "candidate"})

    assert calls == []


def test_so_vits_worker_reports_pretrained_lookup_failure_without_local_checkpoints(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from audiocover.workers import so_vits_svc_worker

    def broken_lookup(*args, **kwargs):
        raise TypeError("unhashable type: 'dict'")

    svc_utils = SimpleNamespace(ensure_pretrained_model=broken_lookup)
    so_vits_svc_worker._install_pretrained_model_compat(tmp_path / "model", svc_utils)

    try:
        svc_utils.ensure_pretrained_model(tmp_path / "model", {"nested": "candidate"})
    except RuntimeError as exc:
        assert "pretrained checkpoint lookup failed" in str(exc)
    else:
        raise AssertionError("expected pretrained lookup compatibility error")


def test_runtime_manager_ignores_non_object_json_logs(tmp_path: Path) -> None:
    runtime_root = tmp_path / "backend-runtimes"
    _write_fake_runtime(
        runtime_root,
        "import json, sys\n"
        "req=json.loads(sys.stdin.read())\n"
        "print(json.dumps({'id': req['id'], 'ok': True, 'result': {'value': 9}}), flush=True)\n"
        "print(json.dumps('HTTP/1.1 206 Partial Content'), flush=True)\n",
    )

    manager = BackendRuntimeManager(runtime_roots=[runtime_root])
    assert manager.invoke("simple-timbre", "train", {}) == {"value": 9}



def test_runtime_manager_finds_json_response_embedded_after_logs(tmp_path: Path) -> None:
    runtime_root = tmp_path / "backend-runtimes"
    _write_fake_runtime(
        runtime_root,
        "import json, sys\n"
        "req=json.loads(sys.stdin.read())\n"
        "print('INFO prefix without protocol newline ' + json.dumps({'id': req['id'], 'ok': True, 'result': {'value': 11}}), flush=True)\n",
    )

    manager = BackendRuntimeManager(runtime_roots=[runtime_root])
    assert manager.invoke("simple-timbre", "train", {}) == {"value": 11}


def test_runtime_manager_coalesces_tqdm_progress_lines(tmp_path: Path) -> None:
    runtime_root = tmp_path / "backend-runtimes"
    _write_fake_runtime(
        runtime_root,
        "import json, sys\n"
        "req=json.loads(sys.stdin.read())\n"
        "print(' 91%|█████████ | 173/191 [01:57<00:12,  1.45it/s]', file=sys.stderr, flush=True)\n"
        "print(json.dumps({'id': req['id'], 'ok': True, 'result': {'value': 12}}), flush=True)\n",
    )

    logs: list[str] = []
    manager = BackendRuntimeManager(runtime_roots=[runtime_root])

    assert manager.invoke("simple-timbre", "train", {}, log=logs.append) == {"value": 12}
    assert any(line.startswith("\rsimple-timbre progress: 91% (173/191)") for line in logs)
    assert not any("████" in line for line in logs)


def test_runtime_manager_handles_carriage_return_progress_without_newlines(tmp_path: Path) -> None:
    runtime_root = tmp_path / "backend-runtimes"
    _write_fake_runtime(
        runtime_root,
        "import json, sys\n"
        "req=json.loads(sys.stdin.read())\n"
        "sys.stderr.write('  0%|          | 0/191 [00:00<?, ?it/s]')\n"
        "sys.stderr.flush()\n"
        "sys.stderr.write('\\r 50%|█████     | 95/191 [01:00<01:00,  1.50it/s]')\n"
        "sys.stderr.flush()\n"
        "sys.stderr.write('\\r100%|██████████| 191/191 [02:00<00:00,  1.59it/s]')\n"
        "sys.stderr.flush()\n"
        "print(json.dumps({'id': req['id'], 'ok': True, 'result': {'value': 13}}), flush=True)\n",
    )

    logs: list[str] = []
    manager = BackendRuntimeManager(runtime_roots=[runtime_root])

    assert manager.invoke("simple-timbre", "train", {}, log=logs.append) == {"value": 13}
    assert any(line.startswith("\rsimple-timbre progress: 0% (0/191)") for line in logs)
    assert any(line.startswith("\rsimple-timbre progress: 50% (95/191)") for line in logs)
    assert any(line.startswith("\rsimple-timbre progress: 100% (191/191)") for line in logs)

def test_runtime_manager_reports_missing_worker_response(tmp_path: Path) -> None:
    runtime_root = tmp_path / "backend-runtimes"
    _write_fake_runtime(
        runtime_root,
        "import json, sys\n"
        "json.loads(sys.stdin.read())\n"
        "print(json.dumps('HTTP/1.1 206 Partial Content'), flush=True)\n",
    )

    manager = BackendRuntimeManager(runtime_roots=[runtime_root])
    try:
        manager.invoke("simple-timbre", "train", {})
    except BackendRuntimeError as exc:
        assert "valid JSON response" in str(exc)
    else:
        raise AssertionError("expected missing worker response to fail")


def test_so_vits_worker_missing_contentvec_fails_closed(monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    monkeypatch.delenv("AUDIOCOVER_CONTENT_VEC_BEST_DIR", raising=False)
    monkeypatch.delenv("AUDIOCOVER_ALLOW_RUNTIME_DOWNLOADS", raising=False)
    monkeypatch.setattr(so_vits_svc_worker, "_contentvec_dir", lambda: None)

    try:
        so_vits_svc_worker._patch_contentvec_loader()
    except RuntimeError as exc:
        assert "ContentVec assets are missing" in str(exc)
    else:
        raise AssertionError("expected missing ContentVec assets to fail")


def test_so_vits_worker_missing_init_checkpoints_fails_closed(tmp_path: Path, monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    monkeypatch.delenv("AUDIOCOVER_SO_VITS_SVC_INIT_DIR", raising=False)
    monkeypatch.delenv("AUDIOCOVER_ALLOW_RUNTIME_DOWNLOADS", raising=False)
    monkeypatch.setattr(so_vits_svc_worker, "_asset_dir", lambda name, required_files: None)

    try:
        so_vits_svc_worker._copy_bundled_init_checkpoints(tmp_path / "model")
    except RuntimeError as exc:
        assert "initialization checkpoints are missing" in str(exc)
    else:
        raise AssertionError("expected missing initialization checkpoints to fail")


def test_so_vits_worker_reports_missing_audio_dependency(monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    def fake_import_module(module_name: str):
        if module_name == "torchaudio":
            raise ModuleNotFoundError("No module named 'torchaudio'")
        return object()

    monkeypatch.setattr(so_vits_svc_worker.importlib, "import_module", fake_import_module)

    reason = so_vits_svc_worker._check_required_dependencies()

    assert reason is not None
    assert "torchaudio" in reason


def test_so_vits_worker_availability_runs_decoder_self_test(monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    monkeypatch.setattr(so_vits_svc_worker, "_check_required_dependencies", lambda: None)
    monkeypatch.setattr(so_vits_svc_worker, "_hubert_import_self_test", lambda: None)
    monkeypatch.setattr(
        so_vits_svc_worker,
        "_torchaudio_decode_self_test",
        lambda: "torchaudio WAV decoder self-test failed: probe",
    )

    available, reason = so_vits_svc_worker._available()

    assert not available
    assert reason is not None
    assert "decoder self-test failed" in reason


def test_so_vits_worker_availability_runs_hubert_self_test(monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    monkeypatch.setattr(so_vits_svc_worker, "_check_required_dependencies", lambda: None)
    monkeypatch.setattr(
        so_vits_svc_worker,
        "_hubert_import_self_test",
        lambda: "transformers HuBERT import self-test failed: probe",
    )
    monkeypatch.setattr(so_vits_svc_worker, "_torchaudio_decode_self_test", lambda: None)

    available, reason = so_vits_svc_worker._available()

    assert not available
    assert reason is not None
    assert "HuBERT import self-test failed" in reason


def test_so_vits_worker_availability_runs_training_entrypoint_self_test(monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    monkeypatch.setattr(so_vits_svc_worker, "_check_required_dependencies", lambda: None)
    monkeypatch.setattr(so_vits_svc_worker, "_hubert_import_self_test", lambda: None)
    monkeypatch.setattr(so_vits_svc_worker, "_torchaudio_decode_self_test", lambda: None)
    monkeypatch.setattr(
        so_vits_svc_worker,
        "_train_entrypoint_self_test",
        lambda: "So-VITS-SVC training entrypoint self-test failed: probe",
    )

    available, reason = so_vits_svc_worker._available()

    assert not available
    assert reason is not None
    assert "training entrypoint self-test failed" in reason


class _FakeTensor:
    def __iadd__(self, _value):
        return self


class _FakeTorchDevice:
    def __init__(self, value: str):
        self.value = value
        self.type = value.split(":", 1)[0]
        self.index = int(value.split(":", 1)[1]) if ":" in value else None

    def __str__(self) -> str:
        return self.value


def test_so_vits_worker_cuda_first_device_falls_back_to_cpu_without_cuda_build(monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            synchronize=lambda _device=None: None,
        ),
        device=_FakeTorchDevice,
        empty=lambda _shape, device=None: _FakeTensor(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    device, reason = so_vits_svc_worker._resolve_torch_device("auto")

    assert device == "cpu"
    assert reason is not None
    assert "CUDA PyTorch build is not installed" in reason


def test_so_vits_worker_cuda_first_device_uses_cuda_when_probe_succeeds(monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.1"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda _index: "Fake CUDA GPU",
            synchronize=lambda _device=None: None,
        ),
        device=_FakeTorchDevice,
        empty=lambda _shape, device=None: _FakeTensor(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    device, reason = so_vits_svc_worker._resolve_torch_device("auto")

    assert device == "cuda"
    assert reason is None


def test_so_vits_worker_patches_numpy_binary_plotting(monkeypatch) -> None:
    import numpy as np
    import pytest

    pytest.importorskip("matplotlib")
    from audiocover.workers import so_vits_svc_worker

    svc_utils = SimpleNamespace(
        plot_spectrogram_to_numpy=lambda _spectrogram: None,
        plot_data_to_numpy=lambda _x, _y: None,
    )
    monkeypatch.setitem(sys.modules, "so_vits_svc_fork", SimpleNamespace(utils=svc_utils))
    monkeypatch.setitem(sys.modules, "so_vits_svc_fork.utils", svc_utils)

    so_vits_svc_worker._patch_so_vits_numpy_plotting()

    spectrogram = svc_utils.plot_spectrogram_to_numpy(np.zeros((2, 3), dtype=np.float32))
    line_plot = svc_utils.plot_data_to_numpy(np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))

    assert spectrogram.shape[-1] == 4
    assert line_plot.shape[-1] == 4
