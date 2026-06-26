from __future__ import annotations

import os
from pathlib import Path

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


def test_so_vits_worker_reports_missing_decoder_dependency(monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    def fake_import_module(module_name: str):
        if module_name == "torchcodec":
            raise ModuleNotFoundError("No module named 'torchcodec'")
        return object()

    monkeypatch.setattr(so_vits_svc_worker.importlib, "import_module", fake_import_module)

    reason = so_vits_svc_worker._check_required_dependencies()

    assert reason is not None
    assert "torchcodec" in reason


def test_so_vits_worker_availability_runs_decoder_self_test(monkeypatch) -> None:
    from audiocover.workers import so_vits_svc_worker

    monkeypatch.setattr(so_vits_svc_worker, "_check_required_dependencies", lambda: None)
    monkeypatch.setattr(
        so_vits_svc_worker,
        "_torchaudio_decode_self_test",
        lambda: "torchaudio WAV decoder self-test failed: probe",
    )

    available, reason = so_vits_svc_worker._available()

    assert not available
    assert reason is not None
    assert "decoder self-test failed" in reason
