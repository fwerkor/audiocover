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
