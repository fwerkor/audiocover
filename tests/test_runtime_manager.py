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
    package = train_model(
        raw,
        model_dir,
        display_name="runtime-test",
        config=TrainingConfig(backend="managed", segment_seconds=2.0),
        consent=True,
    )

    assert package.runtime_backend == "simple-timbre"
    assert package.conversion.backend == "managed"
    assert package.conversion.runtime_backend == "simple-timbre"
    assert package.simple_profile_path is not None
    assert package.simple_profile_path.exists()


def test_frozen_worker_protocol_when_runtime_dir_is_supplied(tmp_path: Path) -> None:
    runtime = tmp_path / "backend-runtimes" / "simple-timbre"
    runtime.mkdir(parents=True)
    if os.name == "nt":
        worker = runtime / "simple-timbre.cmd"
        worker.write_text(
            "@echo off\r\n"
            "python -c \"import json,sys; req=json.load(sys.stdin); print(json.dumps({'id': req['id'], 'ok': True, 'result': {'available': True, 'actions': ['train'], 'description': 'fake'}}))\"\r\n",
            encoding="utf-8",
        )
    else:
        worker = runtime / "simple-timbre"
        worker.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "req=json.loads(sys.stdin.read())\n"
            "print(json.dumps({'id': req['id'], 'ok': True, 'result': {'available': True, 'actions': ['train'], 'description': 'fake'}}))\n",
            encoding="utf-8",
        )
        worker.chmod(0o755)


    manager = BackendRuntimeManager(runtime_roots=[tmp_path / "backend-runtimes"])
    cap = manager.capabilities("simple-timbre")
    assert cap.available
    assert cap.supports("train")


def test_preferred_runtime_must_be_available() -> None:
    manager = BackendRuntimeManager(runtime_roots=[])
    try:
        manager.require_training_backend("rvc")
    except BackendRuntimeError as exc:
        assert "preferred backend runtime is not available" in str(exc)
    else:
        raise AssertionError("expected unavailable preferred runtime to fail")
