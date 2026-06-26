from __future__ import annotations

import importlib.util
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def _build_desktop_module():
    spec = importlib.util.spec_from_file_location("audiocover_build_desktop", ROOT / "scripts" / "build_desktop.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_so_vits_backend_extra_declares_decoder_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = project["project"]["optional-dependencies"]["so-vits-svc-backend"]
    normalized = "\n".join(deps)

    for dependency in (
        "so-vits-svc-fork==4.2.30",
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "torchcodec>=0.8.0",
        "scikit-learn>=1.4.0",
    ):
        assert dependency in normalized


def test_release_matrix_installs_so_vits_decoder_dependencies() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    for dependency in (
        "so-vits-svc-fork==4.2.30",
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "torchcodec>=0.8.0",
        "scikit-learn>=1.4.0",
    ):
        assert workflow.count(dependency) >= 4


def test_build_script_collects_and_self_tests_so_vits_runtime() -> None:
    build_desktop = _build_desktop_module()

    assert "torchcodec" in build_desktop.WORKER_COLLECTS["so-vits-svc"]
    assert build_desktop.RUNTIME_SELF_TESTS["so-vits-svc"] == "self_test"
