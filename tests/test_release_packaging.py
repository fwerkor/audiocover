from __future__ import annotations

import importlib.util
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

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
        "scikit-learn>=1.4.0",
    ):
        assert dependency in normalized


def test_release_matrix_installs_so_vits_decoder_dependencies() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    for dependency in (
        "so-vits-svc-fork==4.2.30",
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "scikit-learn>=1.4.0",
    ):
        assert workflow.count(dependency) >= 4


def test_build_script_self_tests_so_vits_runtime_without_torchcodec() -> None:
    build_desktop = _build_desktop_module()

    assert "torchcodec" not in build_desktop.WORKER_COLLECTS["so-vits-svc"]
    assert build_desktop.RUNTIME_SELF_TESTS["so-vits-svc"] == "self_test"


def test_so_vits_runtime_pack_includes_hubert_import_chain() -> None:
    build_desktop = _build_desktop_module()

    hidden_imports = build_desktop.WORKER_HIDDEN_IMPORTS["so-vits-svc"]
    assert "transformers.models.hubert.modeling_hubert" in hidden_imports
    assert "torch._inductor.test_operators" in hidden_imports
    assert "torch._inductor" not in build_desktop.WORKER_EXCLUDES
