#!/usr/bin/env python3
"""Build backend worker runtimes, desktop bundles, runtime packs, and archives."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
SPEC_PATH = ROOT / "packaging" / "audiocover-gui.spec"
APP_DIR = DIST_DIR / "AudioCover"
MAC_APP_DIR = DIST_DIR / "AudioCover.app"
RUNTIME_DIR = ROOT / "backend-runtimes"
BUNDLE_ASSETS_DIR = BUILD_DIR / "audiocover-bundle-assets"
WORKERS = {
    "simple-timbre": "audiocover.workers.simple_timbre_worker",
    "so-vits-svc": "audiocover.workers.so_vits_svc_worker",
    "demucs-separator": "audiocover.workers.demucs_separator_worker",
}
WORKER_SETS = {
    "desktop": ("simple-timbre",),
    "simple": ("simple-timbre",),
    "so-vits-svc": ("so-vits-svc",),
    "demucs": ("demucs-separator",),
    "all": tuple(WORKERS),
}
WORKER_COLLECTS = {
    "so-vits-svc": ("so_vits_svc_fork", "librosa", "sklearn", "tensorboard"),
    "demucs-separator": ("demucs",),
}
WORKER_HIDDEN_IMPORTS = {
    "so-vits-svc": (
        "transformers.models.hubert.modeling_hubert",
        "torch._inductor.test_operators",
        "torch.utils.tensorboard",
        "torch.utils.tensorboard.writer",
    ),
}
RUNTIME_SELF_TESTS = {
    "so-vits-svc": "self_test",
}
CONTENTVEC_REVISION = "1f864bcb6b3f6138c4c52d83044d0242ea6274d3"
SOVITS_INIT_REVISION = "cf12670fbb4c125a2d1502973bf8d5ab37d6be7e"


def _hf_model_url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


def _hf_dataset_url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


RUNTIME_ASSETS = {
    "so-vits-svc": (
        (
            "content-vec-best/config.json",
            _hf_model_url("fwerkor/content-vec-best", CONTENTVEC_REVISION, "config.json"),
        ),
        (
            "content-vec-best/pytorch_model.bin",
            _hf_model_url("fwerkor/content-vec-best", CONTENTVEC_REVISION, "pytorch_model.bin"),
        ),
        (
            "so-vits-svc-init/D_0.pth",
            _hf_dataset_url(
                "ms903/sovits4.0-768vec-layer12",
                SOVITS_INIT_REVISION,
                "sovits_768l12_pre_large_320k/clean_D_320000.pth",
            ),
        ),
        (
            "so-vits-svc-init/G_0.pth",
            _hf_dataset_url(
                "ms903/sovits4.0-768vec-layer12",
                SOVITS_INIT_REVISION,
                "sovits_768l12_pre_large_320k/clean_G_320000.pth",
            ),
        ),
    ),
}
WORKER_EXCLUDES = (
    "IPython",
    "notebook",
    "pandas",
    "pyarrow",
    "pytest",
    # Optional modules that PyInstaller hooks probe even though AudioCover workers do not use them.
    "pycparser.lextab",
    "pycparser.yacctab",
    "scipy.special._cdflib",
    "torch.distributed._shard.checkpoint",
    "torch.distributed._sharded_tensor",
    "torch.distributed._sharding_spec",
    "triton",
)


def _normalize_system(value: str) -> str:
    value = value.lower()
    if value.startswith("win"):
        return "windows"
    if value == "darwin":
        return "macos"
    if value == "linux":
        return "linux"
    return value.replace(" ", "-")


def _normalize_machine(value: str) -> str:
    value = value.lower()
    aliases = {
        "amd64": "x64",
        "x86_64": "x64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }
    return aliases.get(value, value.replace(" ", "-"))


def _platform_name() -> str:
    return f"{_normalize_system(platform.system())}-{_normalize_machine(platform.machine())}"


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def _download_file(url: str, output_path: Path, *, retries: int = 8) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".download")
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"asset present: {output_path.relative_to(ROOT)}", flush=True)
        return

    for attempt in range(1, retries + 1):
        headers = {"User-Agent": "AudioCover release builder"}
        existing = temp_path.stat().st_size if temp_path.exists() else 0
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        mode = "ab" if existing else "wb"
        try:
            print(
                f"downloading asset {output_path.relative_to(ROOT)}"
                f" attempt {attempt}/{retries} offset={existing}",
                flush=True,
            )
            with urllib.request.urlopen(request, timeout=180) as response, temp_path.open(mode) as handle:
                if existing and response.status == 200:
                    handle.seek(0)
                    handle.truncate()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            temp_path.replace(output_path)
            print(f"downloaded asset: {output_path.relative_to(ROOT)}", flush=True)
            return
        except (TimeoutError, urllib.error.URLError, http.client.HTTPException, OSError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            time.sleep(min(30, attempt * 3))


def _install_runtime_assets(worker_name: str, worker_dist: Path) -> None:
    assets = RUNTIME_ASSETS.get(worker_name, ())
    if not assets:
        return
    asset_root = worker_dist / "assets"
    for relative_name, url in assets:
        _download_file(url, asset_root / relative_name)


def install_bundle_assets(*, clean: bool = False) -> None:
    if clean:
        shutil.rmtree(BUNDLE_ASSETS_DIR, ignore_errors=True)
    for assets in RUNTIME_ASSETS.values():
        for relative_name, url in assets:
            _download_file(url, BUNDLE_ASSETS_DIR / relative_name)


def _capture_json(command: list[str], *, input_json: dict) -> dict:
    process = subprocess.run(
        command,
        cwd=ROOT,
        input=json.dumps(input_json, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr[-4000:] or process.stdout[-4000:])
    lines = process.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError(f"{command[0]} returned no output")
    response = json.loads(lines[-1])
    if not response.get("ok"):
        raise RuntimeError(response.get("error") or "worker returned an error")
    return response["result"]


def _fail(message: str) -> NoReturn:
    raise SystemExit(message)


def _worker_executable(worker_name: str, runtime_dir: Path = RUNTIME_DIR) -> Path:
    suffix = ".exe" if platform.system().lower().startswith("win") else ""
    return runtime_dir / worker_name / f"{worker_name}{suffix}"


def _bundle_executable() -> Path:
    suffix = ".exe" if _normalize_system(platform.system()) == "windows" else ""
    onefile = DIST_DIR / f"AudioCover{suffix}"
    if onefile.is_file():
        return onefile
    system = _normalize_system(platform.system())
    if system == "windows":
        return APP_DIR / "AudioCover.exe"
    if system == "macos" and MAC_APP_DIR.exists():
        return MAC_APP_DIR / "Contents" / "MacOS" / "AudioCover"
    return APP_DIR / "AudioCover"


def _worker_names(worker_set: str) -> tuple[str, ...]:
    try:
        return WORKER_SETS[worker_set]
    except KeyError as exc:
        raise SystemExit(f"unknown worker set: {worker_set}") from exc


def build_workers(worker_names: tuple[str, ...], *, clean: bool, runtime_dir: Path = RUNTIME_DIR) -> None:
    if clean:
        shutil.rmtree(runtime_dir, ignore_errors=True)
        shutil.rmtree(BUILD_DIR / "backend-runtimes", ignore_errors=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = BUILD_DIR / "backend-runtimes" / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    for worker_name in worker_names:
        module = WORKERS[worker_name]
        worker_dist = runtime_dir / worker_name
        worker_build = BUILD_DIR / "backend-runtimes" / worker_name
        shutil.rmtree(worker_dist, ignore_errors=True)
        shutil.rmtree(worker_build, ignore_errors=True)
        command = [
            "pyinstaller",
            "--log-level",
            "ERROR",
            "--clean",
            "--noconfirm",
            "--onefile",
            "--name",
            worker_name,
            "--distpath",
            str(worker_dist),
            "--workpath",
            str(worker_build),
            "--specpath",
            str(spec_dir),
            "--paths",
            str(ROOT / "src"),
        ]
        for collected in WORKER_COLLECTS.get(worker_name, ()):  # backend packages can have dynamic imports
            command.extend(["--collect-all", collected])
        for hidden_import in WORKER_HIDDEN_IMPORTS.get(worker_name, ()):  # transformers/torch lazy imports
            command.extend(["--hidden-import", hidden_import])
        for excluded in WORKER_EXCLUDES:
            command.extend(["--exclude-module", excluded])
        command.append(str(ROOT / "src" / "audiocover" / "workers" / f"{module.rsplit('.', 1)[-1]}.py"))
        _run(command)
        executable = _worker_executable(worker_name, runtime_dir)
        if not executable.exists():
            raise FileNotFoundError(f"worker executable was not created: {executable}")
        _install_runtime_assets(worker_name, worker_dist)


def smoke_test_workers(
    worker_names: tuple[str, ...],
    *,
    runtime_dir: Path = RUNTIME_DIR,
    require_available: bool = False,
) -> None:
    for worker_name in worker_names:
        executable = _worker_executable(worker_name, runtime_dir)
        if not executable.exists():
            raise FileNotFoundError(f"worker executable was not found: {executable}")
        result = _capture_json(
            [str(executable)],
            input_json={"id": "smoke", "action": "capabilities", "payload": {}},
        )
        print(
            f"runtime {worker_name}: available={result.get('available')} actions={result.get('actions')} reason={result.get('reason') or ''}",
            flush=True,
        )
        if require_available and not result.get("available"):
            raise RuntimeError(f"runtime pack worker is inactive: {worker_name}: {result.get('reason')}")
        self_test_action = RUNTIME_SELF_TESTS.get(worker_name)
        if result.get("available") and self_test_action:
            self_test = _capture_json(
                [str(executable)],
                input_json={"id": "self-test", "action": self_test_action, "payload": {}},
            )
            print(
                f"runtime {worker_name}: self-test passed checks={self_test.get('checks', [])}",
                flush=True,
            )

    if "simple-timbre" in worker_names:
        simple = _capture_json(
            [str(_worker_executable("simple-timbre", runtime_dir))],
            input_json={"id": "smoke", "action": "capabilities", "payload": {}},
        )
        if not simple.get("available") or "train" not in simple.get("actions", []):
            raise RuntimeError("simple-timbre worker must be available for offline packaged operation")


def smoke_test_embedded_workers(worker_names: tuple[str, ...], *, require_available: bool = True) -> None:
    executable = _bundle_executable()
    if not executable.exists():
        raise FileNotFoundError(f"Desktop executable was not found: {executable}")
    for worker_name in worker_names:
        result = _capture_json(
            [str(executable), "--audiocover-worker", worker_name],
            input_json={"id": "smoke", "action": "capabilities", "payload": {}},
        )
        print(
            f"embedded runtime {worker_name}: available={result.get('available')} "
            f"actions={result.get('actions')} reason={result.get('reason') or ''}",
            flush=True,
        )
        if require_available and not result.get("available"):
            raise RuntimeError(f"embedded worker is inactive: {worker_name}: {result.get('reason')}")
        self_test_action = RUNTIME_SELF_TESTS.get(worker_name)
        if result.get("available") and self_test_action:
            self_test = _capture_json(
                [str(executable), "--audiocover-worker", worker_name],
                input_json={"id": "self-test", "action": self_test_action, "payload": {}},
            )
            print(
                f"embedded runtime {worker_name}: self-test passed checks={self_test.get('checks', [])}",
                flush=True,
            )


def smoke_test_bundle() -> None:
    executable = _bundle_executable()
    if not executable.exists():
        raise FileNotFoundError(f"Desktop executable was not found: {executable}")
    _run([str(executable), "--smoke-test"])


def build_bundle(clean: bool) -> None:
    if clean:
        shutil.rmtree(DIST_DIR, ignore_errors=True)
        shutil.rmtree(BUILD_DIR / "audiocover-gui", ignore_errors=True)
    install_bundle_assets(clean=clean)
    _run(["pyinstaller", "--log-level", "ERROR", str(SPEC_PATH), "--clean", "--noconfirm"])
    executable = _bundle_executable()
    if not executable.exists():
        raise FileNotFoundError(f"PyInstaller did not create the desktop executable at {executable}")


def package_bundle() -> Path:
    executable = _bundle_executable()
    if not executable.exists():
        raise FileNotFoundError(f"Desktop executable was not found: {executable}")

    system = _normalize_system(platform.system())
    bundle_root = APP_DIR if APP_DIR.exists() else MAC_APP_DIR if MAC_APP_DIR.exists() else executable
    if bundle_root.is_dir():
        artifact_base = DIST_DIR / f"audiocover-{_platform_name()}"
        archive_format = "zip" if system == "windows" else "gztar"
        artifact = Path(
            shutil.make_archive(str(artifact_base), archive_format, root_dir=DIST_DIR, base_dir=bundle_root.name)
        )
        outputs = _split_large_artifact(artifact)
        for item in outputs:
            print(f"Built {item.relative_to(ROOT)}", flush=True)
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a", encoding="utf-8") as handle:
                handle.write(f"artifact={outputs[0].as_posix()}\n")
        return outputs[0]

    suffix = ".exe" if system == "windows" else ""
    artifact = DIST_DIR / f"audiocover-{_platform_name()}{suffix}"
    if executable.resolve() != artifact.resolve():
        shutil.copy2(executable, artifact)
    if system != "windows":
        artifact.chmod(artifact.stat().st_mode | 0o755)
    outputs = _split_large_artifact(artifact)
    for item in outputs:
        print(f"Built {item.relative_to(ROOT)}", flush=True)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"artifact={outputs[0].as_posix()}\n")
    return outputs[0]


def _split_large_artifact(artifact: Path, *, max_part_bytes: int = 1900 * 1024 * 1024) -> list[Path]:
    if artifact.stat().st_size <= max_part_bytes:
        return [artifact]

    parts: list[Path] = []
    index = 1
    with artifact.open("rb") as src:
        while True:
            chunk = src.read(max_part_bytes)
            if not chunk:
                break
            part = artifact.with_name(f"{artifact.name}.part{index:03d}")
            part.write_bytes(chunk)
            parts.append(part)
            index += 1
    artifact.unlink()
    print(f"Split large artifact into {len(parts)} parts: {artifact.name}", flush=True)
    return parts


def package_runtime_pack(pack_name: str, runtime_dir: Path = RUNTIME_DIR) -> Path:
    if not runtime_dir.exists():
        raise FileNotFoundError(f"runtime directory does not exist: {runtime_dir}")
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    artifact_base = DIST_DIR / f"audiocover-backend-runtimes-{pack_name}-{_platform_name()}"
    archive_format = "zip" if _normalize_system(platform.system()) == "windows" else "gztar"
    artifact = Path(
        shutil.make_archive(str(artifact_base), archive_format, root_dir=ROOT, base_dir=runtime_dir.name)
    )
    outputs = _split_large_artifact(artifact)
    for item in outputs:
        print(f"Built {item.relative_to(ROOT)}", flush=True)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"artifact={outputs[0].as_posix()}\n")
    return outputs[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-clean", action="store_true", help="reuse existing build directories")
    parser.add_argument("--worker-set", default="all", choices=sorted(WORKER_SETS), help="worker group to build or smoke-test")
    parser.add_argument("--workers-only", action="store_true", help="build and smoke-test backend runtimes only")
    parser.add_argument("--skip-workers", action="store_true", help="reuse existing backend runtimes")
    parser.add_argument("--runtime-smoke-test-only", action="store_true", help="smoke-test existing backend runtimes")
    parser.add_argument("--runtime-pack", choices=sorted(WORKER_SETS), help="build and archive a backend runtime pack")
    parser.add_argument(
        "--prepare-assets-only",
        action="store_true",
        help="download packaged model assets into build/audiocover-bundle-assets",
    )
    parser.add_argument(
        "--smoke-test-only",
        action="store_true",
        help="run the frozen executable smoke test without rebuilding",
    )
    args = parser.parse_args()

    worker_set = args.runtime_pack or args.worker_set
    worker_names = _worker_names(worker_set)

    if args.prepare_assets_only:
        install_bundle_assets(clean=not args.no_clean)
        return

    if args.runtime_smoke_test_only:
        smoke_test_workers(worker_names)
        return

    if args.smoke_test_only:
        if not _bundle_executable().exists():
            _fail("No existing desktop executable found; build it before running --smoke-test-only")
        smoke_test_embedded_workers(worker_names)
        smoke_test_bundle()
        return

    if args.runtime_pack:
        build_workers(worker_names, clean=not args.no_clean)
        smoke_test_workers(worker_names, require_available=args.runtime_pack not in {"desktop", "simple"})
        package_runtime_pack(args.runtime_pack)
        return

    if args.workers_only:
        if not args.skip_workers:
            build_workers(worker_names, clean=not args.no_clean)
        smoke_test_workers(worker_names)
        return

    build_bundle(clean=not args.no_clean)
    smoke_test_embedded_workers(worker_names)
    smoke_test_bundle()
    package_bundle()


if __name__ == "__main__":
    main()
