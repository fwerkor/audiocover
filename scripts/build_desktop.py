#!/usr/bin/env python3
"""Build backend worker runtimes, desktop bundle, and release archives."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
SPEC_PATH = ROOT / "packaging" / "audiocover-gui.spec"
APP_DIR = DIST_DIR / "AudioCover"
MAC_APP_DIR = DIST_DIR / "AudioCover.app"
RUNTIME_DIR = ROOT / "backend-runtimes"
WORKERS = {
    "simple-timbre": "audiocover.workers.simple_timbre_worker",
    "rvc": "audiocover.workers.rvc_worker",
    "so-vits-svc": "audiocover.workers.so_vits_svc_worker",
    "demucs-separator": "audiocover.workers.demucs_separator_worker",
}
WORKER_EXCLUDES = (
    "IPython",
    "matplotlib",
    "notebook",
    "pandas",
    "pyarrow",
    "pytest",
    "sklearn",
    "tkinter",
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


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=env)


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


def _worker_executable(worker_name: str) -> Path:
    suffix = ".exe" if platform.system().lower().startswith("win") else ""
    return RUNTIME_DIR / worker_name / f"{worker_name}{suffix}"


def _bundle_executable() -> Path:
    system = _normalize_system(platform.system())
    if system == "windows":
        return APP_DIR / "AudioCover.exe"
    if system == "macos" and MAC_APP_DIR.exists():
        return MAC_APP_DIR / "Contents" / "MacOS" / "AudioCover"
    return APP_DIR / "AudioCover"


def build_workers(clean: bool) -> None:
    if clean:
        shutil.rmtree(RUNTIME_DIR, ignore_errors=True)
        shutil.rmtree(BUILD_DIR / "backend-runtimes", ignore_errors=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    spec_dir = BUILD_DIR / "backend-runtimes" / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    for worker_name, module in WORKERS.items():
        worker_dist = RUNTIME_DIR / worker_name
        worker_build = BUILD_DIR / "backend-runtimes" / worker_name
        shutil.rmtree(worker_dist, ignore_errors=True)
        shutil.rmtree(worker_build, ignore_errors=True)
        command = [
            "pyinstaller",
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
        for excluded in WORKER_EXCLUDES:
            command.extend(["--exclude-module", excluded])
        command.append(str(ROOT / "src" / "audiocover" / "workers" / f"{module.rsplit('.', 1)[-1]}.py"))
        _run(command)
        executable = _worker_executable(worker_name)
        if not executable.exists():
            raise FileNotFoundError(f"worker executable was not created: {executable}")


def smoke_test_workers() -> None:
    for worker_name in WORKERS:
        executable = _worker_executable(worker_name)
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
    simple = _capture_json(
        [str(_worker_executable("simple-timbre"))],
        input_json={"id": "smoke", "action": "capabilities", "payload": {}},
    )
    if not simple.get("available") or "train" not in simple.get("actions", []):
        raise RuntimeError("simple-timbre worker must be available for offline packaged operation")


def smoke_test_bundle() -> None:
    executable = _bundle_executable()
    if not executable.exists():
        raise FileNotFoundError(f"Desktop executable was not found: {executable}")
    env = os.environ.copy()
    env["AUDIOCOVER_BACKEND_RUNTIMES"] = str(RUNTIME_DIR)
    _run([str(executable), "--smoke-test"], env=env)


def build_bundle(clean: bool) -> None:
    if clean:
        shutil.rmtree(DIST_DIR, ignore_errors=True)
        shutil.rmtree(BUILD_DIR / "audiocover-gui", ignore_errors=True)
    _run(["pyinstaller", str(SPEC_PATH), "--clean", "--noconfirm"])
    if not (APP_DIR.exists() or MAC_APP_DIR.exists()):
        raise FileNotFoundError(f"PyInstaller did not create {APP_DIR} or {MAC_APP_DIR}")


def package_bundle() -> Path:
    system = _normalize_system(platform.system())
    machine = _normalize_machine(platform.machine())
    artifact_base = DIST_DIR / f"audiocover-{system}-{machine}"

    package_dir = MAC_APP_DIR if MAC_APP_DIR.exists() else APP_DIR
    archive_format = "zip" if system == "windows" else "gztar"
    artifact = Path(
        shutil.make_archive(str(artifact_base), archive_format, root_dir=DIST_DIR, base_dir=package_dir.name)
    )

    print(f"Built {artifact.relative_to(ROOT)}", flush=True)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"artifact={artifact.as_posix()}\n")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-clean", action="store_true", help="reuse existing build directories")
    parser.add_argument("--workers-only", action="store_true", help="build and smoke-test backend runtimes only")
    parser.add_argument("--skip-workers", action="store_true", help="reuse existing backend runtimes")
    parser.add_argument("--runtime-smoke-test-only", action="store_true", help="smoke-test existing backend runtimes")
    parser.add_argument(
        "--smoke-test-only",
        action="store_true",
        help="run the frozen executable smoke test without rebuilding",
    )
    args = parser.parse_args()

    if args.runtime_smoke_test_only:
        smoke_test_workers()
        return

    if args.smoke_test_only:
        if not (APP_DIR.exists() or MAC_APP_DIR.exists()):
            _fail("No existing desktop bundle found; build it before running --smoke-test-only")
        smoke_test_workers()
        smoke_test_bundle()
        return

    if not args.skip_workers:
        build_workers(clean=not args.no_clean)
    smoke_test_workers()
    if args.workers_only:
        return

    build_bundle(clean=not args.no_clean)
    smoke_test_bundle()
    package_bundle()


if __name__ == "__main__":
    main()
