#!/usr/bin/env python3
"""Build a desktop bundle and package it as a release artifact."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
SPEC_PATH = ROOT / "packaging" / "audiocover-gui.spec"
APP_DIR = DIST_DIR / "AudioCover"
MAC_APP_DIR = DIST_DIR / "AudioCover.app"


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


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def build_bundle(clean: bool) -> None:
    if clean:
        shutil.rmtree(DIST_DIR, ignore_errors=True)
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
    _run(["pyinstaller", str(SPEC_PATH), "--clean", "--noconfirm"])
    if not (APP_DIR.exists() or MAC_APP_DIR.exists()):
        raise FileNotFoundError(f"PyInstaller did not create {APP_DIR} or {MAC_APP_DIR}")


def package_bundle() -> Path:
    system = _normalize_system(platform.system())
    machine = _normalize_machine(platform.machine())
    artifact_base = DIST_DIR / f"audiocover-{system}-{machine}"

    package_dir = MAC_APP_DIR if MAC_APP_DIR.exists() else APP_DIR
    if system == "windows":
        artifact = Path(shutil.make_archive(str(artifact_base), "zip", root_dir=DIST_DIR, base_dir=package_dir.name))
    else:
        artifact = Path(shutil.make_archive(str(artifact_base), "gztar", root_dir=DIST_DIR, base_dir=package_dir.name))

    print(f"Built {artifact.relative_to(ROOT)}", flush=True)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"artifact={artifact.as_posix()}\n")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-clean", action="store_true", help="reuse existing build directories")
    args = parser.parse_args()

    build_bundle(clean=not args.no_clean)
    package_bundle()


if __name__ == "__main__":
    main()
