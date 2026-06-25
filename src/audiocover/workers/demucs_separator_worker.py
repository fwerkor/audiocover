from __future__ import annotations

import contextlib
import io
import shutil
import sys
from pathlib import Path
from typing import Any

from audiocover.workers.json_worker import serve


def _demucs_available() -> bool:
    try:
        import demucs.separate  # noqa: F401
    except Exception:
        return False
    return True


def capabilities(_: dict[str, Any]) -> dict[str, Any]:
    available = _demucs_available()
    return {
        "available": available,
        "actions": ["separate"] if available else [],
        "description": "Demucs source-separation worker",
        "reason": None if available else "demucs is not present in this isolated worker runtime",
    }


def separate(payload: dict[str, Any]) -> dict[str, Any]:
    if not _demucs_available():
        raise RuntimeError("demucs is not present in this isolated worker runtime")
    input_wav = Path(payload["input"])
    out_dir = Path(payload["out_dir"])
    model = str(payload.get("model") or "htdemucs_ft")
    two_stems = str(payload.get("two_stems") or "vocals")
    shifts = str(payload.get("shifts") or 1)
    overlap = str(payload.get("overlap") or 0.25)
    device = str(payload.get("device") or "cpu")
    vocals = Path(payload["vocals"])
    instrumental = Path(payload["instrumental"])
    raw = out_dir / "demucs_raw"
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        model,
        "--two-stems",
        two_stems,
        "--shifts",
        shifts,
        "--overlap",
        overlap,
        "-d",
        device,
        "-o",
        str(raw),
        str(input_wav),
    ]
    segment = payload.get("segment")
    if segment:
        cmd.insert(-1, str(segment))
        cmd.insert(-2, "--segment")
    cmd.extend(str(item) for item in payload.get("extra_args") or [])
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = 0
    old_argv = sys.argv[:]
    try:
        from demucs.separate import main as demucs_main

        sys.argv = ["demucs.separate", *cmd[3:]]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                demucs_main()
            except SystemExit as exc:
                exit_code = int(exc.code or 0) if isinstance(exc.code, int) else 1
    finally:
        sys.argv = old_argv
    log_path = out_dir / "demucs-worker.log"
    log_path.write_text(
        "COMMAND\n" + " ".join(cmd) + "\n\nSTDOUT\n" + stdout.getvalue() + "\n\nSTDERR\n" + stderr.getvalue(),
        encoding="utf-8",
    )
    if exit_code != 0:
        raise RuntimeError(stderr.getvalue()[-4000:] or stdout.getvalue()[-4000:])
    stem_dir = raw / model / input_wav.stem
    vocals.parent.mkdir(parents=True, exist_ok=True)
    instrumental.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(stem_dir / "vocals.wav", vocals)
    shutil.copyfile(stem_dir / "no_vocals.wav", instrumental)
    return {"vocals": str(vocals), "instrumental": str(instrumental), "log": str(log_path)}


def main() -> None:
    serve({"capabilities": capabilities, "separate": separate})


if __name__ == "__main__":
    main()
