from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any


def expand_template(template: str, **values: Any) -> list[str]:
    safe = {key: shlex.quote(str(value)) for key, value in values.items()}
    return shlex.split(template.format(**safe))


def run_template(template: str, log_file: Path | None = None, **values: Any) -> None:
    cmd = expand_template(template, **values)
    process = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            "COMMAND\n"
            + " ".join(cmd)
            + "\n\nSTDOUT\n"
            + process.stdout
            + "\n\nSTDERR\n"
            + process.stderr,
            encoding="utf-8",
        )
    if process.returncode != 0:
        raise RuntimeError(process.stderr[-4000:])
