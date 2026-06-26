from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from typing import Any

CREATE_NO_WINDOW = 0x08000000


def _hidden_window_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    if not sys.platform.startswith("win"):
        return kwargs

    hidden_kwargs = dict(kwargs)
    hidden_kwargs["creationflags"] = int(hidden_kwargs.get("creationflags") or 0) | int(
        getattr(subprocess, "CREATE_NO_WINDOW", CREATE_NO_WINDOW)
    )

    if "startupinfo" not in hidden_kwargs and hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
        startupinfo.dwFlags |= int(getattr(subprocess, "STARTF_USESHOWWINDOW", 1))
        startupinfo.wShowWindow = int(getattr(subprocess, "SW_HIDE", 0))
        hidden_kwargs["startupinfo"] = startupinfo

    return hidden_kwargs


def run_hidden(args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, **_hidden_window_kwargs(kwargs))


def popen_hidden(args: Sequence[str], **kwargs: Any) -> subprocess.Popen[str]:
    return subprocess.Popen(args, **_hidden_window_kwargs(kwargs))
