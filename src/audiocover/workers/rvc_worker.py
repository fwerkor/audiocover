from __future__ import annotations

import importlib.util
from typing import Any

from audiocover.workers.json_worker import serve


def _available() -> tuple[bool, str | None]:
    if importlib.util.find_spec("rvc_python") is None and importlib.util.find_spec("rvc") is None:
        return False, "RVC engine package is not present in this isolated worker runtime"
    return True, None


def capabilities(_: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    return {
        "available": available,
        "actions": ["train", "convert"] if available else [],
        "description": "RVC isolated worker adapter",
        "reason": reason,
    }


def train(payload: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    if not available:
        raise RuntimeError(reason)
    raise RuntimeError(
        "RVC adapter is bundled as an isolated worker, but this build does not include a supported "
        "non-interactive RVC training API."
    )


def convert(payload: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    if not available:
        raise RuntimeError(reason)
    raise RuntimeError(
        "RVC adapter is bundled as an isolated worker, but this build does not include a supported "
        "non-interactive RVC inference API."
    )


def main() -> None:
    serve({"capabilities": capabilities, "train": train, "convert": convert})


if __name__ == "__main__":
    main()
