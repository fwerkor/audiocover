from __future__ import annotations

import importlib.util
from typing import Any

from audiocover.workers.json_worker import serve


def _available() -> tuple[bool, str | None]:
    if importlib.util.find_spec("so_vits_svc_fork") is None and importlib.util.find_spec("so_vits_svc") is None:
        return False, "So-VITS-SVC engine package is not present in this isolated worker runtime"
    return True, None


def capabilities(_: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    return {
        "available": available,
        "actions": ["train", "convert"] if available else [],
        "description": "So-VITS-SVC isolated worker adapter",
        "reason": reason,
    }


def train(payload: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    if not available:
        raise RuntimeError(reason)
    raise RuntimeError(
        "So-VITS-SVC adapter is bundled as an isolated worker, but this build does not include a "
        "supported non-interactive training API."
    )


def convert(payload: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    if not available:
        raise RuntimeError(reason)
    raise RuntimeError(
        "So-VITS-SVC adapter is bundled as an isolated worker, but this build does not include a "
        "supported non-interactive inference API."
    )


def main() -> None:
    serve({"capabilities": capabilities, "train": train, "convert": convert})


if __name__ == "__main__":
    main()
