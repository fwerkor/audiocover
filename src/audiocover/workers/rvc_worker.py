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
        "actions": ["convert"] if available else [],
        "description": "RVC isolated inference worker",
        "reason": reason,
    }


def train(payload: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("The packaged rvc-python runtime provides conversion only; train with another packaged runtime or import an existing RVC model package.")


def convert(payload: dict[str, Any]) -> dict[str, Any]:
    available, reason = _available()
    if not available:
        raise RuntimeError(reason)

    from pathlib import Path

    from rvc_python.infer import RVCInference

    model_path = payload.get("model_path")
    if not model_path:
        raise RuntimeError("RVC conversion requires model_path in the model package")
    output_path = Path(payload["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_path = payload.get("index_path") or ""
    rvc = RVCInference(
        device=payload.get("device") or "cpu:0",
        model_path=str(model_path),
        index_path=str(index_path) if index_path else "",
        version=payload.get("version") or "v2",
    )
    rvc.set_params(
        f0method=payload.get("f0_method") or "rmvpe",
        f0up_key=int(payload.get("transpose") or 0),
        index_rate=float(payload.get("index_rate") or 0.75),
        filter_radius=int(payload.get("filter_radius") or 3),
        resample_sr=int(payload.get("resample_sr") or 0),
        rms_mix_rate=float(payload.get("rms_mix_rate") or 0.25),
        protect=float(payload.get("protect") or 0.33),
    )
    rvc.infer_file(str(payload["input"]), str(output_path))
    return {"output": str(output_path)}


def main() -> None:
    serve({"capabilities": capabilities, "train": train, "convert": convert})


if __name__ == "__main__":
    main()
