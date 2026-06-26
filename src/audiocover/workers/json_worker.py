from __future__ import annotations

import json
import multiprocessing
import sys
import traceback
from collections.abc import Callable
from contextlib import redirect_stdout
from typing import Any

Handler = Callable[[dict[str, Any]], dict[str, Any]]


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _write_response(response: dict[str, Any]) -> None:
    text = json.dumps(response, ensure_ascii=False)
    sys.stdout.write("\n" + text + "\n")
    sys.stdout.flush()


def serve(handlers: dict[str, Handler]) -> None:
    multiprocessing.freeze_support()
    _configure_stdio()
    raw = sys.stdin.read()
    request_id = None
    try:
        request = json.loads(raw)
        request_id = request.get("id")
        action = str(request.get("action") or "")
        payload = request.get("payload") or {}
        if action not in handlers:
            raise ValueError(f"unsupported action: {action}")
        with redirect_stdout(sys.stderr):
            result = handlers[action](payload)
        response = {"id": request_id, "ok": True, "result": result}
    except Exception as exc:
        response = {
            "id": request_id,
            "ok": False,
            "error": f"{exc}\n{traceback.format_exc()}",
        }
    _write_response(response)
