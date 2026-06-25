from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable
from typing import Any

Handler = Callable[[dict[str, Any]], dict[str, Any]]


def serve(handlers: dict[str, Handler]) -> None:
    raw = sys.stdin.read()
    request_id = None
    try:
        request = json.loads(raw)
        request_id = request.get("id")
        action = str(request.get("action") or "")
        payload = request.get("payload") or {}
        if action not in handlers:
            raise ValueError(f"unsupported action: {action}")
        result = handlers[action](payload)
        response = {"id": request_id, "ok": True, "result": result}
    except Exception as exc:
        response = {
            "id": request_id,
            "ok": False,
            "error": f"{exc}\n{traceback.format_exc()}",
        }
    print(json.dumps(response, ensure_ascii=False), flush=True)
