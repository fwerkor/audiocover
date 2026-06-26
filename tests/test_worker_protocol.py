from __future__ import annotations

import json
import subprocess
import sys


def test_json_worker_keeps_handler_stdout_out_of_protocol() -> None:
    script = (
        "from audiocover.workers.json_worker import serve\n"
        "def run(payload):\n"
        "    print('handler progress on stdout')\n"
        "    return {'value': payload['value']}\n"
        "serve({'run': run})\n"
    )
    request = {"id": "req-1", "action": "run", "payload": {"value": 5}}

    process = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=False,
    )

    assert process.returncode == 0
    assert "handler progress on stdout" in process.stderr
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"id": "req-1", "ok": True, "result": {"value": 5}}
