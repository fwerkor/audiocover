from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from .process import popen_hidden


class BackendRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    name: str
    command: tuple[str, ...]
    source: str


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    name: str
    available: bool
    actions: tuple[str, ...]
    description: str = ""
    reason: str | None = None

    def supports(self, action: str) -> bool:
        return self.available and action in self.actions


WORKER_MODULES = {
    "simple-timbre": "simple_timbre_worker",
    "rvc": "rvc_worker",
    "so-vits-svc": "so_vits_svc_worker",
    "demucs-separator": "demucs_separator_worker",
}

TRAINING_ORDER = ("rvc", "so-vits-svc", "simple-timbre")
CONVERSION_ORDER = ("rvc", "so-vits-svc", "simple-timbre")
SEPARATOR_ORDER = ("demucs-separator",)


def _is_response_line(line: str, request_id: str) -> bool:
    if not line.startswith("{"):
        return False
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return False
    return isinstance(data, dict) and data.get("id") == request_id and "ok" in data


def _find_worker_response(stdout_lines: list[str], request_id: str) -> dict[str, Any]:
    parse_errors: list[str] = []
    for line in reversed([item for item in stdout_lines if item]):
        if not line.startswith("{"):
            continue
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))
            continue
        if not isinstance(response, dict):
            continue
        if response.get("id") == request_id and "ok" in response:
            return response
    tail = "\n".join(stdout_lines)[-4000:]
    details = f"; parse errors: {parse_errors[-3:]}" if parse_errors else ""
    raise BackendRuntimeError(f"worker did not return a valid JSON response{details}: {tail}")


def _runtime_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("AUDIOCOVER_BACKEND_RUNTIMES")
    if env:
        roots.extend(Path(item).expanduser() for item in env.split(os.pathsep) if item)

    exe_dir = Path(sys.executable).resolve().parent
    roots.extend(
        [
            exe_dir / "backend-runtimes",
            exe_dir.parent / "backend-runtimes",
            exe_dir.parent.parent / "backend-runtimes",
            exe_dir.parent.parent.parent / "backend-runtimes",
            Path.cwd() / "backend-runtimes",
        ]
    )

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass) / "backend-runtimes")

    source_root = Path(__file__).resolve().parents[2]
    roots.append(source_root / "backend-runtimes")

    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve() if root.exists() else root.absolute()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    return unique


def _exe_names(worker_name: str) -> tuple[str, ...]:
    if sys.platform.startswith("win"):
        return (f"{worker_name}.exe", f"{worker_name}.cmd", f"{worker_name}.bat", worker_name)
    return (worker_name, f"{worker_name}.exe")


def _source_command(worker_name: str) -> tuple[str, ...] | None:
    module = WORKER_MODULES.get(worker_name)
    if module is None:
        return None
    module_path = Path(__file__).resolve().parent / "workers" / f"{module}.py"
    if not module_path.exists():
        return None
    return (sys.executable, "-m", f"audiocover.workers.{module}")


class BackendRuntimeManager:
    def __init__(self, runtime_roots: list[Path] | None = None) -> None:
        self.runtime_roots = _runtime_roots() if runtime_roots is None else runtime_roots

    @cached_property
    def runtimes(self) -> dict[str, RuntimeSpec]:
        found: dict[str, RuntimeSpec] = {}
        for worker_name in WORKER_MODULES:
            for root in self.runtime_roots:
                for filename in _exe_names(worker_name):
                    direct = root / worker_name / filename
                    flat = root / filename
                    for candidate in (direct, flat):
                        if candidate.exists() and candidate.is_file():
                            found[worker_name] = RuntimeSpec(
                                worker_name, (str(candidate),), f"frozen:{candidate}"
                            )
                            break
                    if worker_name in found:
                        break
                if worker_name in found:
                    break
            if worker_name not in found:
                command = _source_command(worker_name)
                if command:
                    found[worker_name] = RuntimeSpec(worker_name, command, "source")
        return found

    def invoke(
        self,
        worker_name: str,
        action: str,
        payload: dict[str, Any],
        *,
        log: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        runtime = self.runtimes.get(worker_name)
        if runtime is None:
            raise BackendRuntimeError(f"backend runtime is not bundled: {worker_name}")
        request = {"id": uuid.uuid4().hex, "action": action, "payload": payload}
        request_text = json.dumps(request, ensure_ascii=False)
        if log:
            log(f"starting {worker_name} runtime action: {action}")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        process = popen_hidden(
            list(runtime.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        events: queue.Queue[tuple[str, str]] = queue.Queue()

        def read_stream(stream_name: str, lines: list[str], stream) -> None:
            assert stream is not None
            for raw_line in stream:
                line = raw_line.rstrip("\r\n")
                lines.append(line)
                events.put((stream_name, line))

        threads = [
            threading.Thread(target=read_stream, args=("stdout", stdout_lines, process.stdout), daemon=True),
            threading.Thread(target=read_stream, args=("stderr", stderr_lines, process.stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()

        assert process.stdin is not None
        try:
            process.stdin.write(request_text)
            process.stdin.close()
        except BrokenPipeError:
            pass

        while process.poll() is None or any(thread.is_alive() for thread in threads) or not events.empty():
            try:
                stream_name, line = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if log and line and not _is_response_line(line, request["id"]):
                prefix = "stderr" if stream_name == "stderr" else "stdout"
                log(f"{worker_name} {prefix}: {line}")

        for thread in threads:
            thread.join(timeout=0.2)

        while not events.empty():
            stream_name, line = events.get_nowait()
            if log and line and not _is_response_line(line, request["id"]):
                prefix = "stderr" if stream_name == "stderr" else "stdout"
                log(f"{worker_name} {prefix}: {line}")

        return_code = process.returncode
        if return_code != 0:
            stderr_tail = "\n".join(stderr_lines)[-4000:]
            stdout_tail = "\n".join(stdout_lines)[-4000:]
            raise BackendRuntimeError(
                f"{worker_name} runtime failed with exit code {return_code}: "
                f"{stderr_tail or stdout_tail}"
            )
        if not [line for line in stdout_lines if line]:
            raise BackendRuntimeError(f"{worker_name} runtime returned no JSON response")
        response = _find_worker_response(stdout_lines, request["id"])
        if not response.get("ok"):
            error = response.get("error") or "unknown runtime error"
            raise BackendRuntimeError(f"{worker_name} runtime {action} failed: {error}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise BackendRuntimeError(f"{worker_name} runtime returned a non-object result")
        if log:
            log(f"finished {worker_name} runtime action: {action}")
        return result

    def capabilities(self, worker_name: str) -> RuntimeCapabilities:
        if worker_name not in self.runtimes:
            return RuntimeCapabilities(worker_name, False, (), reason="runtime executable is not bundled")
        try:
            result = self.invoke(worker_name, "capabilities", {})
        except Exception as exc:
            return RuntimeCapabilities(worker_name, False, (), reason=str(exc))
        actions = tuple(str(item) for item in result.get("actions", []))
        return RuntimeCapabilities(
            name=worker_name,
            available=bool(result.get("available")),
            actions=actions,
            description=str(result.get("description") or ""),
            reason=result.get("reason"),
        )

    def select_training_backend(self) -> str:
        return self._select(TRAINING_ORDER, "train")

    def select_conversion_backend(self, preferred: str | None = None) -> str:
        if preferred:
            cap = self.capabilities(preferred)
            if cap.supports("convert"):
                return preferred
            reason = cap.reason or "conversion is not supported"
            raise BackendRuntimeError(f"preferred backend runtime is not available: {preferred}: {reason}")
        return self._select(CONVERSION_ORDER, "convert")

    def require_training_backend(self, preferred: str) -> str:
        cap = self.capabilities(preferred)
        if cap.supports("train"):
            return preferred
        reason = cap.reason or "training is not supported"
        raise BackendRuntimeError(f"preferred backend runtime is not available: {preferred}: {reason}")

    def select_separator_backend(self) -> str | None:
        try:
            return self._select(SEPARATOR_ORDER, "separate")
        except BackendRuntimeError:
            return None

    def _select(self, names: tuple[str, ...], action: str) -> str:
        unavailable: list[str] = []
        for name in names:
            cap = self.capabilities(name)
            if cap.supports(action):
                return name
            reason = cap.reason or "unsupported"
            unavailable.append(f"{name}: {reason}")
        raise BackendRuntimeError(
            f"no bundled backend runtime supports {action}; " + "; ".join(unavailable)
        )
