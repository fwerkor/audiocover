from __future__ import annotations

import json
import os
import queue
import re
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


_JSON_DECODER = json.JSONDecoder()
_PROGRESS_RE = re.compile(
    r"(?P<percent>\d{1,3})%\|[^|]*\|\s*(?P<current>\d+)/(?:\s*)?(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[^<\]]+)(?:<(?P<remaining>[^,\]]+))?(?:,\s*(?P<rate>[^\]]+))?\]"
)
_NOISY_BACKEND_LOG_MARKERS = (
    "F0 inference time",
    "HuBERT inference time",
)


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
    "so-vits-svc": "so_vits_svc_worker",
    "demucs-separator": "demucs_separator_worker",
}

TRAINING_ORDER = ("so-vits-svc", "simple-timbre")
CONVERSION_ORDER = ("so-vits-svc", "simple-timbre")
SEPARATOR_ORDER = ("demucs-separator",)


def _response_from_text(text: str, request_id: str) -> dict[str, Any] | None:
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = _JSON_DECODER.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("id") == request_id and "ok" in data:
            return data
    return None


def _is_response_line(line: str, request_id: str) -> bool:
    return _response_from_text(line, request_id) is not None


def _progress_summary(line: str) -> str | None:
    match = _PROGRESS_RE.search(line)
    if not match:
        return None
    percent = max(0, min(100, int(match.group("percent"))))
    current = int(match.group("current"))
    total = int(match.group("total"))
    remaining = match.group("remaining") or ""
    rate = (match.group("rate") or "").strip()
    suffix_items = [item for item in (f"remaining {remaining}" if remaining else "", rate) if item]
    suffix = f" ({', '.join(suffix_items)})" if suffix_items else ""
    return f"{percent}% ({current}/{total}){suffix}"


def _should_suppress_backend_log(line: str) -> bool:
    return any(marker in line for marker in _NOISY_BACKEND_LOG_MARKERS)


def _prepare_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8:replace")
    env.setdefault("NO_COLOR", "1")
    env.setdefault("RICH_FORCE_TERMINAL", "0")
    if not sys.platform.startswith("win"):
        env.setdefault("LC_ALL", "C.UTF-8")
        env.setdefault("LANG", "C.UTF-8")
    if getattr(sys, "frozen", False):
        env["AUDIOCOVER_BINARY_CPU_ONLY"] = "1"
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
    return env


def _find_worker_response(stdout_lines: list[str], request_id: str) -> dict[str, Any]:
    parse_errors: list[str] = []
    for line in reversed([item for item in stdout_lines if item]):
        response = _response_from_text(line, request_id)
        if response is not None:
            return response
        if "{" in line:
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append(str(exc))
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
    if getattr(sys, "frozen", False):
        return (sys.executable, "--audiocover-worker", worker_name)
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

        env = _prepare_env()
        process = popen_hidden(
            list(runtime.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        events: queue.Queue[tuple[str, str]] = queue.Queue()
        last_progress: dict[str, str] = {}

        def log_worker_line(stream_name: str, line: str) -> None:
            if log is None or not line or _is_response_line(line, request["id"]):
                return
            progress = _progress_summary(line)
            if progress:
                if last_progress.get("value") != progress:
                    last_progress["value"] = progress
                    log(f"\r{worker_name} progress: {progress}")
                return
            if _should_suppress_backend_log(line):
                return
            prefix = "stderr" if stream_name == "stderr" else "stdout"
            log(f"{worker_name} {prefix}: {line}")

        def read_stream(stream_name: str, lines: list[str], stream) -> None:
            assert stream is not None
            for raw_line in stream:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
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
            log_worker_line(stream_name, line)

        for thread in threads:
            thread.join(timeout=0.2)

        while not events.empty():
            stream_name, line = events.get_nowait()
            log_worker_line(stream_name, line)

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
