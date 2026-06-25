from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any


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
        self.runtime_roots = runtime_roots or _runtime_roots()

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

    def invoke(self, worker_name: str, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        runtime = self.runtimes.get(worker_name)
        if runtime is None:
            raise BackendRuntimeError(f"backend runtime is not bundled: {worker_name}")
        request = {"id": uuid.uuid4().hex, "action": action, "payload": payload}
        process = subprocess.run(
            list(runtime.command),
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            raise BackendRuntimeError(
                f"{worker_name} runtime failed with exit code {process.returncode}: "
                f"{process.stderr[-4000:] or process.stdout[-4000:]}"
            )
        stdout = process.stdout.strip().splitlines()
        if not stdout:
            raise BackendRuntimeError(f"{worker_name} runtime returned no JSON response")
        try:
            response = json.loads(stdout[-1])
        except json.JSONDecodeError as exc:
            raise BackendRuntimeError(
                f"{worker_name} runtime returned invalid JSON: {stdout[-1][:1000]}"
            ) from exc
        if response.get("id") != request["id"]:
            raise BackendRuntimeError(f"{worker_name} runtime response id mismatch")
        if not response.get("ok"):
            error = response.get("error") or "unknown runtime error"
            raise BackendRuntimeError(f"{worker_name} runtime {action} failed: {error}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise BackendRuntimeError(f"{worker_name} runtime returned a non-object result")
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
