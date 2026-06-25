from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import integrated_lufs, load_audio, peak_dbfs
from .config import QcConfig


def analyze_audio(path: Path, cfg: QcConfig | None = None) -> dict:
    cfg = cfg or QcConfig()
    data, sr = load_audio(path)
    duration = len(data) / sr if sr else 0.0
    clipping_ratio = float(np.mean(np.abs(data) >= 0.999)) if data.size else 0.0
    frame = max(1, int(sr * 0.05))
    mono = np.mean(np.abs(data), axis=1) if data.size else np.array([], dtype=np.float32)
    n = len(mono) // frame
    if n:
        rms = np.sqrt(np.mean(mono[: n * frame].reshape(n, frame) ** 2, axis=1))
        silence_ratio = float(np.mean(rms < 10 ** (-55 / 20)))
    else:
        silence_ratio = 1.0
    loudness = integrated_lufs(data, sr)
    warnings: list[str] = []
    if clipping_ratio > cfg.max_clipping_ratio:
        warnings.append("clipping")
    if duration < cfg.min_duration_seconds:
        warnings.append("short")
    if silence_ratio > cfg.max_silence_ratio:
        warnings.append("mostly_silence")
    if loudness is not None and loudness > cfg.warn_lufs_above:
        warnings.append("hot_loudness")
    if loudness is not None and loudness < cfg.warn_lufs_below:
        warnings.append("quiet_loudness")
    return {
        "path": str(path),
        "sample_rate": sr,
        "channels": data.shape[1] if data.ndim == 2 else 1,
        "duration_seconds": duration,
        "peak_dbfs": peak_dbfs(data),
        "lufs": loudness,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "warnings": warnings,
    }
