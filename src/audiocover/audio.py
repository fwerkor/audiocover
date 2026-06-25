from __future__ import annotations

import math
import shutil
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def db_to_gain(db):
    return np.power(10.0, np.asarray(db) / 20.0)


def gain_to_db(gain: float) -> float:
    return 20.0 * math.log10(max(float(gain), 1e-12))


def peak_dbfs(data: np.ndarray) -> float:
    return gain_to_db(float(np.max(np.abs(data))) if data.size else 0.0)


def run_command(args: list[str], *, log_file: Path | None = None) -> None:
    process = subprocess.run(args, text=True, capture_output=True)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            "COMMAND\n"
            + " ".join(args)
            + "\n\nSTDOUT\n"
            + process.stdout
            + "\n\nSTDERR\n"
            + process.stderr,
            encoding="utf-8",
        )
    if process.returncode != 0:
        raise RuntimeError(process.stderr[-4000:])


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH")


def convert_to_wav(src: Path, dst: Path, sample_rate: int = 48000, channels: int = 2) -> Path:
    if src.suffix.lower() in {".wav", ".flac", ".ogg"}:
        try:
            data, sr = load_audio(src, sr=sample_rate, mono=channels == 1)
            data = match_channels(data, channels)
            return write_audio(dst, data, sr, subtype="PCM_24")
        except Exception:
            pass

    require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s24le",
            str(dst),
        ]
    )
    return dst


def load_audio(path: Path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    data, file_sr = sf.read(str(path), always_2d=True, dtype="float32")
    if mono:
        data = np.mean(data, axis=1, keepdims=True)
    if sr is not None and file_sr != sr:
        divisor = gcd(file_sr, sr)
        up = sr // divisor
        down = file_sr // divisor
        channels = [signal.resample_poly(data[:, i], up, down) for i in range(data.shape[1])]
        min_len = min(len(x) for x in channels)
        data = np.stack([x[:min_len] for x in channels], axis=1).astype(np.float32)
        file_sr = sr
    return data.astype(np.float32), file_sr


def write_audio(path: Path, data: np.ndarray, sr: int, subtype: str = "PCM_24") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.nan_to_num(data).astype(np.float32), sr, subtype=subtype)
    return path


def match_channels(data: np.ndarray, channels: int) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == channels:
        return data
    if data.shape[1] == 1:
        return np.repeat(data, channels, axis=1)
    if channels == 1:
        return np.mean(data, axis=1, keepdims=True)
    return np.tile(data, (1, math.ceil(channels / data.shape[1])))[:, :channels]


def match_length(data: np.ndarray, length: int) -> np.ndarray:
    if len(data) > length:
        return data[:length]
    if len(data) < length:
        return np.pad(data, [(0, length - len(data)), (0, 0)])
    return data


def biquad_filter(data: np.ndarray, sr: int, kind: str, cutoff) -> np.ndarray:
    if cutoff is None:
        return data
    sos = signal.butter(2, cutoff, btype=kind, fs=sr, output="sos")
    return signal.sosfiltfilt(sos, data, axis=0).astype(np.float32)


def soft_knee_compressor(
    data: np.ndarray,
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 2.8,
    attack_ms: float = 8.0,
    release_ms: float = 90.0,
) -> np.ndarray:
    mono = np.mean(np.abs(data), axis=1)
    level_db = 20 * np.log10(np.maximum(mono, 1e-8))
    over_db = np.maximum(level_db - threshold_db, 0.0)
    gain_reduction_db = over_db * (1.0 - 1.0 / max(ratio, 1.0))
    target_gain = db_to_gain(-gain_reduction_db)
    attack = math.exp(-1.0 / max(1.0, attack_ms * sr / 1000.0))
    release = math.exp(-1.0 / max(1.0, release_ms * sr / 1000.0))
    env = np.empty_like(target_gain)
    current = 1.0
    for i, target in enumerate(target_gain):
        coeff = attack if target < current else release
        current = coeff * current + (1.0 - coeff) * target
        env[i] = current
    return (data * env[:, None]).astype(np.float32)


def deess(data: np.ndarray, sr: int, amount: float = 0.18) -> np.ndarray:
    if amount <= 0:
        return data
    high = min(11000.0, sr / 2.0 - 100.0)
    if high <= 5200.0:
        return data
    band = biquad_filter(data, sr, "bandpass", [5200.0, high])
    controlled = soft_knee_compressor(band, sr, threshold_db=-26.0, ratio=5.0, attack_ms=1.5, release_ms=45.0)
    return (data - band * amount + controlled * amount).astype(np.float32)


def simple_room_reverb(data: np.ndarray, sr: int, wet: float = 0.055, decay: float = 0.28) -> np.ndarray:
    if wet <= 0:
        return data
    output = data.copy()
    for i, ms in enumerate([23, 37, 53, 79, 113, 151, 197]):
        delay = int(sr * ms / 1000.0)
        if 0 < delay < len(data):
            output[delay:] += data[:-delay] * wet * (decay**i)
    return limiter(output)


def integrated_lufs(data: np.ndarray, sr: int) -> float | None:
    try:
        import pyloudnorm as pyln

        return float(pyln.Meter(sr).integrated_loudness(data))
    except Exception:
        return None


def normalize_lufs(data: np.ndarray, sr: int, target_lufs: float = -14.0, peak_db: float = -1.0) -> np.ndarray:
    current = integrated_lufs(data, sr)
    if current is None or not np.isfinite(current):
        current = gain_to_db(float(np.sqrt(np.mean(np.square(data))) + 1e-9))
    out = data * db_to_gain(target_lufs - current)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    ceiling = db_to_gain(peak_db)
    if peak > ceiling:
        out *= ceiling / peak
    return out.astype(np.float32)


def limiter(data: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    ceiling = db_to_gain(peak_db)
    return np.clip(data, -ceiling, ceiling).astype(np.float32)
