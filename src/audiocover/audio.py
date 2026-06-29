from __future__ import annotations

import math
import shutil
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .process import run_hidden


def db_to_gain(db):
    return np.power(10.0, np.asarray(db) / 20.0)


def gain_to_db(gain: float) -> float:
    return 20.0 * math.log10(max(float(gain), 1e-12))


def peak_dbfs(data: np.ndarray) -> float:
    return gain_to_db(float(np.max(np.abs(data))) if data.size else 0.0)


def run_command(args: list[str], *, log_file: Path | None = None) -> None:
    process = run_hidden(args, text=True, capture_output=True)
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


def ffmpeg_executable() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        bundled = None
    if bundled and Path(bundled).is_file():
        return str(bundled)
    return None


def require_ffmpeg() -> str:
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg was not found. Reinstall the AudioCover package or install ffmpeg and make sure it is in PATH."
        )
    return ffmpeg


def convert_to_wav(src: Path, dst: Path, sample_rate: int = 48000, channels: int = 2) -> Path:
    if src.suffix.lower() in {".wav", ".flac", ".ogg"}:
        try:
            data, sr = load_audio(src, sr=sample_rate, mono=channels == 1)
            data = match_channels(data, channels)
            return write_audio(dst, data, sr, subtype="PCM_24")
        except Exception:
            pass

    ffmpeg = require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg,
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


def _mono_rms_envelope(data: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    mono = data.astype(np.float32) if data.ndim == 1 else np.mean(data, axis=1).astype(np.float32)
    if mono.size == 0:
        return np.zeros(0, dtype=np.float32)
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if len(mono) <= frame:
        rms = np.array([float(np.sqrt(np.mean(np.square(mono)) + 1e-12))], dtype=np.float32)
        return np.full(len(mono), rms[0], dtype=np.float32)

    values: list[float] = []
    centers: list[float] = []
    for start in range(0, len(mono), hop):
        end = min(len(mono), start + frame)
        chunk = mono[start:end]
        values.append(float(np.sqrt(np.mean(np.square(chunk)) + 1e-12)))
        centers.append((start + end - 1) / 2.0)
        if end == len(mono):
            break
    return np.interp(np.arange(len(mono)), np.asarray(centers), np.asarray(values)).astype(np.float32)


def smooth_envelope(
    target: np.ndarray,
    sr: int,
    *,
    attack_ms: float = 20.0,
    release_ms: float = 160.0,
) -> np.ndarray:
    attack = math.exp(-1.0 / max(1.0, attack_ms * sr / 1000.0))
    release = math.exp(-1.0 / max(1.0, release_ms * sr / 1000.0))
    env = np.empty_like(target, dtype=np.float32)
    current = 0.0
    for i, value in enumerate(np.asarray(target, dtype=np.float32)):
        coeff = attack if value > current else release
        current = coeff * current + (1.0 - coeff) * float(value)
        env[i] = current
    return env


def vocal_activity_mask(
    reference: np.ndarray,
    sr: int,
    *,
    threshold_db: float = -46.0,
    relative_threshold_db: float = -28.0,
    knee_db: float = 10.0,
    attack_ms: float = 12.0,
    release_ms: float = 180.0,
    floor: float = 0.0,
) -> np.ndarray:
    """Return a full-length 0..1 activity mask from a source vocal stem.

    The threshold combines an absolute noise floor with a relative threshold based on
    the louder parts of the source vocal. This suppresses converter hallucinations
    during lyric-free or separation-bleed sections while preserving quiet sung notes.
    """
    envelope = _mono_rms_envelope(reference, sr)
    if envelope.size == 0:
        return np.zeros((0, 1), dtype=np.float32)
    voiced = envelope[envelope > db_to_gain(-80.0)]
    if voiced.size:
        ref_level = float(np.percentile(voiced, 75))
        threshold = max(db_to_gain(threshold_db), ref_level * db_to_gain(relative_threshold_db))
    else:
        threshold = db_to_gain(threshold_db)
    level_db = 20.0 * np.log10(np.maximum(envelope, 1e-8))
    threshold_db_actual = gain_to_db(threshold)
    knee = max(float(knee_db), 1e-3)
    target = np.clip((level_db - threshold_db_actual) / knee, 0.0, 1.0).astype(np.float32)
    smoothed = smooth_envelope(target, sr, attack_ms=attack_ms, release_ms=release_ms)
    if floor > 0:
        smoothed = floor + (1.0 - floor) * smoothed
    return np.clip(smoothed, 0.0, 1.0)[:, None].astype(np.float32)


def _active_rms(data: np.ndarray, mask: np.ndarray | None = None) -> float:
    if data.size == 0:
        return 0.0
    mono = np.mean(data, axis=1) if data.ndim > 1 else data
    if mask is None or mask.size == 0:
        return float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
    weights = np.asarray(mask, dtype=np.float32)
    if weights.ndim > 1:
        weights = weights[:, 0]
    weights = match_length(weights[:, None], len(mono))[:, 0]
    active = weights > 0.15
    if np.any(active):
        mono = mono[active]
        weights = weights[active]
    weighted_power = np.sum(np.square(mono) * np.maximum(weights, 1e-3)) / max(float(np.sum(np.maximum(weights, 1e-3))), 1e-6)
    return float(np.sqrt(weighted_power + 1e-12))


def match_active_loudness(
    data: np.ndarray,
    reference: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    target_offset_db: float = -1.5,
    max_gain_db: float = 8.0,
) -> tuple[np.ndarray, float]:
    source_rms = _active_rms(reference, mask)
    data_rms = _active_rms(data, mask)
    if source_rms <= 1e-7 or data_rms <= 1e-7:
        return data.astype(np.float32), 0.0
    desired_db = gain_to_db(source_rms / data_rms) + target_offset_db
    applied_db = max(-max_gain_db, min(max_gain_db, desired_db))
    return (data * db_to_gain(applied_db)).astype(np.float32), float(applied_db)


def apply_sidechain_ducking(
    instrumental: np.ndarray,
    activity_mask: np.ndarray,
    duck_db: float = -1.8,
) -> np.ndarray:
    if duck_db >= 0 or activity_mask.size == 0:
        return instrumental.astype(np.float32)
    mask = match_length(activity_mask, len(instrumental))
    duck_gain = db_to_gain(duck_db)
    gain = 1.0 + (duck_gain - 1.0) * np.clip(mask, 0.0, 1.0)
    return (instrumental * gain).astype(np.float32)


def match_dynamic_envelope(
    data: np.ndarray,
    reference: np.ndarray,
    sr: int,
    *,
    mask: np.ndarray | None = None,
    strength: float = 0.45,
    max_gain_db: float = 5.0,
    frame_ms: float = 80.0,
    hop_ms: float = 20.0,
    attack_ms: float = 35.0,
    release_ms: float = 220.0,
) -> np.ndarray:
    if strength <= 0 or data.size == 0 or reference.size == 0:
        return data.astype(np.float32)
    ref_env = _mono_rms_envelope(reference, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    data_env = _mono_rms_envelope(data, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if ref_env.size == 0 or data_env.size == 0:
        return data.astype(np.float32)
    data_env = match_length(data_env[:, None], len(data))[:, 0]
    ref_env = match_length(ref_env[:, None], len(data))[:, 0]
    weights = np.ones(len(data), dtype=np.float32)
    if mask is not None and mask.size:
        weights = match_length(mask, len(data))[:, 0].astype(np.float32)
    active = weights > 0.15
    if not np.any(active):
        return data.astype(np.float32)
    ref_db = 20.0 * np.log10(np.maximum(ref_env, 1e-8))
    data_db = 20.0 * np.log10(np.maximum(data_env, 1e-8))
    ref_mid = float(np.median(ref_db[active]))
    data_mid = float(np.median(data_db[active]))
    target_delta = (ref_db - ref_mid) - (data_db - data_mid)
    gain_db = np.clip(target_delta * float(strength), -max_gain_db, max_gain_db)
    gain_db *= np.clip(weights, 0.0, 1.0)
    smoothed = smooth_envelope(gain_db.astype(np.float32), sr, attack_ms=attack_ms, release_ms=release_ms)
    gain = db_to_gain(smoothed)
    return (data * gain[:, None]).astype(np.float32)


def reduce_vocal_harshness(
    data: np.ndarray,
    sr: int,
    *,
    amount: float = 0.16,
    low_hz: float = 2600.0,
    high_hz: float = 7600.0,
) -> np.ndarray:
    if amount <= 0 or data.size == 0:
        return data.astype(np.float32)
    high = min(float(high_hz), sr / 2.0 - 100.0)
    low = min(float(low_hz), high - 100.0)
    if high <= low:
        return data.astype(np.float32)
    band = biquad_filter(data, sr, "bandpass", [low, high])
    controlled = soft_knee_compressor(band, sr, threshold_db=-30.0, ratio=4.5, attack_ms=2.0, release_ms=70.0)
    return (data - band * amount + controlled * amount).astype(np.float32)


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


def soft_saturation(
    data: np.ndarray,
    *,
    amount: float = 0.12,
    drive_db: float = 3.0,
) -> np.ndarray:
    if amount <= 0 or data.size == 0:
        return data.astype(np.float32)
    drive = float(db_to_gain(drive_db))
    driven = data * drive
    ceiling = np.tanh(drive)
    if abs(ceiling) < 1e-6:
        return data.astype(np.float32)
    saturated = np.tanh(driven) / ceiling
    return (data * (1.0 - amount) + saturated * amount).astype(np.float32)


def parallel_compress(
    data: np.ndarray,
    sr: int,
    *,
    mix: float = 0.16,
    threshold_db: float = -24.0,
    ratio: float = 4.0,
    attack_ms: float = 6.0,
    release_ms: float = 140.0,
    makeup_db: float = 1.8,
) -> np.ndarray:
    if mix <= 0 or data.size == 0:
        return data.astype(np.float32)
    compressed = soft_knee_compressor(
        data,
        sr,
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
    )
    compressed = compressed * db_to_gain(makeup_db)
    return (data * (1.0 - mix) + compressed * mix).astype(np.float32)


def _peaking_eq_sos(sr: int, freq_hz: float, gain_db: float, q: float) -> np.ndarray:
    freq = min(max(float(freq_hz), 20.0), sr / 2.0 - 100.0)
    q = max(float(q), 0.05)
    a = float(db_to_gain(gain_db / 2.0))
    omega = 2.0 * math.pi * freq / float(sr)
    alpha = math.sin(omega) / (2.0 * q)
    cos_w = math.cos(omega)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w
    a2 = 1.0 - alpha / a
    return np.asarray([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def vocal_body_eq(
    data: np.ndarray,
    sr: int,
    *,
    gain_db: float = 1.1,
    freq_hz: float = 220.0,
    q: float = 0.8,
) -> np.ndarray:
    if abs(gain_db) < 1e-6 or data.size == 0:
        return data.astype(np.float32)
    sos = _peaking_eq_sos(sr, freq_hz, gain_db, q)
    return signal.sosfiltfilt(sos, data, axis=0).astype(np.float32)


def plate_reverb(
    data: np.ndarray,
    sr: int,
    *,
    wet: float = 0.10,
    decay: float = 0.82,
    predelay_ms: float = 32.0,
    lowcut_hz: float = 190.0,
    highcut_hz: float = 8500.0,
) -> np.ndarray:
    if wet <= 0 or data.size == 0:
        return data.astype(np.float32)
    taps_ms = [0, 17, 31, 47, 67, 89, 113, 149, 191, 239, 293]
    wet_signal = np.zeros_like(data, dtype=np.float32)
    for i, ms in enumerate(taps_ms):
        delay = int(sr * (predelay_ms + ms) / 1000.0)
        if 0 < delay < len(data):
            gain = (decay ** i) / math.sqrt(i + 1.0)
            wet_signal[delay:] += data[:-delay] * gain
    if lowcut_hz > 0:
        wet_signal = biquad_filter(wet_signal, sr, "highpass", lowcut_hz)
    high = min(float(highcut_hz), sr / 2.0 - 100.0)
    if high > 500.0:
        wet_signal = biquad_filter(wet_signal, sr, "lowpass", high)
    return limiter(data + wet_signal * wet, -0.5)


def vocal_doubler(
    data: np.ndarray,
    sr: int,
    *,
    mix: float = 0.055,
    left_delay_ms: float = 14.0,
    right_delay_ms: float = 23.0,
) -> np.ndarray:
    if mix <= 0 or data.size == 0:
        return data.astype(np.float32)
    channels = data.shape[1] if data.ndim > 1 else 1
    src = data if data.ndim > 1 else data[:, None]
    doubled = np.zeros_like(src, dtype=np.float32)
    delays = [left_delay_ms, right_delay_ms]
    for ch in range(channels):
        delay_ms = delays[ch % 2]
        delay = int(sr * delay_ms / 1000.0)
        if 0 < delay < len(src):
            doubled[delay:, ch] = src[:-delay, ch]
    if channels == 1:
        doubled = doubled * 0.5
    return (src + doubled * mix).astype(np.float32)


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
