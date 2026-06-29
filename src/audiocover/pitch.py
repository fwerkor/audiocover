from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from .audio import load_audio

DEFAULT_CANDIDATE_TRANSPOSES = (-12, -7, -5, 0, 5, 7, 12)


def _finite_positive(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    return arr[np.isfinite(arr) & (arr > 0)]


def estimate_f0_values(
    audio_path: Path,
    *,
    sample_rate: int = 48000,
    f0_min_hz: float = 60.0,
    f0_max_hz: float = 800.0,
    frame_seconds: float = 0.046,
    hop_seconds: float = 0.010,
) -> np.ndarray:
    """Estimate a compact voiced-frame F0 series using normalized autocorrelation.

    The estimator is intentionally dependency-light. It is not a replacement for
    RMVPE/CREPE inside conversion engines; it is only used for octave-level range
    decisions and training metadata.
    """

    data, sr = load_audio(audio_path, sr=sample_rate, mono=True)
    mono = data[:, 0].astype(np.float32)
    if mono.size == 0:
        return np.asarray([], dtype=np.float32)

    # Remove rumble and very high content before autocorrelation. Keep the bounds
    # conservative so speech-like training clips still produce useful estimates.
    high = min(max(f0_max_hz * 4.0, 1200.0), sr / 2.0 - 100.0)
    if high > 200.0:
        try:
            sos = signal.butter(2, [max(40.0, f0_min_hz * 0.6), high], btype="bandpass", fs=sr, output="sos")
            mono = signal.sosfiltfilt(sos, mono).astype(np.float32)
        except Exception:
            pass

    frame_len = max(512, int(frame_seconds * sr))
    hop_len = max(128, int(hop_seconds * sr))
    if mono.size < frame_len:
        mono = np.pad(mono, (0, frame_len - mono.size))

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak <= 1e-6:
        return np.asarray([], dtype=np.float32)

    min_lag = max(1, int(sr / f0_max_hz))
    max_lag = min(frame_len - 2, int(sr / f0_min_hz))
    if min_lag >= max_lag:
        return np.asarray([], dtype=np.float32)

    window = np.hanning(frame_len).astype(np.float32)
    values: list[float] = []
    rms_floor = max(1e-4, peak * 0.025)
    for start in range(0, mono.size - frame_len + 1, hop_len):
        frame = mono[start : start + frame_len]
        rms = float(np.sqrt(np.mean(np.square(frame))))
        if rms < rms_floor:
            continue
        frame = (frame - float(np.mean(frame))) * window
        energy = float(np.dot(frame, frame))
        if energy <= 1e-8:
            continue
        autocorr = signal.correlate(frame, frame, mode="full", method="fft")[frame_len - 1 :]
        region = autocorr[min_lag : max_lag + 1]
        if region.size == 0:
            continue
        idx = int(np.argmax(region)) + min_lag
        confidence = float(autocorr[idx] / max(autocorr[0], 1e-8))
        if confidence < 0.25:
            continue
        if 1 <= idx < len(autocorr) - 1:
            left = float(autocorr[idx - 1])
            center = float(autocorr[idx])
            right = float(autocorr[idx + 1])
            denom = left - 2.0 * center + right
            if abs(denom) > 1e-8:
                idx = float(idx) + 0.5 * (left - right) / denom
        f0 = sr / float(idx)
        if f0_min_hz <= f0 <= f0_max_hz and math.isfinite(f0):
            values.append(float(f0))
    return np.asarray(values, dtype=np.float32)


def summarize_f0(values: Iterable[float]) -> dict[str, Any]:
    arr = _finite_positive(values)
    if arr.size == 0:
        return {
            "valid": False,
            "voiced_frames": 0,
            "f0_min_hz": None,
            "f0_p10_hz": None,
            "f0_median_hz": None,
            "f0_p90_hz": None,
            "f0_max_hz": None,
            "recommended_target_range_hz": None,
        }
    p10, median, p90 = np.percentile(arr, [10, 50, 90])
    return {
        "valid": True,
        "voiced_frames": int(arr.size),
        "f0_min_hz": round(float(np.min(arr)), 3),
        "f0_p10_hz": round(float(p10), 3),
        "f0_median_hz": round(float(median), 3),
        "f0_p90_hz": round(float(p90), 3),
        "f0_max_hz": round(float(np.max(arr)), 3),
        "recommended_target_range_hz": [round(float(p10 * 0.85), 3), round(float(p90 * 1.15), 3)],
    }


def build_voice_profile(
    dataset_wavs: Path,
    output_path: Path,
    *,
    sample_rate: int = 48000,
    display_name: str | None = None,
) -> dict[str, Any]:
    files = sorted(dataset_wavs.glob("*.wav"))
    all_f0: list[np.ndarray] = []
    per_file: list[dict[str, Any]] = []
    for path in files:
        values = estimate_f0_values(path, sample_rate=sample_rate)
        summary = summarize_f0(values)
        per_file.append({"file": path.name, **summary})
        if values.size:
            all_f0.append(values)
    merged = np.concatenate(all_f0) if all_f0 else np.asarray([], dtype=np.float32)
    profile = {
        "schema_version": 1,
        "display_name": display_name,
        "sample_rate": sample_rate,
        "source_files": len(files),
        **summarize_f0(merged),
        "files": per_file,
        "note": "Used for conservative automatic pitch range adaptation during rendering.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def load_voice_profile(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _range_from_profile(profile: dict[str, Any]) -> tuple[float, float] | None:
    explicit = profile.get("recommended_target_range_hz")
    if isinstance(explicit, (list, tuple)) and len(explicit) == 2:
        low, high = float(explicit[0]), float(explicit[1])
    else:
        p10 = profile.get("f0_p10_hz")
        p90 = profile.get("f0_p90_hz")
        if p10 is None or p90 is None:
            return None
        low, high = float(p10) * 0.85, float(p90) * 1.15
    if not (math.isfinite(low) and math.isfinite(high) and low > 0 and high > low):
        return None
    return low, high


def _transpose_values(values: np.ndarray, semitones: int) -> np.ndarray:
    return values * float(2.0 ** (semitones / 12.0))


def _range_penalty(values: np.ndarray, low: float, high: float) -> float:
    log_values = 12.0 * np.log2(np.maximum(values, 1e-6))
    log_low = 12.0 * math.log2(low)
    log_high = 12.0 * math.log2(high)
    below = np.maximum(log_low - log_values, 0.0)
    above = np.maximum(log_values - log_high, 0.0)
    distance = below + above
    return float(np.mean(np.square(distance)))


def choose_auto_transpose(
    input_f0_values: Iterable[float],
    target_profile: dict[str, Any],
    *,
    candidates: tuple[int, ...] = DEFAULT_CANDIDATE_TRANSPOSES,
) -> dict[str, Any]:
    values = _finite_positive(input_f0_values)
    target_range = _range_from_profile(target_profile)
    target_median = target_profile.get("f0_median_hz")
    if values.size == 0 or target_range is None or not target_profile.get("valid", False):
        return {
            "mode": "auto",
            "selected_transpose": 0,
            "reason": "insufficient_pitch_data",
            "input": summarize_f0(values),
            "target": {k: target_profile.get(k) for k in ("f0_p10_hz", "f0_median_hz", "f0_p90_hz", "recommended_target_range_hz")},
            "candidates": [],
        }

    low, high = target_range
    scored: list[dict[str, Any]] = []
    for semitones in candidates:
        shifted = _transpose_values(values, semitones)
        score = _range_penalty(shifted, low, high)
        if target_median:
            input_median = float(np.median(shifted))
            median_distance = abs(12.0 * math.log2(max(input_median, 1e-6) / float(target_median)))
            score += 0.05 * median_distance
        score += 0.02 * abs(semitones)
        scored.append(
            {
                "transpose": semitones,
                "score": round(float(score), 4),
                "shifted_summary": summarize_f0(shifted),
            }
        )

    scored.sort(key=lambda item: (float(item["score"]), abs(int(item["transpose"]))))
    zero_score = next((float(item["score"]) for item in scored if int(item["transpose"]) == 0), None)
    best = scored[0]
    best_transpose = int(best["transpose"])
    selected = 0
    reason = "already_in_target_range"
    if target_median:
        input_median = float(np.median(values))
        median_gap = 12.0 * math.log2(max(input_median, 1e-6) / float(target_median))
    else:
        median_gap = 0.0
    improvement = (zero_score - float(best["score"])) if zero_score is not None else 0.0
    octave_ambiguous = (
        abs(best_transpose) == 12
        and 8.0 <= abs(median_gap) <= 16.5
        and abs(abs(median_gap) - 12.0) <= 3.0
    )
    if best_transpose == 0:
        reason = "already_in_target_range"
    elif octave_ambiguous:
        reason = "kept_original_due_to_octave_ambiguous_f0"
    elif abs(median_gap) < 5.0:
        reason = "kept_original_small_pitch_gap"
    elif improvement < 6.0:
        reason = "kept_original_marginal_pitch_improvement"
    else:
        selected = best_transpose
        reason = "matched_target_range"

    return {
        "mode": "auto",
        "selected_transpose": selected,
        "reason": reason,
        "input": summarize_f0(values),
        "target": {k: target_profile.get(k) for k in ("f0_p10_hz", "f0_median_hz", "f0_p90_hz", "recommended_target_range_hz")},
        "candidates": scored,
    }
