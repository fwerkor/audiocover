from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal

from .audio import load_audio, match_channels, write_audio


def _mean_spectrum(files: list[Path], sample_rate: int, n_fft: int = 2048) -> np.ndarray:
    specs: list[np.ndarray] = []
    for path in files:
        x, sr = load_audio(path, sr=sample_rate, mono=True)
        mono = x[:, 0]
        if len(mono) < n_fft:
            continue
        freqs, times, stft = signal.stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        mag = np.abs(stft)
        if mag.size:
            specs.append(np.mean(mag, axis=1))
    if not specs:
        return np.ones(n_fft // 2 + 1, dtype=np.float32)
    spec = np.mean(np.stack(specs, axis=0), axis=0)
    spec = spec / (np.mean(spec) + 1e-8)
    return spec.astype(np.float32)


def train_simple_timbre(dataset_wavs: Path, output_path: Path, sample_rate: int = 48000) -> Path:
    files = sorted(dataset_wavs.glob("*.wav"))
    if not files:
        raise RuntimeError(f"no prepared wav files found in {dataset_wavs}")
    spectrum = _mean_spectrum(files, sample_rate)
    profile = {
        "backend": "simple-timbre",
        "sample_rate": sample_rate,
        "n_bins": int(len(spectrum)),
        "mean_spectrum": spectrum.tolist(),
        "note": "Lightweight spectral profile for tests/fallback. Use external RVC/SVC backend for best quality.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def apply_simple_timbre(input_vocal: Path, output_vocal: Path, profile_path: Path) -> Path:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    target = np.asarray(profile["mean_spectrum"], dtype=np.float32)
    x, sr = load_audio(input_vocal, sr=int(profile.get("sample_rate", 48000)), mono=False)
    n_fft = (len(target) - 1) * 2
    out_channels: list[np.ndarray] = []
    for ch in range(x.shape[1]):
        freqs, times, stft = signal.stft(x[:, ch], fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        mag = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        source = np.mean(mag, axis=1)
        source = source / (np.mean(source) + 1e-8)
        curve = target / (source + 1e-4)
        curve = np.clip(curve, 0.35, 2.8)
        curve = signal.savgol_filter(curve, min(31, len(curve) // 2 * 2 - 1), 3) if len(curve) > 33 else curve
        adjusted = mag * curve[:, None]
        _, y = signal.istft(adjusted * phase, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        out_channels.append(y[: len(x)])
    min_len = min(len(ch) for ch in out_channels)
    y = np.stack([ch[:min_len] for ch in out_channels], axis=1).astype(np.float32)
    y = match_channels(y, x.shape[1])
    write_audio(output_vocal, y, sr)
    return output_vocal
