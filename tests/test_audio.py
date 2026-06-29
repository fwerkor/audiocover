import sys
import types

import numpy as np

from audiocover import audio
from audiocover.audio import (
    apply_sidechain_ducking,
    limiter,
    match_active_loudness,
    match_channels,
    match_length,
    normalize_lufs,
    peak_dbfs,
    vocal_activity_mask,
)


def test_match_channels_and_length() -> None:
    x = np.ones((10, 1), dtype=np.float32)
    y = match_channels(x, 2)
    assert y.shape == (10, 2)
    z = match_length(y, 15)
    assert z.shape == (15, 2)


def test_limiter() -> None:
    x = np.array([[2.0, -2.0]], dtype=np.float32)
    y = limiter(x, -1.0)
    assert np.max(np.abs(y)) < 1.0


def test_normalize_lufs_finite() -> None:
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = (0.1 * np.sin(2 * np.pi * 440 * t))[:, None].astype(np.float32)
    y = normalize_lufs(x, sr, -16.0)
    assert np.all(np.isfinite(y))
    assert peak_dbfs(y) <= -0.9


def test_require_ffmpeg_uses_imageio_fallback(monkeypatch, tmp_path) -> None:
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text("", encoding="utf-8")
    monkeypatch.setattr(audio.shutil, "which", lambda _name: None)
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        types.SimpleNamespace(get_ffmpeg_exe=lambda: str(fake_ffmpeg)),
    )

    assert audio.require_ffmpeg() == str(fake_ffmpeg)



def test_vocal_activity_mask_suppresses_silent_sections() -> None:
    sr = 1000
    silent = np.zeros((sr, 1), dtype=np.float32)
    voiced = np.ones((sr, 1), dtype=np.float32) * 0.08
    ref = np.concatenate([silent, voiced, silent], axis=0)
    mask = vocal_activity_mask(ref, sr, threshold_db=-50, relative_threshold_db=-30, attack_ms=5, release_ms=30)

    assert mask.shape == ref.shape
    assert float(np.mean(mask[: sr // 2])) < 0.05
    assert float(np.mean(mask[sr : 2 * sr])) > 0.80
    assert float(np.mean(mask[2 * sr + sr // 2 :])) < 0.10


def test_match_active_loudness_clamps_gain() -> None:
    ref = np.ones((1000, 1), dtype=np.float32) * 0.10
    quiet = np.ones((1000, 1), dtype=np.float32) * 0.001
    mask = np.ones((1000, 1), dtype=np.float32)

    adjusted, gain_db = match_active_loudness(quiet, ref, mask=mask, target_offset_db=0.0, max_gain_db=6.0)

    assert 5.9 < gain_db <= 6.0
    assert float(np.sqrt(np.mean(np.square(adjusted)))) > float(np.sqrt(np.mean(np.square(quiet))))


def test_sidechain_ducking_reduces_active_regions_only() -> None:
    instrumental = np.ones((1000, 2), dtype=np.float32) * 0.5
    mask = np.zeros((1000, 1), dtype=np.float32)
    mask[250:750] = 1.0

    ducked = apply_sidechain_ducking(instrumental, mask, -6.0)

    assert np.allclose(ducked[:200], instrumental[:200])
    assert float(np.mean(ducked[300:700])) < float(np.mean(instrumental[300:700]))
