import numpy as np

from audiocover_lab.audio import limiter, match_channels, match_length, normalize_lufs, peak_dbfs


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
