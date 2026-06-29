import sys
import types

import numpy as np

from audiocover import audio
from audiocover.audio import (
    animate_sustains,
    apply_sidechain_ducking,
    chorus_vocal_doubler,
    limiter,
    match_active_loudness,
    match_channels,
    match_dynamic_envelope,
    match_length,
    match_original_stem_balance,
    normalize_lufs,
    parallel_compress,
    peak_dbfs,
    plate_reverb,
    reduce_electronic_artifacts,
    reduce_vocal_noise,
    soft_saturation,
    suppress_vocal_tails,
    vocal_activity_mask,
    vocal_body_eq,
    vocal_doubler,
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


def test_match_dynamic_envelope_restores_section_contrast() -> None:
    sr = 1000
    quiet_ref = np.ones((sr, 1), dtype=np.float32) * 0.02
    loud_ref = np.ones((sr, 1), dtype=np.float32) * 0.12
    reference = np.concatenate([quiet_ref, loud_ref], axis=0)
    flat = np.ones_like(reference) * 0.06
    mask = np.ones_like(reference)

    shaped = match_dynamic_envelope(flat, reference, sr, mask=mask, strength=1.0, max_gain_db=12.0, attack_ms=5, release_ms=20)

    quiet_rms = float(np.sqrt(np.mean(np.square(shaped[:sr]))))
    loud_rms = float(np.sqrt(np.mean(np.square(shaped[sr:]))))
    assert loud_rms > quiet_rms * 1.8


def test_professional_vocal_polish_helpers_keep_audio_safe() -> None:
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    base = (0.08 * np.sin(2 * np.pi * 180 * t) + 0.02 * np.sin(2 * np.pi * 2200 * t))[:, None].astype(np.float32)

    polished = soft_saturation(base, amount=0.12, drive_db=3.0)
    polished = parallel_compress(polished, sr, mix=0.18)
    polished = vocal_body_eq(polished, sr, gain_db=1.15)
    polished = plate_reverb(polished, sr, wet=0.08, decay=0.7)
    polished = vocal_doubler(polished, sr, mix=0.05)

    assert polished.shape == base.shape
    assert np.all(np.isfinite(polished))
    assert peak_dbfs(polished) <= 0.0
    assert not np.allclose(polished, base)


def test_animate_sustains_adds_subtle_motion_to_loud_sections() -> None:
    sr = 1000
    x = np.ones((sr * 3, 1), dtype=np.float32) * 0.08
    ref = x.copy()
    mask = np.ones_like(x)

    animated = animate_sustains(x, sr, reference=ref, mask=mask, amount_db=1.0, rate_hz=0.5)

    assert animated.shape == x.shape
    assert np.all(np.isfinite(animated))
    assert float(np.std(animated[:, 0])) > 0.0005
    assert peak_dbfs(animated) <= -18.0


def test_chorus_vocal_doubler_targets_loud_sections() -> None:
    sr = 1000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    tone = np.sin(2 * np.pi * 120 * t).astype(np.float32)[:, None]
    envelope = np.concatenate([
        np.ones((sr, 1), dtype=np.float32) * 0.03,
        np.ones((sr, 1), dtype=np.float32) * 0.12,
        np.ones((sr, 1), dtype=np.float32) * 0.03,
    ])
    data = tone * envelope
    ref = data.copy()
    mask = np.ones_like(data)

    doubled = chorus_vocal_doubler(
        data,
        sr,
        reference=ref,
        mask=mask,
        mix=0.20,
        delay_ms=20.0,
        threshold_percentile=60.0,
        highpass_hz=20.0,
        lowpass_hz=300.0,
    )

    assert doubled.shape == data.shape
    assert np.all(np.isfinite(doubled))
    assert float(np.mean(np.abs(doubled[sr + 100 : 2 * sr] - data[sr + 100 : 2 * sr]))) > 1e-4
    assert np.allclose(doubled[: sr // 2], data[: sr // 2], atol=1e-4)


def test_match_original_stem_balance_restores_vocal_instrumental_ratio() -> None:
    ref_vocal = np.ones((1000, 1), dtype=np.float32) * 0.10
    ref_inst = np.ones((1000, 1), dtype=np.float32) * 0.05
    converted_vocal = np.ones((1000, 1), dtype=np.float32) * 0.02
    inst = np.ones((1000, 1), dtype=np.float32) * 0.05
    mask = np.ones((1000, 1), dtype=np.float32)

    matched, gain_db = match_original_stem_balance(converted_vocal, inst, ref_vocal, ref_inst, mask=mask, max_gain_db=16.0)

    assert gain_db > 5.0
    ratio = float(np.sqrt(np.mean(np.square(matched))) / np.sqrt(np.mean(np.square(inst))))
    assert 1.8 < ratio < 2.2


def test_suppress_vocal_tails_fades_inactive_regions() -> None:
    sr = 1000
    data = np.ones((sr * 3, 1), dtype=np.float32) * 0.08
    mask = np.zeros_like(data)
    mask[:sr] = 1.0

    cleaned = suppress_vocal_tails(data, mask, sr)

    assert float(np.mean(np.abs(cleaned[: sr // 2]))) > 0.05
    assert float(np.mean(np.abs(cleaned[2 * sr :]))) < 0.005


def test_reduce_electronic_artifacts_keeps_signal_safe() -> None:
    sr = 48000
    t = np.linspace(0, 1, sr, endpoint=False)
    base = (0.05 * np.sin(2 * np.pi * 220 * t) + 0.015 * np.sin(2 * np.pi * 7200 * t))[:, None].astype(np.float32)

    cleaned = reduce_electronic_artifacts(base, sr, amount=0.25)

    assert cleaned.shape == base.shape
    assert np.all(np.isfinite(cleaned))
    assert peak_dbfs(cleaned) <= 0.0
    assert not np.allclose(cleaned, base)


def test_reduce_vocal_noise_is_safe_and_active() -> None:
    sr = 48000
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, sr, endpoint=False)
    voiced = 0.045 * np.sin(2 * np.pi * 220 * t)
    noise = rng.normal(0.0, 0.004, size=sr)
    data = (voiced + noise).astype(np.float32)[:, None]

    cleaned = reduce_vocal_noise(data, sr, amount=0.25, floor=0.72)

    assert cleaned.shape == data.shape
    assert np.all(np.isfinite(cleaned))
    assert peak_dbfs(cleaned) <= 0.0
    assert not np.allclose(cleaned, data)
