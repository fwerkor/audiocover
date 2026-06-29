from pathlib import Path

import numpy as np
import soundfile as sf

from audiocover.pitch import build_voice_profile, choose_auto_transpose, estimate_f0_values


def _write_tone(path: Path, freq: float, seconds: float = 2.0, sr: int = 48000) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    x = (0.08 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, sr)


def test_pitch_estimator_tracks_simple_tone(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    _write_tone(wav, 220.0)
    values = estimate_f0_values(wav, sample_rate=48000)
    assert values.size > 50
    assert 215.0 <= float(np.median(values)) <= 225.0


def test_auto_transpose_keeps_source_key_for_moderate_gap() -> None:
    target = {
        "valid": True,
        "f0_p10_hz": 95.0,
        "f0_median_hz": 130.0,
        "f0_p90_hz": 180.0,
        "recommended_target_range_hz": [80.0, 210.0],
    }
    selection = choose_auto_transpose([240.0, 260.0, 280.0, 300.0], target)
    assert selection["selected_transpose"] == 0
    assert selection["reason"] == "kept_original_moderate_pitch_gap"


def test_build_voice_profile_writes_summary(tmp_path: Path) -> None:
    wavs = tmp_path / "wavs"
    _write_tone(wavs / "a.wav", 160.0)
    _write_tone(wavs / "b.wav", 200.0)
    profile = build_voice_profile(wavs, tmp_path / "voice_profile.json", sample_rate=48000, display_name="x")
    assert profile["valid"] is True
    assert profile["source_files"] == 2
    assert (tmp_path / "voice_profile.json").exists()


def test_auto_transpose_caps_very_large_gap_to_partial_shift() -> None:
    target = {
        "valid": True,
        "f0_p10_hz": 95.0,
        "f0_median_hz": 130.0,
        "f0_p90_hz": 180.0,
        "recommended_target_range_hz": [80.0, 210.0],
    }
    selection = choose_auto_transpose([430.0, 450.0, 470.0, 490.0], target)
    assert selection["selected_transpose"] == -7
    assert selection["reason"] == "partially_reduced_large_pitch_gap"


def test_auto_transpose_keeps_source_key_for_small_gap() -> None:
    target = {
        "valid": True,
        "f0_p10_hz": 95.0,
        "f0_median_hz": 130.0,
        "f0_p90_hz": 180.0,
        "recommended_target_range_hz": [80.0, 210.0],
    }
    selection = choose_auto_transpose([145.0, 150.0, 155.0, 160.0], target)
    assert selection["selected_transpose"] == 0
