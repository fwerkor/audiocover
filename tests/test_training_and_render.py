from pathlib import Path

import numpy as np
import soundfile as sf

from audiocover.config import RenderConfig, TrainingConfig
from audiocover.pipeline import render_cover
from audiocover.training import train_model


def _write_tone(path: Path, freq: float, seconds: float = 2.5, sr: int = 48000) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    x = (0.08 * np.sin(2 * np.pi * freq * t))[:, None].astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, sr)


def test_train_simple_and_render_debug(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_tone(raw / "a.wav", 220)
    _write_tone(raw / "b.wav", 330)

    model_dir = tmp_path / "model"
    package = train_model(
        raw,
        model_dir,
        display_name="test",
        config=TrainingConfig(backend="simple-timbre", segment_seconds=2.0),
        consent=True,
    )
    assert (model_dir / "model.yaml").exists()
    assert (model_dir / "voice_profile.json").exists()
    assert package.voice_profile_path == model_dir / "voice_profile.json"
    assert package.conversion.backend == "simple-timbre"

    song = tmp_path / "song.wav"
    stereo = np.concatenate([
        np.ones((48000 * 3, 1), dtype=np.float32) * 0.01,
        np.ones((48000 * 3, 1), dtype=np.float32) * 0.02,
    ], axis=1)
    sf.write(song, stereo, 48000)

    cfg = RenderConfig.from_yaml(Path("configs/cpu_debug.yaml"))
    manifest = render_cover(song, model_dir / "model.yaml", tmp_path / "run", config=cfg, consent=True)
    assert Path(manifest["outputs"]["final_mix"]).exists()
    assert Path(manifest["outputs"]["converted_vocal"]).exists()
    assert manifest["pitch_adaptation"]["reason"] == "conversion_backend_does_not_support_transpose"
    assert (tmp_path / "run" / "reports" / "auto_pitch.json").exists()


def test_consent_required(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_tone(raw / "a.wav", 220)
    try:
        train_model(raw, tmp_path / "model", display_name="x", config=TrainingConfig(), consent=False)
    except PermissionError:
        pass
    else:
        raise AssertionError("expected PermissionError")
