from pathlib import Path

import yaml

from audiocover.config import ModelPackage, RenderConfig, TrainingConfig, default_config_path


def test_config_loads() -> None:
    cfg = RenderConfig.from_yaml(Path("configs/high_quality.yaml"))
    assert cfg.separator.model == "htdemucs_ft"
    assert cfg.mix.sample_rate == 48000


def test_training_config_defaults_match_so_vits_preset() -> None:
    data = yaml.safe_load(Path("configs/training_simple.yaml").read_text(encoding="utf-8"))
    cfg = TrainingConfig.model_validate(data)

    assert cfg.sample_rate == 44100
    assert cfg.f0_method == "crepe-tiny"


def test_default_config_path_uses_pyinstaller_meipass(monkeypatch, tmp_path: Path) -> None:
    import audiocover.config as config_module

    root = tmp_path / "_MEI12345"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "high_quality.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(config_module.sys, "_MEIPASS", str(root), raising=False)

    assert default_config_path() == root / "configs" / "high_quality.yaml"


def test_model_package_roundtrip(tmp_path: Path) -> None:
    package = ModelPackage(display_name="x", training=TrainingConfig(), simple_profile_path=Path("simple_timbre.json"))
    path = tmp_path / "model.yaml"
    package.write_yaml(path)
    loaded = ModelPackage.from_yaml(path)
    assert loaded.display_name == "x"
    assert loaded.simple_profile_path == tmp_path / "simple_timbre.json"



def test_render_config_f0_wins_over_legacy_model_package_default() -> None:
    base = RenderConfig().conversion.model_copy(update={"f0_method": "crepe-tiny"})
    package = ModelPackage(
        display_name="legacy",
        conversion={"backend": "managed", "runtime_backend": "so-vits-svc", "f0_method": "harvest"},
        f0_method="harvest",
    )

    merged = package.merged_conversion(base)

    assert merged.backend == "managed"
    assert merged.runtime_backend == "so-vits-svc"
    assert merged.f0_method == "crepe-tiny"


def test_high_quality_defaults_include_expression_controls() -> None:
    cfg = RenderConfig.from_yaml(Path("configs/high_quality.yaml"))

    assert cfg.conversion.noise_scale == 0.035
    assert cfg.mix.match_vocal_dynamics is True
    assert cfg.mix.match_vocal_macro_dynamics is True
    assert cfg.mix.vocal_macro_dynamics_strength > 0
    assert cfg.mix.vocal_dynamics_strength >= 0.85
    assert cfg.mix.compressor_ratio < 2.0
    assert cfg.mix.harshness_reduction_amount >= 0.34
    assert cfg.mix.vocal_saturation_amount <= 0.10
    assert cfg.mix.parallel_compression_mix <= 0.12
    assert cfg.mix.vocal_body_gain_db >= 1.1
    assert cfg.mix.vocal_warmth_gain_db > 0
    assert cfg.mix.sustain_motion_amount_db > 0
    assert cfg.mix.reverb_predelay_ms > 0
    assert cfg.mix.vocal_doubler_mix > 0
