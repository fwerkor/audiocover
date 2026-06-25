from pathlib import Path

from audiocover_lab.config import ModelPackage, RenderConfig, TrainingConfig


def test_config_loads() -> None:
    cfg = RenderConfig.from_yaml(Path("configs/high_quality.yaml"))
    assert cfg.separator.model == "htdemucs_ft"
    assert cfg.mix.sample_rate == 48000


def test_model_package_roundtrip(tmp_path: Path) -> None:
    package = ModelPackage(display_name="x", training=TrainingConfig(), simple_profile_path=Path("simple_timbre.json"))
    path = tmp_path / "model.yaml"
    package.write_yaml(path)
    loaded = ModelPackage.from_yaml(path)
    assert loaded.display_name == "x"
    assert loaded.simple_profile_path == tmp_path / "simple_timbre.json"
