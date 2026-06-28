from pathlib import Path

from audiocover.config import ModelPackage, RenderConfig, TrainingConfig, default_config_path


def test_config_loads() -> None:
    cfg = RenderConfig.from_yaml(Path("configs/high_quality.yaml"))
    assert cfg.separator.model == "htdemucs_ft"
    assert cfg.mix.sample_rate == 48000


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
