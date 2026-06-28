from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class SeparatorConfig(BaseModel):
    backend: Literal["auto", "managed", "demucs", "external", "none"] = "auto"
    model: str = "htdemucs_ft"
    device: str = "auto"
    shifts: int = Field(default=8, ge=0, le=32)
    overlap: float = Field(default=0.5, ge=0.0, le=0.99)
    two_stems: str = "vocals"
    segment: int | None = None
    command_template: str | None = None
    extra_args: list[str] = Field(default_factory=list)


class ConversionConfig(BaseModel):
    backend: Literal["auto", "managed", "external", "passthrough", "simple-timbre"] = "auto"
    pitch_shift_mode: Literal["auto", "manual"] = "auto"
    runtime_backend: str | None = None
    command_template: str | None = None
    model_path: Path | None = None
    index_path: Path | None = None
    config_path: Path | None = None
    cluster_model_path: Path | None = None
    simple_profile_path: Path | None = None
    speaker: str | None = None
    f0_method: str = "rmvpe"
    transpose: int = Field(default=0, ge=-24, le=24)
    protect: float = Field(default=0.33, ge=0.0, le=1.0)
    index_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    rms_mix_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    extra_args: list[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    backend: Literal["auto", "managed", "simple-timbre", "external"] = "auto"
    runtime_backend: str | None = None
    device: str = "auto"
    sample_rate: int = 48000
    segment_seconds: float = Field(default=12.0, ge=2.0, le=30.0)
    epochs: int = Field(default=200, ge=1)
    batch_size: int = Field(default=8, ge=1)
    f0_method: str = "rmvpe"
    commands: list[str] = Field(default_factory=list)


class MixConfig(BaseModel):
    sample_rate: int = 48000
    instrumental_gain_db: float = -1.5
    vocal_gain_db: float = 0.0
    vocal_highpass_hz: float = 70.0
    vocal_lowpass_hz: float | None = 18000.0
    compressor_threshold_db: float = -18.0
    compressor_ratio: float = 2.8
    compressor_attack_ms: float = 8.0
    compressor_release_ms: float = 90.0
    deess_amount: float = Field(default=0.18, ge=0.0, le=1.0)
    reverb_wet: float = Field(default=0.055, ge=0.0, le=1.0)
    reverb_decay: float = Field(default=0.28, ge=0.05, le=2.0)
    target_lufs: float = -14.0
    final_peak_db: float = -1.0


class QcConfig(BaseModel):
    max_clipping_ratio: float = 0.0005
    min_duration_seconds: float = 5.0
    max_silence_ratio: float = 0.65
    warn_lufs_above: float = -8.0
    warn_lufs_below: float = -24.0


class RenderConfig(BaseModel):
    separator: SeparatorConfig = Field(default_factory=SeparatorConfig)
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    mix: MixConfig = Field(default_factory=MixConfig)
    qc: QcConfig = Field(default_factory=QcConfig)
    keep_intermediates: bool = True
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> RenderConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


class ModelPackage(BaseModel):
    display_name: str
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    conversion: ConversionConfig = Field(default_factory=ConversionConfig)
    runtime_backend: str | None = None
    model_path: Path | None = None
    index_path: Path | None = None
    config_path: Path | None = None
    cluster_model_path: Path | None = None
    simple_profile_path: Path | None = None
    voice_profile_path: Path | None = None
    speaker: str | None = None
    transpose: int = 0
    f0_method: str = "rmvpe"
    created_by: str = "audiocover"
    notes: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> ModelPackage:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base_dir = path.parent
        obj = cls.model_validate(data)
        return obj.resolve_relative_paths(base_dir)

    def resolve_relative_paths(self, base_dir: Path) -> ModelPackage:
        data = self.model_dump()
        for key in (
            "model_path",
            "index_path",
            "config_path",
            "cluster_model_path",
            "simple_profile_path",
            "voice_profile_path",
        ):
            value = getattr(self, key)
            if value is not None and not value.is_absolute():
                data[key] = base_dir / value
        conv = data.get("conversion") or {}
        for key in ("model_path", "index_path", "config_path", "cluster_model_path", "simple_profile_path"):
            value = conv.get(key)
            if value is not None:
                p = Path(value)
                if not p.is_absolute():
                    conv[key] = base_dir / p
        data["conversion"] = conv
        return ModelPackage.model_validate(data)

    def merged_conversion(self, base: ConversionConfig) -> ConversionConfig:
        data: dict[str, Any] = base.model_dump()
        data.update({k: v for k, v in self.conversion.model_dump().items() if v is not None})
        if self.model_path is not None:
            data["model_path"] = self.model_path
        if self.index_path is not None:
            data["index_path"] = self.index_path
        if self.config_path is not None:
            data["config_path"] = self.config_path
        if self.cluster_model_path is not None:
            data["cluster_model_path"] = self.cluster_model_path
        if self.simple_profile_path is not None:
            data["simple_profile_path"] = self.simple_profile_path
        if self.runtime_backend is not None:
            data["runtime_backend"] = self.runtime_backend
        if self.speaker is not None:
            data["speaker"] = self.speaker
        data["transpose"] = self.transpose
        data["f0_method"] = self.f0_method
        return ConversionConfig.model_validate(data)

    def write_yaml(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path


def application_root() -> Path:
    """Return the root containing bundled data files such as configs/.

    In source checkouts this is the project root. In PyInstaller one-file
    builds, data files are extracted under sys._MEIPASS while module __file__
    lives under an embedded package path, so resolving from __file__ would miss
    bundled configs on Windows and other frozen platforms.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return application_root() / "configs" / "high_quality.yaml"


def default_training_config_path() -> Path:
    return application_root() / "configs" / "training_simple.yaml"



def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def best_available_torch_device() -> str:
    if os.environ.get("AUDIOCOVER_BINARY_CPU_ONLY", "").lower() in {"1", "true", "yes"}:
        return "cpu"
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def resolve_separator_config(cfg: SeparatorConfig) -> SeparatorConfig:
    if cfg.backend != "auto":
        if cfg.backend == "demucs" and cfg.device == "auto":
            data = cfg.model_dump()
            data["device"] = best_available_torch_device()
            return SeparatorConfig.model_validate(data)
        return cfg

    data = cfg.model_dump()
    if _module_available("demucs"):
        data["backend"] = "demucs"
        data["device"] = best_available_torch_device()
    elif cfg.command_template:
        data["backend"] = "external"
    else:
        data["backend"] = "none"
        data["device"] = "cpu"
    return SeparatorConfig.model_validate(data)


def resolve_conversion_config(cfg: ConversionConfig) -> ConversionConfig:
    if cfg.backend != "auto":
        return cfg

    data = cfg.model_dump()
    if cfg.runtime_backend:
        data["backend"] = "managed"
    elif cfg.command_template:
        data["backend"] = "external"
    elif cfg.simple_profile_path:
        data["backend"] = "simple-timbre"
    elif cfg.model_path or cfg.index_path:
        data["backend"] = "external"
    else:
        data["backend"] = "passthrough"
    return ConversionConfig.model_validate(data)


def resolve_training_config(cfg: TrainingConfig) -> TrainingConfig:
    if cfg.backend != "auto":
        return cfg
    data = cfg.model_dump()
    data["backend"] = "external" if cfg.commands else "managed"
    return TrainingConfig.model_validate(data)
