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
    f0_method: str = "crepe"
    transpose: int = Field(default=0, ge=-24, le=24)
    protect: float = Field(default=0.33, ge=0.0, le=1.0)
    index_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    rms_mix_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    noise_scale: float = Field(default=0.085, ge=0.0, le=2.0)
    db_thresh: int = -40
    pad_seconds: float = Field(default=0.5, ge=0.0, le=5.0)
    chunk_seconds: float = Field(default=0.5, gt=0.0, le=30.0)
    extra_args: list[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    backend: Literal["auto", "managed", "simple-timbre", "external"] = "auto"
    runtime_backend: str | None = None
    device: str = "auto"
    sample_rate: int = 48000
    segment_seconds: float = Field(default=12.0, ge=2.0, le=30.0)
    epochs: int = Field(default=200, ge=1)
    batch_size: int = Field(default=8, ge=1)
    f0_method: str = "crepe"
    commands: list[str] = Field(default_factory=list)


class MixConfig(BaseModel):
    sample_rate: int = 48000
    instrumental_gain_db: float = 0.0
    vocal_gain_db: float = 0.0
    vocal_highpass_hz: float = 70.0
    vocal_lowpass_hz: float | None = 18000.0
    compressor_threshold_db: float = -14.0
    compressor_ratio: float = 1.25
    compressor_attack_ms: float = 14.0
    compressor_release_ms: float = 180.0
    deess_amount: float = Field(default=0.16, ge=0.0, le=1.0)
    harshness_reduction_amount: float = Field(default=0.12, ge=0.0, le=1.0)
    electronic_artifact_reduction_amount: float = Field(default=0.16, ge=0.0, le=1.0)
    vocal_denoise_amount: float = Field(default=0.18, ge=0.0, le=1.0)
    vocal_denoise_floor: float = Field(default=0.78, ge=0.35, le=1.0)
    vocal_saturation_amount: float = Field(default=0.0, ge=0.0, le=1.0)
    vocal_saturation_drive_db: float = Field(default=0.0, ge=0.0, le=12.0)
    parallel_compression_mix: float = Field(default=0.0, ge=0.0, le=1.0)
    parallel_compression_threshold_db: float = -24.0
    parallel_compression_ratio: float = Field(default=3.4, ge=1.0, le=20.0)
    parallel_compression_makeup_db: float = Field(default=0.0, ge=0.0, le=12.0)
    vocal_body_gain_db: float = Field(default=0.45, ge=-6.0, le=6.0)
    vocal_body_freq_hz: float = Field(default=220.0, ge=40.0, le=800.0)
    vocal_body_q: float = Field(default=0.8, gt=0.0, le=8.0)
    vocal_warmth_gain_db: float = Field(default=0.0, ge=-6.0, le=6.0)
    vocal_warmth_freq_hz: float = Field(default=360.0, ge=80.0, le=1200.0)
    vocal_warmth_q: float = Field(default=0.65, gt=0.0, le=8.0)
    sustain_motion_amount_db: float = Field(default=0.0, ge=0.0, le=3.0)
    sustain_motion_rate_hz: float = Field(default=0.42, ge=0.05, le=3.0)
    reverb_wet: float = Field(default=0.006, ge=0.0, le=1.0)
    reverb_decay: float = Field(default=0.18, ge=0.05, le=2.0)
    reverb_predelay_ms: float = Field(default=8.0, ge=0.0, le=200.0)
    reverb_lowcut_hz: float = Field(default=190.0, ge=20.0, le=1000.0)
    reverb_highcut_hz: float = Field(default=7200.0, ge=1000.0, le=20000.0)
    vocal_doubler_mix: float = Field(default=0.0, ge=0.0, le=0.5)
    vocal_doubler_left_delay_ms: float = Field(default=14.0, ge=1.0, le=80.0)
    vocal_doubler_right_delay_ms: float = Field(default=23.0, ge=1.0, le=80.0)
    chorus_doubler_mix: float = Field(default=0.0, ge=0.0, le=0.35)
    chorus_doubler_delay_ms: float = Field(default=32.0, ge=8.0, le=45.0)
    chorus_doubler_threshold_percentile: float = Field(default=66.0, ge=40.0, le=95.0)
    chorus_doubler_highpass_hz: float = Field(default=130.0, ge=20.0, le=1000.0)
    chorus_doubler_lowpass_hz: float = Field(default=9000.0, ge=1000.0, le=20000.0)
    target_lufs: float = -14.0
    final_peak_db: float = -1.0
    vocal_silence_gate: bool = True
    vocal_gate_threshold_db: float = -46.0
    vocal_gate_relative_db: float = -28.0
    vocal_gate_knee_db: float = Field(default=10.0, gt=0.0)
    vocal_gate_attack_ms: float = Field(default=12.0, ge=1.0)
    vocal_gate_release_ms: float = Field(default=180.0, ge=1.0)
    vocal_gate_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    vocal_tail_cleanup: bool = True
    vocal_tail_gate_threshold_db: float = -48.0
    vocal_tail_gate_relative_db: float = -32.0
    vocal_tail_gate_knee_db: float = Field(default=7.0, gt=0.0)
    vocal_tail_gate_attack_ms: float = Field(default=6.0, ge=1.0)
    vocal_tail_gate_release_ms: float = Field(default=45.0, ge=1.0)
    match_vocal_loudness: bool = True
    match_original_stem_balance: bool = True
    original_stem_balance_gain_limit_db: float = Field(default=5.0, ge=0.0)
    vocal_loudness_offset_db: float = 0.0
    vocal_loudness_gain_limit_db: float = Field(default=10.0, ge=0.0)
    match_vocal_dynamics: bool = True
    vocal_dynamics_strength: float = Field(default=0.94, ge=0.0, le=1.0)
    vocal_dynamics_gain_limit_db: float = Field(default=10.0, ge=0.0)
    vocal_dynamics_attack_ms: float = Field(default=18.0, ge=1.0)
    vocal_dynamics_release_ms: float = Field(default=150.0, ge=1.0)
    match_vocal_macro_dynamics: bool = True
    vocal_macro_dynamics_strength: float = Field(default=0.82, ge=0.0, le=1.0)
    vocal_macro_dynamics_gain_limit_db: float = Field(default=7.0, ge=0.0)
    vocal_macro_dynamics_frame_ms: float = Field(default=700.0, ge=50.0)
    vocal_macro_dynamics_hop_ms: float = Field(default=120.0, ge=10.0)
    vocal_macro_dynamics_attack_ms: float = Field(default=260.0, ge=1.0)
    vocal_macro_dynamics_release_ms: float = Field(default=900.0, ge=1.0)
    sidechain_ducking_db: float = -0.8


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
    f0_method: str = "crepe"
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
        package_locked_keys = {
            "backend",
            "runtime_backend",
            "command_template",
            "model_path",
            "index_path",
            "config_path",
            "cluster_model_path",
            "simple_profile_path",
            "speaker",
        }
        data.update(
            {
                k: v
                for k, v in self.conversion.model_dump().items()
                if v is not None and k in package_locked_keys
            }
        )
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
        data["transpose"] = max(-24, min(24, int(data.get("transpose") or 0) + int(self.transpose or 0)))
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

    In source checkouts this is the project root. In PyInstaller builds, data
    files may live under sys._MEIPASS or beside the frozen executable while
    module __file__ lives under an embedded package path, so resolving from
    __file__ would miss bundled configs on frozen platforms.
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
