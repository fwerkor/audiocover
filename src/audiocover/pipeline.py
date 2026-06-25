from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from .audio import convert_to_wav
from .config import ConversionConfig, ModelPackage, RenderConfig, resolve_conversion_config
from .pitch import choose_auto_transpose, estimate_f0_values, load_voice_profile
from .qc import analyze_audio
from .stages import convert_vocal, polish_and_mix, separate


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _supports_pitch_shift(cfg: ConversionConfig) -> bool:
    return cfg.backend == "external" or (cfg.backend == "managed" and cfg.runtime_backend != "simple-timbre")


def _hz_text(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    return "unknown"


def apply_auto_pitch_adaptation(
    vocals: Path,
    package: ModelPackage,
    cfg: ConversionConfig,
    reports_dir: Path,
    *,
    sample_rate: int,
    log: Callable[[str], None] | None = None,
) -> tuple[ConversionConfig, dict]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    base_report: dict = {
        "mode": cfg.pitch_shift_mode,
        "configured_transpose": cfg.transpose,
        "selected_transpose": 0,
        "effective_transpose": cfg.transpose,
    }

    if cfg.pitch_shift_mode == "manual":
        report = {**base_report, "reason": "manual_pitch_shift_mode"}
    elif package.voice_profile_path is None:
        report = {**base_report, "reason": "missing_voice_profile"}
    elif not _supports_pitch_shift(cfg):
        report = {
            **base_report,
            "reason": "conversion_backend_does_not_support_transpose",
            "backend": cfg.backend,
            "runtime_backend": cfg.runtime_backend,
        }
    else:
        target_profile = load_voice_profile(package.voice_profile_path)
        input_values = estimate_f0_values(vocals, sample_rate=sample_rate)
        selection = choose_auto_transpose(input_values, target_profile or {})
        selected = int(selection.get("selected_transpose") or 0)
        effective = max(-24, min(24, cfg.transpose + selected))
        report = {
            **base_report,
            **selection,
            "configured_transpose": cfg.transpose,
            "selected_transpose": selected,
            "effective_transpose": effective,
            "voice_profile_path": str(package.voice_profile_path),
        }
        if effective != cfg.transpose + selected:
            report["reason"] = "clamped_to_supported_transpose_range"
        if log:
            input_summary = selection.get("input") if isinstance(selection.get("input"), dict) else {}
            target = selection.get("target") if isinstance(selection.get("target"), dict) else {}
            log(f"input vocal median f0: {_hz_text(input_summary.get('f0_median_hz'))} Hz")
            log(f"model median f0: {_hz_text(target.get('f0_median_hz'))} Hz")
            log(f"selected pitch shift: {selected:+d} semitone(s); effective transpose: {effective:+d}")
        cfg = cfg.model_copy(update={"transpose": effective})

    (reports_dir / "auto_pitch.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return cfg, report


def render_cover(
    input_song: Path,
    model_package_path: Path,
    output_dir: Path,
    *,
    config: RenderConfig,
    consent: bool,
    log: Callable[[str], None] | None = None,
) -> dict:
    if not consent:
        raise PermissionError("rendering requires explicit rights/consent confirmation")
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    package = ModelPackage.from_yaml(model_package_path)
    normalized = convert_to_wav(input_song, output_dir / "input" / "input.wav", config.mix.sample_rate)
    stems = separate(normalized, output_dir / "stems", config.separator)
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    conversion_cfg = resolve_conversion_config(package.merged_conversion(config.conversion))
    conversion_cfg, pitch_report = apply_auto_pitch_adaptation(
        stems.vocals,
        package,
        conversion_cfg,
        reports,
        sample_rate=config.mix.sample_rate,
        log=log,
    )
    converted = convert_vocal(stems.vocals, output_dir / "converted", conversion_cfg, output_dir, log=log)
    polished, final = polish_and_mix(stems.instrumental, converted.vocal, output_dir / "mix", config.mix)

    qc = {
        "input": analyze_audio(normalized, config.qc),
        "vocals": analyze_audio(stems.vocals, config.qc),
        "instrumental": analyze_audio(stems.instrumental, config.qc),
        "converted_vocal": analyze_audio(converted.vocal, config.qc),
        "polished_vocal": analyze_audio(polished, config.qc),
        "final_mix": analyze_audio(final, config.qc),
    }
    qc_path = reports / "qc.json"
    qc_path.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_song": str(input_song),
        "input_sha256": sha256_file(input_song),
        "model_package": str(model_package_path),
        "model": package.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
        "pitch_adaptation": pitch_report,
        "outputs": {
            "normalized_input": str(normalized),
            "vocals": str(stems.vocals),
            "instrumental": str(stems.instrumental),
            "converted_vocal": str(converted.vocal),
            "polished_vocal": str(polished),
            "final_mix": str(final),
            "qc": str(qc_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
