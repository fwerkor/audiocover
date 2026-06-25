from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .audio import convert_to_wav
from .config import ModelPackage, RenderConfig
from .qc import analyze_audio
from .stages import convert_vocal, polish_and_mix, separate


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def render_cover(
    input_song: Path,
    model_package_path: Path,
    output_dir: Path,
    *,
    config: RenderConfig,
    consent: bool,
) -> dict:
    if not consent:
        raise PermissionError("rendering requires explicit rights/consent confirmation")
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    package = ModelPackage.from_yaml(model_package_path)
    normalized = convert_to_wav(input_song, output_dir / "input" / "input.wav", config.mix.sample_rate)
    stems = separate(normalized, output_dir / "stems", config.separator)
    conversion_cfg = package.merged_conversion(config.conversion)
    converted = convert_vocal(stems.vocals, output_dir / "converted", conversion_cfg, output_dir)
    polished, final = polish_and_mix(stems.instrumental, converted.vocal, output_dir / "mix", config.mix)

    qc = {
        "input": analyze_audio(normalized, config.qc),
        "vocals": analyze_audio(stems.vocals, config.qc),
        "instrumental": analyze_audio(stems.instrumental, config.qc),
        "converted_vocal": analyze_audio(converted.vocal, config.qc),
        "polished_vocal": analyze_audio(polished, config.qc),
        "final_mix": analyze_audio(final, config.qc),
    }
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    qc_path = reports / "qc.json"
    qc_path.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_song": str(input_song),
        "input_sha256": sha256_file(input_song),
        "model_package": str(model_package_path),
        "model": package.model_dump(mode="json"),
        "config": config.model_dump(mode="json"),
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
