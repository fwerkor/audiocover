from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .audio import (
    biquad_filter,
    db_to_gain,
    deess,
    limiter,
    load_audio,
    match_channels,
    match_length,
    normalize_lufs,
    simple_room_reverb,
    soft_knee_compressor,
    write_audio,
)
from .config import (
    ConversionConfig,
    MixConfig,
    SeparatorConfig,
    resolve_conversion_config,
    resolve_separator_config,
)
from .external import run_template
from .process import run_hidden
from .runtime import BackendRuntimeManager
from .simple_timbre import apply_simple_timbre


@dataclass(slots=True)
class Stems:
    vocals: Path
    instrumental: Path


@dataclass(slots=True)
class Converted:
    vocal: Path


def separate(input_wav: Path, out_dir: Path, cfg: SeparatorConfig) -> Stems:
    out_dir.mkdir(parents=True, exist_ok=True)
    vocals = out_dir / "vocals.wav"
    instrumental = out_dir / "instrumental.wav"

    if cfg.backend in {"auto", "managed"}:
        manager = BackendRuntimeManager()
        runtime_backend = manager.select_separator_backend()
        if runtime_backend:
            data = cfg.model_dump()
            if data.get("device") == "auto":
                data["device"] = "cpu"
            result = manager.invoke(
                runtime_backend,
                "separate",
                {
                    "input": str(input_wav),
                    "out_dir": str(out_dir),
                    "vocals": str(vocals),
                    "instrumental": str(instrumental),
                    **data,
                },
            )
            return Stems(Path(result["vocals"]), Path(result["instrumental"]))
        if cfg.backend == "managed":
            raise RuntimeError("no bundled separator runtime is available")

    cfg = resolve_separator_config(cfg)

    if cfg.backend == "none":
        vocals.write_bytes(input_wav.read_bytes())
        instrumental.write_bytes(input_wav.read_bytes())
        return Stems(vocals, instrumental)

    if cfg.backend == "external":
        if not cfg.command_template:
            raise ValueError("external separator requires command_template")
        run_template(cfg.command_template, input=input_wav, vocals=vocals, instrumental=instrumental, outdir=out_dir)
        return Stems(vocals, instrumental)

    raw = out_dir / "demucs_raw"
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        cfg.model,
        "--two-stems",
        cfg.two_stems,
        "--shifts",
        str(cfg.shifts),
        "--overlap",
        str(cfg.overlap),
        "-d",
        cfg.device,
        "-o",
        str(raw),
        str(input_wav),
    ]
    if cfg.segment:
        cmd.insert(-1, str(cfg.segment))
        cmd.insert(-2, "--segment")
    cmd.extend(cfg.extra_args)

    process = run_hidden(cmd, text=True, capture_output=True)
    (out_dir / "demucs.log").write_text(
        "COMMAND\n" + " ".join(cmd) + "\n\nSTDOUT\n" + process.stdout + "\n\nSTDERR\n" + process.stderr,
        encoding="utf-8",
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr[-4000:])

    stem_dir = raw / cfg.model / input_wav.stem
    vocals.write_bytes((stem_dir / "vocals.wav").read_bytes())
    instrumental.write_bytes((stem_dir / "no_vocals.wav").read_bytes())
    return Stems(vocals, instrumental)


def convert_vocal(
    vocals: Path,
    out_dir: Path,
    cfg: ConversionConfig,
    workdir: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> Converted:
    cfg = resolve_conversion_config(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "converted_vocal.wav"
    if cfg.backend == "passthrough":
        output.write_bytes(vocals.read_bytes())
    elif cfg.backend == "managed":
        manager = BackendRuntimeManager()
        runtime_backend = manager.select_conversion_backend(cfg.runtime_backend)
        result = manager.invoke(
            runtime_backend,
            "convert",
            {
                "input": str(vocals),
                "output": str(output),
                "model_path": str(cfg.model_path) if cfg.model_path else None,
                "index_path": str(cfg.index_path) if cfg.index_path else None,
                "config_path": str(cfg.config_path) if cfg.config_path else None,
                "cluster_model_path": str(cfg.cluster_model_path) if cfg.cluster_model_path else None,
                "simple_profile_path": str(cfg.simple_profile_path) if cfg.simple_profile_path else None,
                "speaker": cfg.speaker,
                "f0_method": cfg.f0_method,
                "transpose": cfg.transpose,
                "protect": cfg.protect,
                "index_rate": cfg.index_rate,
                "rms_mix_rate": cfg.rms_mix_rate,
                "workdir": str(workdir),
                "extra_args": cfg.extra_args,
            },
            log=log,
        )
        output = Path(result.get("output") or output)
    elif cfg.backend == "simple-timbre":
        if not cfg.simple_profile_path:
            raise ValueError("simple-timbre conversion requires simple_profile_path")
        apply_simple_timbre(vocals, output, cfg.simple_profile_path)
    elif cfg.backend == "external":
        if not cfg.command_template:
            raise ValueError("external conversion requires command_template")
        run_template(
            cfg.command_template,
            log_file=out_dir / "conversion.log",
            input=vocals,
            output=output,
            model=cfg.model_path or "",
            index=cfg.index_path or "",
            f0_method=cfg.f0_method,
            transpose=cfg.transpose,
            protect=cfg.protect,
            index_rate=cfg.index_rate,
            rms_mix_rate=cfg.rms_mix_rate,
            workdir=workdir,
        )
    else:
        raise ValueError(f"unknown conversion backend: {cfg.backend}")

    if not output.exists():
        raise FileNotFoundError(f"converter did not create {output}")
    return Converted(output)


def polish_and_mix(instrumental_path: Path, vocal_path: Path, out_dir: Path, cfg: MixConfig) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    instrumental, sr = load_audio(instrumental_path, sr=cfg.sample_rate, mono=False)
    vocal, _ = load_audio(vocal_path, sr=cfg.sample_rate, mono=False)
    channels = max(instrumental.shape[1], vocal.shape[1])
    instrumental = match_channels(instrumental, channels)
    vocal = match_channels(vocal, channels)
    length = max(len(instrumental), len(vocal))
    instrumental = match_length(instrumental, length)
    vocal = match_length(vocal, length)

    vocal = biquad_filter(vocal, sr, "highpass", cfg.vocal_highpass_hz)
    vocal = biquad_filter(vocal, sr, "lowpass", cfg.vocal_lowpass_hz)
    vocal = deess(vocal, sr, cfg.deess_amount)
    vocal = soft_knee_compressor(
        vocal,
        sr,
        threshold_db=cfg.compressor_threshold_db,
        ratio=cfg.compressor_ratio,
        attack_ms=cfg.compressor_attack_ms,
        release_ms=cfg.compressor_release_ms,
    )
    vocal = simple_room_reverb(vocal, sr, wet=cfg.reverb_wet, decay=cfg.reverb_decay)

    polished = out_dir / "polished_vocal.wav"
    write_audio(polished, limiter(vocal, cfg.final_peak_db), sr)

    mixed = instrumental * db_to_gain(cfg.instrumental_gain_db) + vocal * db_to_gain(cfg.vocal_gain_db)
    mixed = normalize_lufs(mixed, sr, cfg.target_lufs, cfg.final_peak_db)
    mixed = limiter(mixed, cfg.final_peak_db)
    final = out_dir / "final_mix.wav"
    write_audio(final, mixed, sr)
    return polished, final
