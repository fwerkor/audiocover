from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .audio import (
    animate_sustains,
    apply_sidechain_ducking,
    biquad_filter,
    db_to_gain,
    deess,
    limiter,
    load_audio,
    match_active_loudness,
    match_channels,
    match_dynamic_envelope,
    match_length,
    normalize_lufs,
    parallel_compress,
    plate_reverb,
    reduce_vocal_harshness,
    soft_knee_compressor,
    soft_saturation,
    vocal_activity_mask,
    vocal_body_eq,
    vocal_doubler,
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


def separate(input_wav: Path, out_dir: Path, cfg: SeparatorConfig, *, log: Callable[[str], None] | None = None) -> Stems:
    out_dir.mkdir(parents=True, exist_ok=True)
    vocals = out_dir / "vocals.wav"
    instrumental = out_dir / "instrumental.wav"

    if cfg.backend in {"auto", "managed"}:
        manager = BackendRuntimeManager()
        runtime_backend = manager.select_separator_backend()
        if runtime_backend:
            if log:
                log(f"selected separator backend: {runtime_backend}")
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
                log=log,
            )
            return Stems(Path(result["vocals"]), Path(result["instrumental"]))
        if cfg.backend == "managed":
            raise RuntimeError("no bundled separator runtime is available")

    cfg = resolve_separator_config(cfg)

    if cfg.backend == "none":
        if log:
            log("separator disabled; using input audio for both stems")
        vocals.write_bytes(input_wav.read_bytes())
        instrumental.write_bytes(input_wav.read_bytes())
        return Stems(vocals, instrumental)

    if cfg.backend == "external":
        if not cfg.command_template:
            raise ValueError("external separator requires command_template")
        if log:
            log("running external separator command")
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
    if log:
        log(f"running demucs separator on {cfg.device}")
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
                "noise_scale": cfg.noise_scale,
                "db_thresh": cfg.db_thresh,
                "pad_seconds": cfg.pad_seconds,
                "chunk_seconds": cfg.chunk_seconds,
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
            noise_scale=cfg.noise_scale,
            db_thresh=cfg.db_thresh,
            pad_seconds=cfg.pad_seconds,
            chunk_seconds=cfg.chunk_seconds,
            workdir=workdir,
        )
    else:
        raise ValueError(f"unknown conversion backend: {cfg.backend}")

    if not output.exists():
        raise FileNotFoundError(f"converter did not create {output}")
    return Converted(output)


def polish_and_mix(
    instrumental_path: Path,
    vocal_path: Path,
    out_dir: Path,
    cfg: MixConfig,
    *,
    reference_vocal_path: Path | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    instrumental, sr = load_audio(instrumental_path, sr=cfg.sample_rate, mono=False)
    vocal, _ = load_audio(vocal_path, sr=cfg.sample_rate, mono=False)
    reference_vocal = None
    if reference_vocal_path is not None and reference_vocal_path.exists():
        reference_vocal, _ = load_audio(reference_vocal_path, sr=cfg.sample_rate, mono=False)

    channels = max(instrumental.shape[1], vocal.shape[1], reference_vocal.shape[1] if reference_vocal is not None else 1)
    instrumental = match_channels(instrumental, channels)
    vocal = match_channels(vocal, channels)
    if reference_vocal is not None:
        reference_vocal = match_channels(reference_vocal, channels)
    length = max(len(instrumental), len(vocal), len(reference_vocal) if reference_vocal is not None else 0)
    instrumental = match_length(instrumental, length)
    vocal = match_length(vocal, length)
    if reference_vocal is not None:
        reference_vocal = match_length(reference_vocal, length)

    activity_mask = None
    if cfg.vocal_silence_gate and reference_vocal is not None:
        activity_mask = vocal_activity_mask(
            reference_vocal,
            sr,
            threshold_db=cfg.vocal_gate_threshold_db,
            relative_threshold_db=cfg.vocal_gate_relative_db,
            knee_db=cfg.vocal_gate_knee_db,
            attack_ms=cfg.vocal_gate_attack_ms,
            release_ms=cfg.vocal_gate_release_ms,
            floor=cfg.vocal_gate_floor,
        )
        activity_mask = match_length(activity_mask, length)
        vocal = (vocal * activity_mask).astype("float32")

    vocal = biquad_filter(vocal, sr, "highpass", cfg.vocal_highpass_hz)
    vocal = biquad_filter(vocal, sr, "lowpass", cfg.vocal_lowpass_hz)
    if cfg.match_vocal_loudness and reference_vocal is not None:
        vocal, _gain_db = match_active_loudness(
            vocal,
            reference_vocal,
            mask=activity_mask,
            target_offset_db=cfg.vocal_loudness_offset_db,
            max_gain_db=cfg.vocal_loudness_gain_limit_db,
        )
    vocal = deess(vocal, sr, cfg.deess_amount)
    vocal = reduce_vocal_harshness(vocal, sr, amount=cfg.harshness_reduction_amount)
    vocal = soft_knee_compressor(
        vocal,
        sr,
        threshold_db=cfg.compressor_threshold_db,
        ratio=cfg.compressor_ratio,
        attack_ms=cfg.compressor_attack_ms,
        release_ms=cfg.compressor_release_ms,
    )
    if cfg.match_vocal_dynamics and reference_vocal is not None:
        vocal = match_dynamic_envelope(
            vocal,
            reference_vocal,
            sr,
            mask=activity_mask,
            strength=cfg.vocal_dynamics_strength,
            max_gain_db=cfg.vocal_dynamics_gain_limit_db,
            attack_ms=cfg.vocal_dynamics_attack_ms,
            release_ms=cfg.vocal_dynamics_release_ms,
        )
    if cfg.match_vocal_macro_dynamics and reference_vocal is not None:
        vocal = match_dynamic_envelope(
            vocal,
            reference_vocal,
            sr,
            mask=activity_mask,
            strength=cfg.vocal_macro_dynamics_strength,
            max_gain_db=cfg.vocal_macro_dynamics_gain_limit_db,
            frame_ms=cfg.vocal_macro_dynamics_frame_ms,
            hop_ms=cfg.vocal_macro_dynamics_hop_ms,
            attack_ms=cfg.vocal_macro_dynamics_attack_ms,
            release_ms=cfg.vocal_macro_dynamics_release_ms,
        )
    vocal = animate_sustains(
        vocal,
        sr,
        reference=reference_vocal,
        mask=activity_mask,
        amount_db=cfg.sustain_motion_amount_db,
        rate_hz=cfg.sustain_motion_rate_hz,
    )
    vocal = soft_saturation(vocal, amount=cfg.vocal_saturation_amount, drive_db=cfg.vocal_saturation_drive_db)
    vocal = parallel_compress(
        vocal,
        sr,
        mix=cfg.parallel_compression_mix,
        threshold_db=cfg.parallel_compression_threshold_db,
        ratio=cfg.parallel_compression_ratio,
        makeup_db=cfg.parallel_compression_makeup_db,
    )
    vocal = vocal_body_eq(
        vocal,
        sr,
        gain_db=cfg.vocal_body_gain_db,
        freq_hz=cfg.vocal_body_freq_hz,
        q=cfg.vocal_body_q,
    )
    vocal = vocal_body_eq(
        vocal,
        sr,
        gain_db=cfg.vocal_warmth_gain_db,
        freq_hz=cfg.vocal_warmth_freq_hz,
        q=cfg.vocal_warmth_q,
    )
    vocal = plate_reverb(
        vocal,
        sr,
        wet=cfg.reverb_wet,
        decay=cfg.reverb_decay,
        predelay_ms=cfg.reverb_predelay_ms,
        lowcut_hz=cfg.reverb_lowcut_hz,
        highcut_hz=cfg.reverb_highcut_hz,
    )
    vocal = vocal_doubler(
        vocal,
        sr,
        mix=cfg.vocal_doubler_mix,
        left_delay_ms=cfg.vocal_doubler_left_delay_ms,
        right_delay_ms=cfg.vocal_doubler_right_delay_ms,
    )

    polished = out_dir / "polished_vocal.wav"
    write_audio(polished, limiter(vocal, cfg.final_peak_db), sr)

    mix_instrumental = instrumental
    if activity_mask is not None and cfg.sidechain_ducking_db < 0:
        mix_instrumental = apply_sidechain_ducking(instrumental, activity_mask, cfg.sidechain_ducking_db)
    mixed = mix_instrumental * db_to_gain(cfg.instrumental_gain_db) + vocal * db_to_gain(cfg.vocal_gain_db)
    mixed = normalize_lufs(mixed, sr, cfg.target_lufs, cfg.final_peak_db)
    mixed = limiter(mixed, cfg.final_peak_db)
    final = out_dir / "final_mix.wav"
    write_audio(final, mixed, sr)
    return polished, final
