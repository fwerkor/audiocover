from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import convert_to_wav, load_audio, write_audio
from .qc import analyze_audio

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}


def discover_audio_files(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    segment_seconds: float = 12.0,
    sample_rate: int = 48000,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wavs"
    converted_dir = output_dir / "_converted"
    wav_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    rejected: list[dict] = []
    segment_len = int(segment_seconds * sample_rate)

    for src in discover_audio_files(input_dir):
        try:
            temp = converted_dir / f"{src.stem}.wav"
            convert_to_wav(src, temp, sample_rate, channels=1)
            data, sr = load_audio(temp, sr=sample_rate, mono=True)
            if len(data) < sr * 2:
                rejected.append({"source": str(src), "reason": "too_short"})
                continue

            n_segments = max(1, int(np.ceil(len(data) / segment_len)))
            for i in range(n_segments):
                chunk = data[i * segment_len : (i + 1) * segment_len]
                if len(chunk) < sr * 2:
                    continue
                if float(np.max(np.abs(chunk))) > 0.995:
                    rejected.append({"source": str(src), "segment": i, "reason": "clipping"})
                    continue
                dst = wav_dir / f"{src.stem}_{i:04d}.wav"
                write_audio(dst, chunk, sr)
                qc = analyze_audio(dst)
                if qc["silence_ratio"] > 0.70:
                    dst.unlink(missing_ok=True)
                    rejected.append({"source": str(src), "segment": i, "reason": "mostly_silence"})
                    continue
                items.append(
                    {
                        "source": str(src),
                        "output": str(dst),
                        "duration_seconds": qc["duration_seconds"],
                        "peak_dbfs": qc["peak_dbfs"],
                        "warnings": qc["warnings"],
                    }
                )
        except Exception as exc:
            rejected.append({"source": str(src), "reason": repr(exc)})

    report = {"items": items, "rejected": rejected}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + ("\n" if items else ""),
        encoding="utf-8",
    )
    return report
