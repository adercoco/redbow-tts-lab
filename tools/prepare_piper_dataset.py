#!/usr/bin/env python3
"""Prepare an LJSpeech-style dataset from a Qwen3 teacher corpus.

The output is intentionally plain: ``metadata.csv`` plus ``wavs/*.wav``.
That keeps it usable by Piper/VITS training recipes and simple enough to
audit before training a mobile-sized student.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
DEFAULT_CORPUS = BASE / "teacher_qwen3_1p7b_distill_v1"
DEFAULT_OUT = BASE / "datasets" / "piper_ljspeech_distill_v1"
DEFAULT_SAMPLE_RATE = 22050


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def write_wav(src: Path, dst: Path, sample_rate: int) -> float:
    audio, _ = librosa.load(str(src), sr=sample_rate, mono=True)
    sf.write(str(dst), audio, sample_rate, subtype="PCM_16")
    return float(len(audio) / sample_rate)


def clean_text(text: str) -> str:
    return " ".join(text.replace("|", " ").split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CORPUS / "manifest.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N smoke dataset.")
    args = parser.parse_args()

    manifest = resolve_path(args.manifest)
    out = resolve_path(args.out)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if args.limit > 0:
        rows = rows[: args.limit]

    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    metadata_lines = []
    dataset_rows = []
    total_seconds = 0.0

    for row in rows:
        if row["status"] != "ok" or not row.get("audio"):
            continue
        item_id = row["id"]
        src = resolve_path(row["audio"])
        if not src.exists():
            raise FileNotFoundError(src)
        dst = wav_dir / f"{item_id}.wav"
        seconds = write_wav(src, dst, args.sample_rate)
        text = clean_text(row["text"])
        metadata_lines.append(f"{item_id}|{text}")
        total_seconds += seconds
        dataset_rows.append(
            {
                "id": item_id,
                "text": text,
                "wav": str(dst.relative_to(out)),
                "source_audio": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
                "seconds": seconds,
                "sample_rate": args.sample_rate,
            }
        )

    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.csv").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    (out / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "source_manifest": str(manifest.relative_to(ROOT)) if manifest.is_relative_to(ROOT) else str(manifest),
                "items": len(dataset_rows),
                "sample_rate": args.sample_rate,
                "total_seconds": total_seconds,
                "total_hours": total_seconds / 3600,
                "rows": dataset_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# Piper/VITS LJSpeech Dataset\n\n"
        "Synthetic Qwen3 teacher data for the `taiwan_mandarin_low_r` voice.\n\n"
        "Format: `metadata.csv` plus `wavs/*.wav`.\n\n"
        f"- Items: {len(dataset_rows)}\n"
        f"- Audio: {total_seconds:.2f}s / {total_seconds / 3600:.3f}h\n"
        f"- Sample rate: {args.sample_rate} Hz mono PCM16\n\n"
        "This dataset is intended for small single-speaker student experiments. "
        "The v1 corpus is pipeline-quality; final quality should use a broader natural corpus.\n",
        encoding="utf-8",
    )
    print(out)
    print(f"items={len(metadata_lines)}")
    print(f"seconds={total_seconds:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
