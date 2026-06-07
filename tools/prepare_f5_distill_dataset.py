#!/usr/bin/env python3
"""Prepare a Qwen3 teacher distillation corpus for F5-TTS training."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "distillation" / "taiwan_mandarin_low_r" / "teacher_qwen3_1p7b_distill_v1"
DEFAULT_DATASET_NAME = "taiwan_low_r_distill_v1"
DEFAULT_F5_ROOT = ROOT / ".venv-f5" / "lib" / "python3.12"


def f5_data_root(f5_root: Path) -> Path:
    return f5_root / "data"


def ensure_f5_vocab(dataset_dir: Path, f5_root: Path) -> None:
    """Copy the pretrained F5 vocab so fine-tuning keeps the base tokenizer."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    base_vocab = f5_root / "data" / "Emilia_ZH_EN_pinyin" / "vocab.txt"
    destinations = [dataset_dir / "vocab.txt", base_vocab]
    if all(dst.exists() for dst in destinations):
        return

    candidates = [
        base_vocab,
        ROOT / ".venv-f5" / "lib" / "python3.12" / "site-packages" / "f5_tts" / "infer" / "examples" / "vocab.txt",
    ]
    source = None
    for candidate in candidates:
        if candidate.exists():
            source = candidate
            break

    if source is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to fetch F5TTS_v1_Base/vocab.txt") from exc
        source = Path(hf_hub_download("SWivid/F5-TTS", filename="F5TTS_v1_Base/vocab.txt"))

    for dst in destinations:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(source, dst)


def write_metadata(corpus_dir: Path, metadata_csv: Path, limit: int | None) -> int:
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest if row.get("status") == "ok" and row.get("audio")]
    if limit is not None:
        rows = rows[:limit]
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["audio_file|text"]
    for row in rows:
        audio = Path(row["audio"]).resolve()
        text = str(row["text"]).replace("|", " ").strip()
        lines.append(f"{audio.as_posix()}|{text}")
    metadata_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--f5-root", type=Path, default=DEFAULT_F5_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dataset_dir = f5_data_root(args.f5_root) / f"{args.dataset_name}_pinyin"
    metadata_csv = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / args.dataset_name / "metadata.csv"
    count = write_metadata(args.corpus, metadata_csv, args.limit)
    ensure_f5_vocab(dataset_dir, args.f5_root)

    prepare_script = ROOT / ".venv-f5" / "lib" / "python3.12" / "site-packages" / "f5_tts" / "train" / "datasets" / "prepare_csv_wavs.py"
    subprocess.run(
        [
            str(ROOT / ".venv-f5" / "bin" / "python"),
            str(prepare_script),
            str(metadata_csv),
            str(dataset_dir),
            "--workers",
            str(args.workers),
        ],
        check=True,
    )
    print(f"dataset_name={args.dataset_name}")
    print(f"metadata={metadata_csv}")
    print(f"prepared={dataset_dir}")
    print(f"items={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
