#!/usr/bin/env python3
"""Prepare strict-clean Downloads female voice clips as an F5-TTS fine-tune dataset."""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
REF_DIR = BASE / "clean_strict_24k"
DATASET_NAME = "downloads_female_voice_finetune_v1"
F5_ROOT = ROOT / ".venv-f5" / "lib" / "python3.12"


def ensure_vocab(dataset_dir: Path) -> None:
    candidates = [
        F5_ROOT / "data" / "Emilia_ZH_EN_pinyin" / "vocab.txt",
        ROOT / ".venv-f5" / "lib" / "python3.12" / "site-packages" / "f5_tts" / "infer" / "examples" / "vocab.txt",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise RuntimeError("Cannot find F5 vocab.txt")
    for destination in [dataset_dir / "vocab.txt", F5_ROOT / "data" / f"{DATASET_NAME}_pinyin" / "vocab.txt"]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)


def write_metadata(metadata_csv: Path) -> int:
    rows = []
    with (REF_DIR / "metadata.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            transcript = row.get("transcript", "").strip()
            if not transcript:
                continue
            rows.append((str((REF_DIR / row["file"]).resolve()), transcript.replace("|", " ")))
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_csv.write_text(
        "audio_file|text\n" + "\n".join(f"{audio}|{text}" for audio, text in rows) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def main() -> int:
    metadata_csv = BASE / "f5_finetune_v1" / "metadata.csv"
    dataset_dir = F5_ROOT / "data" / f"{DATASET_NAME}_pinyin"
    count = write_metadata(metadata_csv)
    ensure_vocab(dataset_dir)
    prepare_script = F5_ROOT / "site-packages" / "f5_tts" / "train" / "datasets" / "prepare_csv_wavs.py"
    subprocess.run(
        [
            str(ROOT / ".venv-f5" / "bin" / "python"),
            str(prepare_script),
            str(metadata_csv),
            str(dataset_dir),
            "--workers",
            "2",
        ],
        check=True,
    )
    print(f"dataset_name={DATASET_NAME}")
    print(f"metadata={metadata_csv}")
    print(f"prepared={dataset_dir}")
    print(f"items={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
