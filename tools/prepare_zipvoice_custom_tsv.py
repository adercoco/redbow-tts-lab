#!/usr/bin/env python3
"""Create ZipVoice custom fine-tune TSV files from prosody-filtered teacher data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASE = Path("/Users/ader/Documents/App/distillation/taiwan_mandarin_low_r")
PHASE_DIR = BASE / "datasets" / "prosody_filtered_teacher_v1"
OUT_DIR = BASE / "datasets" / "zipvoice_custom_finetune_v1"
ZIPVOICE_EGS_RAW = Path("/Users/ader/Documents/App/external/ZipVoice/egs/zipvoice/data/raw")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            text = str(row["text"]).replace("\t", " ").strip()
            audio = str(row["audio"]).strip()
            handle.write(f'{row["id"]}\t{text}\t{audio}\n')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-dir", type=Path, default=PHASE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dev-count", type=int, default=32)
    parser.add_argument("--sync-zipvoice-egs", action="store_true")
    args = parser.parse_args()

    phase1 = read_jsonl(args.phase_dir / "phase1_flat_core.jsonl")
    phase2 = read_jsonl(args.phase_dir / "phase2_natural_tail.jsonl")

    dev_phase1 = phase1[-args.dev_count // 2 :]
    dev_phase2 = phase2[: args.dev_count - len(dev_phase1)]
    dev_ids = {row["id"] for row in dev_phase1 + dev_phase2}
    train = [row for row in phase1 if row["id"] not in dev_ids]
    dev = dev_phase1 + dev_phase2

    train_tsv = args.out_dir / "custom_train.tsv"
    dev_tsv = args.out_dir / "custom_dev.tsv"
    write_tsv(train_tsv, train)
    write_tsv(dev_tsv, dev)

    summary = {
        "source_phase_dir": str(args.phase_dir),
        "train_count": len(train),
        "dev_count": len(dev),
        "train_tsv": str(train_tsv),
        "dev_tsv": str(dev_tsv),
        "zipvoice_expected_format": "{uniq_id}\\t{text}\\t{wav_path}",
        "notes": [
            "Train split uses Phase 1 flat core only, with the tail held out for dev.",
            "Dev split mixes late Phase 1 and early Phase 2 to catch over-flattening.",
            "This prepares ZipVoice fine-tuning input; it does not train the model yet.",
        ],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.sync_zipvoice_egs:
        ZIPVOICE_EGS_RAW.mkdir(parents=True, exist_ok=True)
        write_tsv(ZIPVOICE_EGS_RAW / "custom_train.tsv", train)
        write_tsv(ZIPVOICE_EGS_RAW / "custom_dev.tsv", dev)
        summary["synced_zipvoice_egs_raw"] = str(ZIPVOICE_EGS_RAW)
        (args.out_dir / "manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(train_tsv)
    print(dev_tsv)
    print(f"train={len(train)} dev={len(dev)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
