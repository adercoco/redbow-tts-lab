#!/usr/bin/env python3
"""Prepare ZipVoice TSV/manifests input from the Cosy teacher corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
DEFAULT_CORPUS = BASE / "teacher_cosy_raw_best2_distill_v1"
DEFAULT_OUT = BASE / "datasets" / "zipvoice_cosy_teacher_smoke_v1"
ZIPVOICE_RAW = ROOT / "external" / "ZipVoice" / "egs" / "zipvoice" / "data" / "raw"


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            text = str(row["text"]).replace("\t", " ").strip()
            audio_path = Path(str(row["audio"]).strip())
            if not audio_path.is_absolute():
                audio_path = ROOT / audio_path
            audio = str(audio_path)
            handle.write(f"{row['id']}\t{text}\t{audio}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dev-count", type=int, default=8)
    parser.add_argument("--sync-zipvoice-egs", action="store_true")
    parser.add_argument("--prefix", default="custom-cosy")
    args = parser.parse_args()

    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest if row.get("status") == "ok" and row.get("audio") and Path(row["audio"]).exists()]
    rows.sort(key=lambda row: row["id"])
    if len(rows) <= args.dev_count:
        raise ValueError(f"Not enough rows: {len(rows)} <= dev-count {args.dev_count}")

    dev = rows[-args.dev_count :]
    train = rows[: -args.dev_count]
    train_tsv = args.out_dir / f"{args.prefix}_train.tsv"
    dev_tsv = args.out_dir / f"{args.prefix}_dev.tsv"
    write_tsv(train_tsv, train)
    write_tsv(dev_tsv, dev)

    teacher_pack_id = rows[0].get("teacher_pack_id", "unknown") if rows else "unknown"
    teacher_model = rows[0].get("teacher_model", "unknown") if rows else "unknown"
    summary = {
        "corpus": str(args.corpus),
        "prefix": args.prefix,
        "train_count": len(train),
        "dev_count": len(dev),
        "train_tsv": str(train_tsv),
        "dev_tsv": str(dev_tsv),
        "teacher": f"{teacher_model} {teacher_pack_id}",
        "zipvoice_format": "{uniq_id}\\t{text}\\t{wav_path}",
    }

    if args.sync_zipvoice_egs:
        ZIPVOICE_RAW.mkdir(parents=True, exist_ok=True)
        synced_train = ZIPVOICE_RAW / f"{args.prefix}_train.tsv"
        synced_dev = ZIPVOICE_RAW / f"{args.prefix}_dev.tsv"
        write_tsv(synced_train, train)
        write_tsv(synced_dev, dev)
        summary["synced_train_tsv"] = str(synced_train)
        summary["synced_dev_tsv"] = str(synced_dev)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(train_tsv)
    print(dev_tsv)
    print(f"train={len(train)} dev={len(dev)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
