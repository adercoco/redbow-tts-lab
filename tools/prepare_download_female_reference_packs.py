#!/usr/bin/env python3
"""Create multi-utterance reference packs from strict-clean female voice clips."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
REF_DIR = BASE / "clean_strict_24k"
OUT = BASE / "reference_packs_v1"

PACKS = [
    {
        "pack_id": "pack_best2_7s",
        "label": "best2 7.6s",
        "refs": ["ref_01", "ref_02"],
        "note": "短、乾淨、聲紋初篩高；適合 F5 / CosyVoice / IndexTTS2 單 prompt。",
    },
    {
        "pack_id": "pack_best3_11s",
        "label": "best3 11s",
        "refs": ["ref_01", "ref_02", "ref_06"],
        "note": "多一段語氣資訊；接近 F5 推薦上限，可能更像也可能拖慢。",
    },
    {
        "pack_id": "pack_gpt_aux",
        "label": "GPT aux refs",
        "refs": ["ref_01", "ref_02", "ref_04", "ref_06", "ref_07", "ref_09"],
        "note": "全部落在 GPT-SoVITS 3-10 秒規則內，用於 auxiliary references，不串接。",
    },
]


def load_rows() -> list[dict[str, str]]:
    rows = []
    with (REF_DIR / "metadata.csv").open(encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle), 1):
            row = dict(row)
            row["ref_id"] = f"ref_{index:02d}"
            row["path"] = str(REF_DIR / row["file"])
            rows.append(row)
    return rows


def concatenate(rows: list[dict[str, str]], output: Path) -> None:
    chunks = []
    sample_rate = None
    for row in rows:
        audio, sr = sf.read(row["path"], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate is None:
            sample_rate = sr
        if sr != sample_rate:
            raise RuntimeError(f"sample rate mismatch: {row['path']} {sr} != {sample_rate}")
        chunks.append(audio)
        chunks.append(np.zeros(int(sample_rate * 0.18), dtype=np.float32))
    merged = np.concatenate(chunks[:-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, merged, sample_rate)


def main() -> int:
    all_rows = load_rows()
    by_id = {row["ref_id"]: row for row in all_rows}
    manifest = []
    OUT.mkdir(parents=True, exist_ok=True)
    for pack in PACKS:
        rows = [by_id[ref_id] for ref_id in pack["refs"]]
        text = " ".join(row["transcript"].strip() for row in rows if row.get("transcript"))
        item = {
            **pack,
            "ref_files": [row["file"] for row in rows],
            "ref_audio_paths": [row["path"] for row in rows],
            "ref_text": text,
            "total_duration": round(sum(float(row["duration"]) for row in rows), 3),
        }
        if pack["pack_id"] != "pack_gpt_aux":
            output = OUT / f"{pack['pack_id']}.wav"
            concatenate(rows, output)
            item["pack_audio"] = str(output)
        manifest.append(item)
    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
