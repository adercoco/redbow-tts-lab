#!/usr/bin/env python3
"""Prepare a tiny CosyVoice speaker fine-tune dataset from authorized clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch


TRANSCRIPT_FIXES = {
    "haibara_001": "是啊，当你来到博士家跟我们集合之前，大家先带着一起。",
    "haibara_002": "受不了。我把感冒药找出来给你吃吧。",
    "haibara_003": "我只是为了以防不时之需，才把药打开。",
    "haibara_004": "不要，我才不要把药交给一变回原样就只顾着谈恋爱什么都不管的侦探。",
    "haibara_005": "那个人的脚怎么了吗？",
    "haibara_006": "女汤这边人也好多喔。",
}


def utt_id_from_clip(path: str) -> str:
    stem = Path(path).stem
    return "_".join(stem.split("_")[:2])


def write_kaldi_dir(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_scp = []
    text = []
    utt2spk = []
    for row in rows:
        wav_scp.append(f"{row['utt']} {row['wav']}\n")
        text.append(f"{row['utt']} {row['text']}\n")
        utt2spk.append(f"{row['utt']} haibara_authorized\n")
    (out_dir / "wav.scp").write_text("".join(wav_scp), encoding="utf-8")
    (out_dir / "text").write_text("".join(text), encoding="utf-8")
    (out_dir / "utt2spk").write_text("".join(utt2spk), encoding="utf-8")
    (out_dir / "spk2utt").write_text(
        "haibara_authorized " + " ".join(row["utt"] for row in rows) + "\n",
        encoding="utf-8",
    )


def make_parquet(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "parquet_000000000.tar"
    split_data_dir = out_dir.parent.parent / "data" / out_dir.name
    utt2embedding = None
    spk2embedding = None
    utt2speech_token = None
    if (split_data_dir / "utt2embedding.pt").exists():
        utt2embedding = torch.load(split_data_dir / "utt2embedding.pt", map_location="cpu")
    if (split_data_dir / "spk2embedding.pt").exists():
        spk2embedding = torch.load(split_data_dir / "spk2embedding.pt", map_location="cpu")
    if (split_data_dir / "utt2speech_token.pt").exists():
        utt2speech_token = torch.load(split_data_dir / "utt2speech_token.pt", map_location="cpu")

    data = {
        "utt": [row["utt"] for row in rows],
        "audio_data": [Path(row["wav"]).read_bytes() for row in rows],
        "wav": [row["wav"] for row in rows],
        "text": [row["text"] for row in rows],
        "spk": ["haibara_authorized" for _ in rows],
    }
    if utt2embedding is not None:
        data["utt_embedding"] = [utt2embedding[row["utt"]] for row in rows]
    if spk2embedding is not None:
        data["spk_embedding"] = [spk2embedding["haibara_authorized"] for _ in rows]
    if utt2speech_token is not None:
        data["speech_token"] = [utt2speech_token[row["utt"]] for row in rows]

    frame = pd.DataFrame(data)
    frame.to_parquet(parquet)
    utt2parquet = {row["utt"]: str(parquet) for row in rows}
    spk2parquet = {"haibara_authorized": str(parquet)}
    (out_dir / "utt2parquet_000000000.json").write_text(
        json.dumps(utt2parquet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "spk2parquet_000000000.json").write_text(
        json.dumps(spk2parquet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "data.list").write_text(str(parquet) + "\n", encoding="utf-8")
    (out_dir / "utt2data.list").write_text(
        str(out_dir / "utt2parquet_000000000.json") + "\n", encoding="utf-8"
    )
    (out_dir / "spk2data.list").write_text(
        str(out_dir / "spk2parquet_000000000.json") + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = Path(args.out)
    all_rows = []
    for item in manifest:
        if item["role"] != "haibara":
            continue
        utt = utt_id_from_clip(item["clip"])
        samples, sr = sf.read(item["clip"])
        duration = len(samples) / sr
        all_rows.append(
            {
                "utt": utt,
                "wav": str(Path(item["clip"]).resolve()),
                "text": TRANSCRIPT_FIXES.get(utt, item["transcript"]),
                "original_asr": item["transcript"],
                "duration": round(duration, 3),
                "sample_rate": sr,
                "rms_db": item.get("rms_db"),
                "peak_db": item.get("peak_db"),
            }
        )

    all_rows.sort(key=lambda row: row["utt"])
    train_rows = [row for row in all_rows if row["utt"] != "haibara_006"]
    dev_rows = [row for row in all_rows if row["utt"] == "haibara_006"]
    if not dev_rows and all_rows:
        dev_rows = [all_rows[-1]]
        train_rows = all_rows[:-1]

    for split, rows in [("train", train_rows), ("dev", dev_rows)]:
        write_kaldi_dir(rows, out / "data" / split)
        make_parquet(rows, out / "parquet" / split)

    summary = {
        "speaker": "haibara_authorized",
        "num_clips_total": len(all_rows),
        "num_train": len(train_rows),
        "num_dev": len(dev_rows),
        "duration_seconds_total": round(sum(row["duration"] for row in all_rows), 3),
        "duration_seconds_train": round(sum(row["duration"] for row in train_rows), 3),
        "duration_seconds_dev": round(sum(row["duration"] for row in dev_rows), 3),
        "rows": all_rows,
        "notes": [
            "This is a tiny authorized speaker dataset for pipeline validation.",
            "The transcripts are manually corrected from local ASR output but still need human verification before serious fine-tuning.",
        ],
    }
    (out / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
