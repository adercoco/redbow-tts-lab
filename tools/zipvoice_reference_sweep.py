#!/usr/bin/env python3
"""Sweep teacher reference clips for ZipVoice zero-shot similarity."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from build_zipvoice_three_way_report import (
    BASE,
    SELECTED_IDS,
    TEACHER_AUDIO_DIR,
    TEXTS,
    ensure_zipvoice_tts,
    read_jsonl,
    score_against_teacher,
    zipvoice_generate,
)


OUT = BASE / "reports" / "zipvoice_reference_sweep"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refs",
        nargs="+",
        default=["seed_0001", "seed_0002", "seed_0003", "seed_0004", "seed_0005", "seed_0006"],
    )
    parser.add_argument("--eval-ids", nargs="+", default=SELECTED_IDS)
    args = parser.parse_args()

    rows_by_id = {row["id"]: row for row in read_jsonl(TEXTS)}
    tts = ensure_zipvoice_tts()
    summary = []
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zipvoice-ref-sweep-") as tmp:
        tmp_dir = Path(tmp)
        for ref_id in args.refs:
            ref_audio = TEACHER_AUDIO_DIR / f"{ref_id}.wav"
            ref_text = rows_by_id[ref_id]["text"]
            scores = []
            seconds = []
            for eval_id in args.eval_ids:
                output = tmp_dir / ref_id / f"{eval_id}.wav"
                elapsed = zipvoice_generate(tts, rows_by_id[eval_id]["text"], ref_audio, ref_text, output)
                total, mfcc, f0, centroid = score_against_teacher(TEACHER_AUDIO_DIR / f"{eval_id}.wav", output)
                scores.append({"id": eval_id, "score": total, "mfcc": mfcc, "f0": f0, "centroid": centroid})
                seconds.append(elapsed)
            avg = sum(item["score"] for item in scores) / len(scores)
            avg_seconds = sum(seconds) / len(seconds)
            row = {
                "reference_id": ref_id,
                "reference_text": ref_text,
                "eval_count": len(scores),
                "avg_score": avg,
                "avg_seconds": avg_seconds,
                "scores": scores,
            }
            summary.append(row)
            print(ref_id, f"score={avg:.3f}", f"seconds={avg_seconds:.2f}")

    summary.sort(key=lambda item: item["avg_score"], reverse=True)
    (OUT / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("best", summary[0]["reference_id"], f"{summary[0]['avg_score']:.3f}")
    print(OUT / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
