#!/usr/bin/env python3
"""Transcribe strict-clean female reference clips with MLX Whisper."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import mlx_whisper
import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
CLIPS_24K = BASE / "clean_strict_24k"
CLIPS_16K = BASE / "clean_strict_16k"
REVIEW = BASE / "transcript_review.html"
MODEL = "mlx-community/whisper-large-v3-mlx"


def transcribe_one(path: Path) -> dict[str, object]:
    sr, audio = wavfile.read(CLIPS_16K / path.name)
    if sr != 16000:
        raise ValueError(f"expected 16k ASR audio, got {sr}")
    audio = audio.astype(np.float32) / max(1, np.iinfo(audio.dtype).max)
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL,
        language="zh",
        task="transcribe",
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=False,
        initial_prompt="以下是台湾中文访谈或口语讲话。",
        no_speech_threshold=0.45,
    )
    text = str(result.get("text", "")).strip()
    segments = result.get("segments", [])
    avg_logprob = ""
    no_speech_prob = ""
    if segments:
        avg_logprob = segments[0].get("avg_logprob", "")
        no_speech_prob = segments[0].get("no_speech_prob", "")
    return {
        "file": path.name,
        "transcript": text,
        "asr_model": MODEL,
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
    }


def build_review(rows: list[dict[str, object]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
            <article>
              <div><b>{row['file']}</b><span>{row.get('avg_logprob', '')}</span></div>
              <p>{row['transcript']}</p>
              <audio controls preload="metadata" src="clean_strict_24k/{row['file']}"></audio>
            </article>
            """
        )
    REVIEW.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>女聲片段逐字稿審核</title>
  <style>
    body {{ margin:0; background:#f7f8fb; color:#17181c; font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif; }}
    main {{ max-width:900px; margin:0 auto; padding:16px 12px; }}
    h1 {{ margin:0 0 8px; font-size:28px; }}
    .lead {{ color:#5f6977; margin:0 0 12px; }}
    article {{ margin-top:10px; padding:12px; background:#fff; border:1px solid #d9dee8; border-radius:8px; }}
    div {{ display:flex; justify-content:space-between; gap:8px; color:#b0182b; }}
    p {{ margin:8px 0; font-weight:650; line-height:1.5; }}
    audio {{ width:100%; height:38px; }}
  </style>
</head>
<body><main>
  <h1>女聲片段逐字稿審核</h1>
  <p class="lead">ASR 模型：{MODEL}。這是自動稿，餵 TTS 前最好聽過並修正錯字。</p>
  {''.join(cards)}
</main></body></html>
""",
        encoding="utf-8",
    )


def main() -> int:
    clips = sorted(CLIPS_24K.glob("*.wav"))
    rows = []
    for clip in clips:
        row = transcribe_one(clip)
        rows.append(row)
        print(f"{clip.name}: {row['transcript']}")

    # Merge transcript into existing metadata for both 16k and 24k folders.
    by_file = {row["file"]: row for row in rows}
    for folder in [CLIPS_24K, CLIPS_16K]:
        metadata = folder / "metadata.csv"
        old_rows = list(csv.DictReader(metadata.open(encoding="utf-8")))
        for row in old_rows:
            asr = by_file.get(row["file"], {})
            row["transcript"] = str(asr.get("transcript", ""))
            row["asr_model"] = str(asr.get("asr_model", ""))
            row["avg_logprob"] = str(asr.get("avg_logprob", ""))
            row["no_speech_prob"] = str(asr.get("no_speech_prob", ""))
        fieldnames = list(dict.fromkeys([key for row in old_rows for key in row.keys()]))
        with metadata.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(old_rows)
        with (folder / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in old_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (BASE / "transcripts_whisper_large_v3_turbo.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    build_review(rows)
    print(REVIEW)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
