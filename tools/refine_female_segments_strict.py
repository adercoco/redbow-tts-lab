#!/usr/bin/env python3
"""Strictly refine female voice clips by rejecting male, music, and singing-like segments."""

from __future__ import annotations

import csv
import json
import math
import shutil
import wave
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
SEGMENTS = BASE / "segments"
OUT16 = BASE / "clean_strict_16k"
OUT24 = BASE / "clean_strict_24k"
REJECTED = BASE / "rejected_review"
REPORT = BASE / "strict_review.html"
CSV_PATH = BASE / "strict_review.csv"


def write_wav(path: Path, sr: int, data: np.ndarray) -> None:
    peak = float(np.max(np.abs(data)) + 1e-9)
    data = data / max(1.0, peak / 0.92)
    pcm = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def db(x: float) -> float:
    return 20 * math.log10(max(x, 1e-9))


def max_run(mask: np.ndarray) -> int:
    best = cur = 0
    for value in mask:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def analyze(path: Path) -> dict[str, float | str | bool]:
    y, sr = librosa.load(path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=32)
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y * y) + 1e-12))

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("E5"),
        sr=sr,
        frame_length=1024,
        hop_length=160,
    )
    voiced_f0 = f0[np.isfinite(f0)]
    voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
    if len(voiced_f0):
        median_f0 = float(np.median(voiced_f0))
        p10_f0 = float(np.percentile(voiced_f0, 10))
        p90_f0 = float(np.percentile(voiced_f0, 90))
        f0_std = float(np.std(voiced_f0))
        low_f0_ratio = float(np.mean(voiced_f0 < 165))
    else:
        median_f0 = p10_f0 = p90_f0 = f0_std = low_f0_ratio = 0.0

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    harmonic, percussive = librosa.effects.hpss(y)
    harm_energy = float(np.mean(harmonic * harmonic) + 1e-12)
    perc_energy = float(np.mean(percussive * percussive) + 1e-12)
    harmonic_ratio = harm_energy / (harm_energy + perc_energy)

    # Singing often has long stable-pitch runs. Speech has voiced pitch too, but it moves more.
    stable = np.zeros_like(f0, dtype=bool)
    valid_idx = np.where(np.isfinite(f0))[0]
    if len(valid_idx) > 2:
        cents = 1200 * np.log2(f0[valid_idx][1:] / f0[valid_idx][:-1])
        stable_pairs = np.abs(cents) < 28
        stable[valid_idx[1:]] = stable_pairs
    max_stable_s = max_run(stable) * 0.01
    stable_ratio = float(np.mean(stable)) if len(stable) else 0.0

    # Background music often keeps energy active even where speech energy dips.
    intervals = librosa.effects.split(y, top_db=28)
    active_ratio = float(sum((b - a) for a, b in intervals) / max(1, len(y)))
    non_silent_chunks = len(intervals)

    reasons: list[str] = []
    if duration < 2.5 or duration > 10.5:
        reasons.append("bad_duration")
    if median_f0 < 175 or p10_f0 < 145 or low_f0_ratio > 0.22:
        reasons.append("male_or_low_voice_risk")
    if median_f0 > 285 or p90_f0 > 340:
        reasons.append("high_pitch_or_singing_risk")
    if voiced_ratio < 0.42:
        reasons.append("not_enough_voiced_speech")
    if rms < 0.018:
        reasons.append("too_quiet")
    if harmonic_ratio > 0.72 and max_stable_s > 0.55 and stable_ratio > 0.18:
        reasons.append("singing_or_sustained_pitch_risk")
    if active_ratio > 0.94 and non_silent_chunks <= 1 and duration > 5.5 and float(np.median(flat)) > 0.015:
        reasons.append("continuous_music_or_noise_risk")
    if float(np.median(cent)) > 3600 or float(np.median(bandwidth)) > 3600:
        reasons.append("bright_music_or_noise_risk")
    if float(np.median(zcr)) > 0.16:
        reasons.append("noisy_or_music_risk")

    keep = not reasons
    return {
        "duration": duration,
        "rms_db": db(rms),
        "median_f0": median_f0,
        "p10_f0": p10_f0,
        "p90_f0": p90_f0,
        "f0_std": f0_std,
        "low_f0_ratio": low_f0_ratio,
        "voiced_ratio": voiced_ratio,
        "spectral_centroid": float(np.median(cent)),
        "spectral_flatness": float(np.median(flat)),
        "zcr": float(np.median(zcr)),
        "bandwidth": float(np.median(bandwidth)),
        "harmonic_ratio": harmonic_ratio,
        "stable_ratio": stable_ratio,
        "max_stable_s": max_stable_s,
        "active_ratio": active_ratio,
        "keep": keep,
        "reason": ";".join(reasons) if reasons else "keep",
    }


def render_html(rows: list[dict[str, object]]) -> None:
    cards = []
    for row in rows:
        folder = "clean_strict_16k" if row["keep"] else "rejected_review"
        cards.append(
            f"""
            <article class="{ 'keep' if row['keep'] else 'reject' }">
              <div class="head"><b>{row['file']}</b><span>{row['reason']}</span></div>
              <p>dur {float(row['duration']):.2f}s / f0 {float(row['median_f0']):.0f}Hz / low-f0 {float(row['low_f0_ratio']):.2f} / voiced {float(row['voiced_ratio']):.2f} / stable {float(row['max_stable_s']):.2f}s / harm {float(row['harmonic_ratio']):.2f}</p>
              <audio controls preload="metadata" src="{folder}/{row['file']}"></audio>
            </article>
            """
        )
    REPORT.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>嚴格女聲片段審核</title>
  <style>
    body {{ margin:0; background:#f7f8fb; color:#17181c; font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif; }}
    main {{ max-width:960px; margin:0 auto; padding:16px 12px; }}
    h1 {{ margin:0 0 8px; font-size:28px; }}
    p {{ margin:0; color:#5f6977; line-height:1.5; }}
    article {{ margin-top:10px; padding:12px; background:white; border:1px solid #d9dee8; border-radius:8px; }}
    article.reject {{ opacity:.72; border-color:#e0b8bd; }}
    .head {{ display:flex; justify-content:space-between; gap:8px; color:#b0182b; }}
    article p {{ margin:8px 0; }}
    audio {{ width:100%; height:38px; }}
  </style>
</head>
<body><main>
  <h1>嚴格女聲片段審核</h1>
  <p>保留標準偏保守：移除低音男聲風險、配樂/噪音風險、唱歌或長音風險。Reject 不代表一定不能用，只是不要優先餵 TTS。</p>
  {''.join(cards)}
</main></body></html>
""",
        encoding="utf-8",
    )


def main() -> int:
    for folder in [OUT16, OUT24, REJECTED]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for path in sorted(SEGMENTS.glob("*.wav")):
        features = analyze(path)
        row: dict[str, object] = {"file": path.name, **features, "transcript": ""}
        rows.append(row)
        y, sr = sf.read(path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if row["keep"]:
            shutil.copy2(path, OUT16 / path.name)
            y24 = resample_poly(y, 3, 2).astype(np.float32)
            write_wav(OUT24 / path.name, 24000, y24)
        else:
            shutil.copy2(path, REJECTED / path.name)

    rows.sort(key=lambda row: (not bool(row["keep"]), str(row["file"])))
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for folder in [OUT16, OUT24]:
        kept = [row for row in rows if row["keep"]]
        with (folder / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(kept[0].keys()) if kept else ["file"])
            writer.writeheader()
            writer.writerows(kept)
        with (folder / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    render_html(rows)
    print(f"total={len(rows)} keep={sum(1 for row in rows if row['keep'])} reject={sum(1 for row in rows if not row['keep'])}")
    print(OUT16)
    print(OUT24)
    print(REJECTED)
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
