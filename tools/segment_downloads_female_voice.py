#!/usr/bin/env python3
"""Segment a Downloads female-voice audio file into TTS-ready clips."""

from __future__ import annotations

import csv
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
INPUT = BASE / "raw" / "female_voice_16k_mono.wav"
OUT = BASE / "segments"
HTML = BASE / "preview.html"
CSV_PATH = BASE / "segments.csv"


@dataclass
class Segment:
    start: float
    end: float
    rms_db: float
    voiced_ratio: float
    median_f0: float
    score: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype != np.float32:
        data = data.astype(np.float32) / max(1, np.iinfo(data.dtype).max)
    return sr, data


def highpass(data: np.ndarray, sr: int) -> np.ndarray:
    b, a = butter(3, 70 / (sr / 2), btype="highpass")
    return filtfilt(b, a, data).astype(np.float32)


def frame_rms_db(data: np.ndarray, frame: int, hop: int) -> np.ndarray:
    values = []
    for start in range(0, max(1, len(data) - frame + 1), hop):
        chunk = data[start : start + frame]
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        values.append(20 * math.log10(rms + 1e-9))
    return np.array(values, dtype=np.float32)


def estimate_f0_frame(frame_data: np.ndarray, sr: int) -> float:
    frame_data = frame_data - np.mean(frame_data)
    energy = float(np.sqrt(np.mean(frame_data * frame_data) + 1e-12))
    if energy < 0.008:
        return 0.0
    frame_data *= np.hanning(len(frame_data))
    corr = np.correlate(frame_data, frame_data, mode="full")[len(frame_data) - 1 :]
    min_lag = int(sr / 320)
    max_lag = int(sr / 90)
    if max_lag >= len(corr):
        return 0.0
    window = corr[min_lag:max_lag]
    if len(window) == 0:
        return 0.0
    lag = int(np.argmax(window) + min_lag)
    peak = corr[lag] / (corr[0] + 1e-9)
    if peak < 0.25:
        return 0.0
    return float(sr / lag)


def segment_f0(data: np.ndarray, sr: int, start_s: float, end_s: float) -> tuple[float, float]:
    start = int(start_s * sr)
    end = int(end_s * sr)
    clip = data[start:end]
    frame = int(0.04 * sr)
    hop = int(0.02 * sr)
    f0s = []
    for pos in range(0, max(1, len(clip) - frame + 1), hop):
        f0 = estimate_f0_frame(clip[pos : pos + frame], sr)
        if 90 <= f0 <= 320:
            f0s.append(f0)
    if not f0s:
        return 0.0, 0.0
    f0_arr = np.array(f0s)
    return float(np.median(f0_arr)), float(len(f0s) / max(1, (len(clip) - frame) // hop + 1))


def find_voice_regions(data: np.ndarray, sr: int) -> list[tuple[float, float]]:
    frame = int(0.03 * sr)
    hop = int(0.01 * sr)
    rms = frame_rms_db(data, frame, hop)
    floor = float(np.percentile(rms, 25))
    active_threshold = max(floor + 13.0, -42.0)
    active = rms > active_threshold

    # Smooth VAD by requiring local support.
    kernel = np.ones(7, dtype=np.float32) / 7
    smooth = np.convolve(active.astype(np.float32), kernel, mode="same") > 0.35

    regions: list[tuple[float, float]] = []
    start_idx: int | None = None
    for idx, is_active in enumerate(smooth):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            regions.append((start_idx * hop / sr, (idx * hop + frame) / sr))
            start_idx = None
    if start_idx is not None:
        regions.append((start_idx * hop / sr, len(data) / sr))

    # Merge short gaps, then split overly long regions into TTS-sized clips.
    merged: list[tuple[float, float]] = []
    for start, end in regions:
        if not merged or start - merged[-1][1] > 0.45:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], end)

    final: list[tuple[float, float]] = []
    for start, end in merged:
        start = max(0.0, start - 0.08)
        end = min(len(data) / sr, end + 0.12)
        dur = end - start
        if dur < 2.2:
            continue
        if dur <= 12.0:
            final.append((start, end))
            continue
        cursor = start
        while end - cursor > 12.0:
            final.append((cursor, cursor + 9.0))
            cursor += 8.4
        if end - cursor >= 2.2:
            final.append((cursor, end))
    return final


def score_segment(data: np.ndarray, sr: int, start: float, end: float) -> Segment:
    clip = data[int(start * sr) : int(end * sr)]
    rms = float(np.sqrt(np.mean(clip * clip) + 1e-12))
    rms_db = 20 * math.log10(rms + 1e-9)
    median_f0, voiced_ratio = segment_f0(data, sr, start, end)
    duration = end - start
    f0_score = 1.0 if 165 <= median_f0 <= 285 else 0.35 if 145 <= median_f0 < 165 else 0.0
    dur_score = 1.0 if 3.0 <= duration <= 10.0 else 0.6
    level_score = 1.0 if -34 <= rms_db <= -12 else 0.5
    score = 0.45 * f0_score + 0.25 * voiced_ratio + 0.2 * dur_score + 0.1 * level_score
    return Segment(start, end, rms_db, voiced_ratio, median_f0, score)


def write_wav(path: Path, sr: int, clip: np.ndarray) -> None:
    peak = float(np.max(np.abs(clip)) + 1e-9)
    if peak > 0:
        clip = clip / max(1.0, peak / 0.92)
    pcm = np.clip(clip * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def build_html(rows: list[dict[str, object]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
            <article>
              <div><b>{row['file']}</b><span>{row['start']:.2f}s–{row['end']:.2f}s</span></div>
              <p>dur {row['duration']:.2f}s / f0 {row['median_f0']:.0f}Hz / voiced {row['voiced_ratio']:.2f} / score {row['score']:.2f}</p>
              <audio controls preload="metadata" src="segments/{row['file']}"></audio>
            </article>
            """
        )
    HTML.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>女聲 TTS 片段候選</title>
  <style>
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC",sans-serif; background:#f7f8fb; color:#17181c; }}
    main {{ max-width:920px; margin:0 auto; padding:16px 12px; }}
    h1 {{ font-size:28px; margin:0 0 8px; }}
    p {{ margin:0; color:#5f6977; }}
    article {{ margin-top:10px; padding:12px; background:white; border:1px solid #d9dee8; border-radius:8px; }}
    article div {{ display:flex; justify-content:space-between; gap:8px; color:#b0182b; }}
    article p {{ margin:8px 0; }}
    audio {{ width:100%; height:38px; }}
  </style>
</head>
<body><main>
  <h1>女聲 TTS 片段候選</h1>
  <p>已轉成 16k mono wav。這是自動粗切，請優先聽 score 高、沒有背景音、沒有重疊人聲的片段。</p>
  {''.join(cards)}
</main></body></html>
""",
        encoding="utf-8",
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sr, data = read_wav(INPUT)
    if sr != 16000:
        raise ValueError(f"expected 16k wav, got {sr}")
    filtered = highpass(data, sr)
    raw_regions = find_voice_regions(filtered, sr)
    scored = [score_segment(filtered, sr, start, end) for start, end in raw_regions]
    selected = [
        seg
        for seg in scored
        if 2.5 <= seg.duration <= 12.0
        and seg.voiced_ratio >= 0.28
        and seg.median_f0 >= 145
        and seg.score >= 0.42
    ]
    selected.sort(key=lambda s: (-s.score, s.start))
    selected = selected[:80]
    selected.sort(key=lambda s: s.start)

    rows: list[dict[str, object]] = []
    for idx, seg in enumerate(selected, 1):
        clip = filtered[int(seg.start * sr) : int(seg.end * sr)]
        filename = f"female_seg_{idx:03d}_{seg.start:07.2f}_{seg.end:07.2f}.wav"
        write_wav(OUT / filename, sr, clip)
        rows.append(
            {
                "file": filename,
                "start": seg.start,
                "end": seg.end,
                "duration": seg.duration,
                "rms_db": seg.rms_db,
                "median_f0": seg.median_f0,
                "voiced_ratio": seg.voiced_ratio,
                "score": seg.score,
                "transcript": "",
            }
        )

    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    build_html(rows)
    print(f"regions={len(raw_regions)} selected={len(rows)}")
    print(CSV_PATH)
    print(HTML)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
