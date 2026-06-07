#!/usr/bin/env python3
"""Export top female-voice segments as TTS-ready 16k/24k datasets."""

from __future__ import annotations

import csv
import json
import shutil
import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
SEGMENTS = BASE / "segments"
CSV_PATH = BASE / "segments.csv"
READY_16K = BASE / "tts_ready_16k_top30"
READY_24K = BASE / "tts_ready_24k_top30"


def write_wav(path: Path, sr: int, data: np.ndarray) -> None:
    peak = float(np.max(np.abs(data)) + 1e-9)
    data = data / max(1.0, peak / 0.92)
    pcm = np.clip(data * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def main() -> int:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    rows.sort(key=lambda row: (-float(row["score"]), float(row["start"])))
    chosen = rows[:30]
    chosen.sort(key=lambda row: float(row["start"]))
    READY_16K.mkdir(parents=True, exist_ok=True)
    READY_24K.mkdir(parents=True, exist_ok=True)

    exported = []
    for index, row in enumerate(chosen, 1):
        source = SEGMENTS / row["file"]
        base_name = f"female_ref_{index:03d}.wav"
        dst16 = READY_16K / base_name
        dst24 = READY_24K / base_name
        shutil.copy2(source, dst16)
        sr, data = wavfile.read(source)
        if data.dtype != np.float32:
            data_f = data.astype(np.float32) / max(1, np.iinfo(data.dtype).max)
        else:
            data_f = data
        data_24k = resample_poly(data_f, 3, 2).astype(np.float32)
        write_wav(dst24, 24000, data_24k)
        exported.append(
            {
                "file": base_name,
                "source_file": row["file"],
                "start": float(row["start"]),
                "end": float(row["end"]),
                "duration": float(row["duration"]),
                "median_f0": float(row["median_f0"]),
                "voiced_ratio": float(row["voiced_ratio"]),
                "score": float(row["score"]),
                "transcript": "",
            }
        )

    for folder in [READY_16K, READY_24K]:
        with (folder / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(exported[0].keys()))
            writer.writeheader()
            writer.writerows(exported)
        with (folder / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for item in exported:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(READY_16K)
    print(READY_24K)
    print(f"exported={len(exported)} total_seconds={sum(i['duration'] for i in exported):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
