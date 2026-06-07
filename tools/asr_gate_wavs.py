#!/usr/bin/env python3
"""Run a lightweight Chinese ASR gate for generated TTS wav files."""

from __future__ import annotations

import argparse
import json
import math
import re
import wave
from pathlib import Path

import mlx_whisper
import numpy as np
from scipy.signal import resample_poly

try:
    import soundfile as sf
except ModuleNotFoundError:
    sf = None


ROOT = Path(__file__).resolve().parents[1]


def load_wav_16k(path: Path) -> np.ndarray:
    if sf is not None:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    else:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            raw = wav.readframes(wav.getnframes())
        if sample_width != 2:
            raise ValueError(f"{path}: expected 16-bit PCM wav, got sample_width={sample_width}")
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype(np.float32)
    return audio


def normalize_text(text: str) -> str:
    folded = text.lower().translate(
        str.maketrans(
            {
                "願": "愿",
                "話": "话",
                "們": "们",
                "這": "这",
                "邊": "边",
                "聽": "听",
                "緊": "紧",
                "張": "张",
                "關": "关",
                "係": "系",
                "剛": "刚",
                "應": "应",
                "該": "该",
                "題": "题",
                "處": "处",
                "確": "确",
                "認": "认",
                "讓": "让",
            }
        )
    )
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", folded))


def lcs_similarity(expected: str, actual: str) -> float:
    a = normalize_text(expected)
    b = normalize_text(actual)
    if not a:
        return 0.0
    dp = [0] * (len(b) + 1)
    for char_a in a:
        next_dp = dp[:]
        for j, char_b in enumerate(b, start=1):
            if char_a == char_b:
                next_dp[j] = dp[j - 1] + 1
            else:
                next_dp[j] = max(dp[j], next_dp[j - 1])
        dp = next_dp
    return dp[-1] / len(a)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True, help="JSON rows with name/path/expected.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/whisper-large-v3-mlx")
    parser.add_argument("--pass-threshold", type=float, default=0.6)
    args = parser.parse_args()

    items_path = args.items if args.items.is_absolute() else ROOT / args.items
    out = args.out if args.out.is_absolute() else ROOT / args.out
    rows = json.loads(items_path.read_text(encoding="utf-8"))
    results = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            path = ROOT / path
        print(f"ASR {row.get('name', path.name)}")
        audio = load_wav_16k(path)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=args.model,
            language="zh",
            task="transcribe",
            verbose=False,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        text = str(result.get("text", "")).strip()
        similarity = lcs_similarity(str(row.get("expected", "")), text)
        out_row = {
            **row,
            "path": str(path),
            "asr": text,
            "asr_similarity": round(similarity, 4),
            "asr_pass": similarity >= args.pass_threshold,
        }
        results.append(out_row)
        print(f"  sim={similarity:.3f} pass={out_row['asr_pass']} text={text}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    passed = sum(1 for row in results if row["asr_pass"])
    print(f"passed={passed}/{len(results)}")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
