#!/usr/bin/env python3
"""Benchmark Piper ONNX with fast frontend and process memory snapshots."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import wave
from pathlib import Path
from time import perf_counter

import numpy as np
import resource
from piper.voice import PiperVoice

from prepare_piper_phoneme_ids_dataset import text_to_phonemes


def rss_mb() -> float:
    output = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        text=True,
    ).strip()
    return int(output) / 1024


def peak_rss_mb() -> float:
    # macOS returns ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setframerate(sample_rate)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(audio_i16.tobytes())


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    samples = [
        ("fast_1.wav", "欸，你先不要急啦，我有在聽，我們一步一步確認就好。"),
        ("fast_2.wav", "這個聲音如果要放在手機上，重點是反應要快，聽起來也要像台灣女生。"),
    ]

    result: dict[str, object] = {
        "model": str(args.model),
        "model_size_mb": args.model.stat().st_size / (1024 * 1024),
        "rss_before_load_mb": rss_mb(),
        "samples": [],
    }

    load_start = perf_counter()
    voice = PiperVoice.load(args.model)
    result["load_s"] = perf_counter() - load_start
    result["rss_after_load_mb"] = rss_mb()

    sample_results: list[dict[str, object]] = []
    for name, text in samples:
        phoneme_start = perf_counter()
        phonemes = text_to_phonemes(text)
        phoneme_ids = voice.phonemes_to_ids(phonemes)
        frontend_s = perf_counter() - phoneme_start

        audio_start = perf_counter()
        audio = voice.phoneme_ids_to_audio(phoneme_ids)
        audio_s = perf_counter() - audio_start

        out = args.out_dir / name
        write_wav(out, audio, voice.config.sample_rate)
        duration_s = wav_duration(out)
        total_s = frontend_s + audio_s

        item = {
            "name": name,
            "text": text,
            "file": str(out),
            "frontend_s": frontend_s,
            "audio_s": audio_s,
            "total_s": total_s,
            "duration_s": duration_s,
            "rtf": total_s / duration_s,
            "rss_after_sample_mb": rss_mb(),
        }
        sample_results.append(item)
        print(
            f"{name}\tgen={total_s:.3f}s\tdur={duration_s:.3f}s\t"
            f"rtf={total_s / duration_s:.3f}\trss={item['rss_after_sample_mb']:.1f}MB\t"
            f"file={out}"
        )

    result["samples"] = sample_results
    result["rss_after_all_mb"] = rss_mb()
    result["peak_rss_mb"] = peak_rss_mb()
    print(
        f"load={result['load_s']:.3f}s\t"
        f"rss_load={result['rss_after_load_mb']:.1f}MB\t"
        f"peak_rss={result['peak_rss_mb']:.1f}MB"
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
