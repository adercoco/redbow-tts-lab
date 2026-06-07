#!/usr/bin/env python3
"""Run the exported Matcha ONNX acoustic model with our pinyin frontend."""

from __future__ import annotations

import argparse
import json
import re
import wave
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from matcha.cli import load_vocoder, to_waveform
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse

from prepare_matcha_pinyin_dataset import text_to_pinyin_ascii


SAMPLES = [
    ("matcha_onnx_1", "你先不要急，我们慢慢来，把事情一件一件处理好。"),
    ("matcha_onnx_2", "我刚刚看了一下，应该不是你的问题，你不用太担心。"),
    ("matcha_onnx_3", "没关系啦，你先讲，我在这边听，真的不用紧张。"),
]


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def format_pinyin(text: str, mode: str) -> str:
    pinyin = text_to_pinyin_ascii(text, tones=True)
    if mode == "spaced":
        return pinyin
    if mode == "compact":
        return pinyin.replace(" ", "")
    if mode == "punct_tight":
        pinyin = re.sub(r"\s+([,.;:!?])", r"\1", pinyin)
        pinyin = re.sub(r"([,.;:!?])\s+", r"\1", pinyin)
        return pinyin
    raise ValueError(f"unknown pinyin mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vocoder", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.50)
    parser.add_argument("--speaking-rate", type=float, default=0.85)
    parser.add_argument(
        "--pinyin-mode",
        choices=("spaced", "punct_tight", "compact"),
        default="compact",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    load_start = perf_counter()
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    vocoder, denoiser = load_vocoder("hifigan_T2_v1", args.vocoder, device)
    load_seconds = perf_counter() - load_start

    metrics = {
        "model": str(args.model),
        "load_seconds": load_seconds,
        "samples": [],
    }
    for sample_id, text in SAMPLES:
        pinyin = format_pinyin(text, args.pinyin_mode)
        token_ids = intersperse(text_to_sequence(pinyin, []), 0)
        x = np.asarray([token_ids], dtype=np.int64)
        x_lengths = np.asarray([len(token_ids)], dtype=np.int64)
        scales = np.asarray([args.temperature, args.speaking_rate], dtype=np.float32)
        phones = sequence_to_text(token_ids)

        start = perf_counter()
        mel, mel_lengths = session.run(
            None,
            {
                "x": x,
                "x_lengths": x_lengths,
                "scales": scales,
            },
        )
        mel_seconds = perf_counter() - start

        voc_start = perf_counter()
        with torch.inference_mode():
            waveform = to_waveform(torch.from_numpy(mel).to(device), vocoder, denoiser)
        voc_seconds = perf_counter() - voc_start

        out = args.out_dir / f"{sample_id}.wav"
        sf.write(str(out), waveform, 22050, "PCM_24")
        duration = wav_duration(out)
        total = mel_seconds + voc_seconds
        print(
            f"{out.name}\tmel={mel_seconds:.3f}s\tvoc={voc_seconds:.3f}s\t"
            f"total={total:.3f}s\tdur={duration:.3f}s\trtf={total / duration:.3f}\t"
            f"pinyin={pinyin}\tphones={phones}\tmel_len={mel_lengths.tolist()}"
        )
        metrics["samples"].append(
            {
                "id": sample_id,
                "text": text,
                "pinyin": pinyin,
                "phones": phones,
                "wav": str(out),
                "mel_seconds": mel_seconds,
                "vocoder_seconds": voc_seconds,
                "total_seconds": total,
                "audio_seconds": duration,
                "rtf": total / duration,
            }
        )

    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
