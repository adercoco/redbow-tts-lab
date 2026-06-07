#!/usr/bin/env python3
"""Synthesize the IndexTTS2-distilled Matcha pinyin student.

The upstream Matcha CLI always applies english_cleaners2, which requires a
system espeak install. Our student was trained with cleaners=[] on precomputed
ASCII pinyin, so inference must mirror that frontend exactly.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import sys
import wave
from pathlib import Path
from time import perf_counter

import soundfile as sf
import torch

from matcha.cli import load_vocoder, to_waveform
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse

from prepare_matcha_pinyin_dataset import text_to_pinyin_ascii


SAMPLES = [
    {
        "id": "matcha_long_1",
        "text": "你先不要急，我们慢慢来，把事情一件一件处理好。",
    },
    {
        "id": "matcha_long_2",
        "text": "我刚刚看了一下，应该不是你的问题，你不用太担心。",
    },
    {
        "id": "matcha_long_3",
        "text": "没关系啦，你先讲，我在这边听，真的不用紧张。",
    },
    {
        "id": "matcha_long_4",
        "text": "如果你愿意的话，我们等一下再一起确认一次。",
    },
]


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / 1024 / 1024
    return rss / 1024


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocoder", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.667)
    parser.add_argument("--speaking-rate", type=float, default=1.0)
    parser.add_argument(
        "--pinyin-mode",
        choices=("spaced", "punct_tight", "compact"),
        default="spaced",
        help="How to format the pinyin frontend before Matcha text_to_sequence.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    load_start = perf_counter()
    model = MatchaTTS.load_from_checkpoint(args.checkpoint, map_location=device, weights_only=False).eval()
    vocoder, denoiser = load_vocoder("hifigan_T2_v1", args.vocoder, device)
    load_elapsed = perf_counter() - load_start
    print(f"load={load_elapsed:.3f}s")

    metrics = {"load_seconds": load_elapsed, "peak_rss_mb_after_load": peak_rss_mb(), "samples": []}
    for sample in SAMPLES:
        pinyin = format_pinyin(sample["text"], args.pinyin_mode)
        x = torch.tensor(intersperse(text_to_sequence(pinyin, []), 0), dtype=torch.long, device=device)[None]
        x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
        x_phones = sequence_to_text(x.squeeze(0).tolist())

        start = perf_counter()
        with torch.inference_mode():
            output = model.synthesise(
                x,
                x_lengths,
                n_timesteps=args.steps,
                temperature=args.temperature,
                spks=None,
                length_scale=args.speaking_rate,
            )
            waveform = to_waveform(output["mel"], vocoder, denoiser)
        gen_elapsed = perf_counter() - start

        out = args.out_dir / f"{sample['id']}.wav"
        sf.write(str(out), waveform, 22050, "PCM_24")
        duration = wav_duration(out)
        print(
            f"{out.name}\tgen={gen_elapsed:.3f}s\tdur={duration:.3f}s\t"
            f"rtf={gen_elapsed / duration:.3f}\trss={peak_rss_mb():.1f}MB\tpinyin={pinyin}\tphones={x_phones}"
        )
        metrics["samples"].append(
            {
                "id": sample["id"],
                "text": sample["text"],
                "pinyin": pinyin,
                "pinyin_mode": args.pinyin_mode,
                "wav": str(out),
                "gen_seconds": gen_elapsed,
                "audio_seconds": duration,
                "rtf": gen_elapsed / duration,
                "peak_rss_mb": peak_rss_mb(),
            }
        )

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
