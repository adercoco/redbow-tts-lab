#!/usr/bin/env python3
"""Synthesize Matcha mel with BigVGAN as a replacement vocoder."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from time import perf_counter

import soundfile as sf
import torch

from BigVGAN.bigvgan import BigVGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse

from prepare_matcha_pinyin_dataset import text_to_pinyin_ascii
from synthesize_matcha_pinyin_long import format_pinyin


SAMPLES = [
    ("bigvgan_1", "你先不要急，我们慢慢来，把事情一件一件处理好。"),
    ("bigvgan_2", "我刚刚看了一下，应该不是你的问题，你不用太担心。"),
    ("bigvgan_3", "没关系啦，你先讲，我在这边听，真的不用紧张。"),
]


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="nvidia/bigvgan_v2_22khz_80band_fmax8k_256x")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.50)
    parser.add_argument("--speaking-rate", type=float, default=0.90)
    parser.add_argument("--pinyin-mode", choices=("spaced", "punct_tight", "compact"), default="spaced")
    args = parser.parse_args()

    device = torch.device("cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    load_start = perf_counter()
    matcha = MatchaTTS.load_from_checkpoint(args.checkpoint, map_location=device, weights_only=False).eval()
    bigvgan = BigVGAN.from_pretrained(args.repo_id, use_cuda_kernel=False).eval().to(device)
    bigvgan.remove_weight_norm()
    load_elapsed = perf_counter() - load_start
    print(f"load={load_elapsed:.3f}s repo={args.repo_id}")

    metrics = {"load_seconds": load_elapsed, "repo_id": args.repo_id, "samples": []}
    for sample_id, text in SAMPLES:
        pinyin = format_pinyin(text, args.pinyin_mode)
        x = torch.tensor(intersperse(text_to_sequence(pinyin, []), 0), dtype=torch.long, device=device)[None]
        x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
        x_phones = sequence_to_text(x.squeeze(0).tolist())

        start = perf_counter()
        with torch.inference_mode():
            output = matcha.synthesise(
                x,
                x_lengths,
                n_timesteps=args.steps,
                temperature=args.temperature,
                spks=None,
                length_scale=args.speaking_rate,
            )
            wav = bigvgan(output["mel"]).clamp(-1, 1).squeeze().cpu()
        gen_elapsed = perf_counter() - start

        out = args.out_dir / f"{sample_id}.wav"
        sf.write(str(out), wav.numpy(), 22050, "PCM_24")
        duration = wav_duration(out)
        print(
            f"{out.name}\tgen={gen_elapsed:.3f}s\tdur={duration:.3f}s\t"
            f"rtf={gen_elapsed / duration:.3f}\tpinyin={pinyin}\tphones={x_phones}"
        )
        metrics["samples"].append(
            {
                "id": sample_id,
                "text": text,
                "pinyin": pinyin,
                "wav": str(out),
                "gen_seconds": gen_elapsed,
                "audio_seconds": duration,
                "rtf": gen_elapsed / duration,
            }
        )

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
