#!/usr/bin/env python3
"""Synthesize report samples for the long Piper student."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from time import perf_counter

import numpy as np
from piper.voice import PiperVoice

from prepare_piper_phoneme_ids_dataset import text_to_phonemes


SAMPLES = [
    {
        "id": "piper_long_1",
        "text": "你先不要急，我们慢慢来，把事情一件一件处理好。",
    },
    {
        "id": "piper_long_2",
        "text": "我刚刚看了一下，应该不是你的问题，你不用太担心。",
    },
    {
        "id": "piper_long_3",
        "text": "没关系啦，你先讲，我在这边听，真的不用紧张。",
    },
]


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
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    load_start = perf_counter()
    voice = PiperVoice.load(args.model)
    load_elapsed = perf_counter() - load_start
    print(f"load={load_elapsed:.3f}s")

    metrics = {"load_seconds": load_elapsed, "samples": []}
    for sample in SAMPLES:
        frontend_start = perf_counter()
        phonemes = text_to_phonemes(sample["text"])
        phoneme_ids = voice.phonemes_to_ids(phonemes)
        frontend_elapsed = perf_counter() - frontend_start

        audio_start = perf_counter()
        audio = voice.phoneme_ids_to_audio(phoneme_ids)
        audio_elapsed = perf_counter() - audio_start

        out = args.out_dir / f"{sample['id']}.wav"
        write_wav(out, audio, voice.config.sample_rate)
        duration = wav_duration(out)
        gen_elapsed = frontend_elapsed + audio_elapsed
        print(
            f"{out.name}\tgen={gen_elapsed:.3f}s\tdur={duration:.3f}s\t"
            f"rtf={gen_elapsed / duration:.3f}\tfrontend={frontend_elapsed:.3f}s\taudio={audio_elapsed:.3f}s"
        )
        metrics["samples"].append(
            {
                "id": sample["id"],
                "text": sample["text"],
                "wav": str(out),
                "gen_seconds": gen_elapsed,
                "frontend_seconds": frontend_elapsed,
                "audio_seconds": duration,
                "rtf": gen_elapsed / duration,
            }
        )

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
