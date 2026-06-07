#!/usr/bin/env python3
"""Benchmark a Piper ONNX voice with one persistent Python process."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path
from time import perf_counter

from piper.voice import PiperVoice


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        ("api_1.wav", "欸，你先不要急啦，我有在聽，我們一步一步確認就好。"),
        ("api_2.wav", "這個聲音如果要放在手機上，重點是反應要快，聽起來也要像台灣女生。"),
    ]

    load_start = perf_counter()
    voice = PiperVoice.load(args.model)
    print(f"load={perf_counter() - load_start:.3f}s")

    for name, text in samples:
        out = args.out_dir / name
        phoneme_start = perf_counter()
        sentence_phonemes = voice.phonemize(text)
        phoneme_elapsed = perf_counter() - phoneme_start

        id_start = perf_counter()
        sentence_ids = [voice.phonemes_to_ids(phonemes) for phonemes in sentence_phonemes]
        id_elapsed = perf_counter() - id_start

        audio_start = perf_counter()
        for phoneme_ids in sentence_ids:
            voice.phoneme_ids_to_audio(phoneme_ids)
        audio_elapsed = perf_counter() - audio_start

        start = perf_counter()
        with wave.open(str(out), "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        elapsed = perf_counter() - start
        duration = wav_duration(out)
        print(
            f"{name}\tgen={elapsed:.3f}s\tdur={duration:.3f}s\t"
            f"rtf={elapsed / duration:.3f}\tphonemize={phoneme_elapsed:.3f}s\t"
            f"ids={id_elapsed:.3f}s\taudio_only={audio_elapsed:.3f}s\tfile={out}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
