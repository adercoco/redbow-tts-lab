#!/usr/bin/env python3
"""Synthesize with Piper ONNX using the lightweight pypinyin phoneme frontend."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path
from time import perf_counter

import numpy as np
from piper.voice import PiperVoice

from prepare_piper_phoneme_ids_dataset import text_to_phonemes


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

    samples = [
        ("fast_1.wav", "欸，你先不要急啦，我有在聽，我們一步一步確認就好。"),
        ("fast_2.wav", "這個聲音如果要放在手機上，重點是反應要快，聽起來也要像台灣女生。"),
    ]

    load_start = perf_counter()
    voice = PiperVoice.load(args.model)
    print(f"load={perf_counter() - load_start:.3f}s")

    for name, text in samples:
        phoneme_start = perf_counter()
        phonemes = text_to_phonemes(text)
        phoneme_ids = voice.phonemes_to_ids(phonemes)
        frontend_elapsed = perf_counter() - phoneme_start

        audio_start = perf_counter()
        audio = voice.phoneme_ids_to_audio(phoneme_ids)
        audio_elapsed = perf_counter() - audio_start

        out = args.out_dir / name
        write_wav(out, audio, voice.config.sample_rate)
        duration = wav_duration(out)
        total = frontend_elapsed + audio_elapsed
        print(
            f"{name}\tgen={total:.3f}s\tdur={duration:.3f}s\t"
            f"rtf={total / duration:.3f}\tfrontend={frontend_elapsed:.3f}s\t"
            f"audio={audio_elapsed:.3f}s\tfile={out}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
