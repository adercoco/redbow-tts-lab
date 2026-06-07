#!/usr/bin/env python3
"""Benchmark app-level chunked playback for Qwen->ZipVoice Sherpa runtime."""

from __future__ import annotations

import argparse
import json
import re
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
ZIP_EGS = ROOT / "external" / "ZipVoice" / "egs" / "zipvoice"
MODEL_DIR = ZIP_EGS / "exp" / "zipvoice_distill_qwen_teacher_stage2_fewstep10_onnx_epoch10"
PACKAGED = ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia-zh-min"
VOCODER = ROOT / "models" / "sherpa" / "vocos_24khz.onnx"
PROMPT_WAV = BASE / "teacher_qwen3_1p7b_distill_v1" / "audio" / "distill_0158.wav"
PROMPT_TEXT = "这句话听起来很重要，刚刚那个细节可能不是巧合，你先冷静一点，我有在听。"
OUT = BASE / "benchmarks" / "qwen_zipvoice_speedup_v1" / "chunked_playback"

LONG_TESTS = [
    (
        "long_01",
        "你先不要急，我们慢慢来，把事情一件一件处理好。等一下我先看一下讯息，晚一点再跟你说。你不用马上证明自己没有错，我有在听。",
    ),
    (
        "long_02",
        "我知道你现在有点紧张，可是我们先不要急着下结论。先把资料放在同一个地方整理清楚，等线索够完整，再决定下一步要怎么做。",
    ),
]


def peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if raw > 10_000_000:
        return raw / 1024 / 1024
    return raw / 1024


def model_size_mb(vocoder: Path) -> dict[str, float]:
    return {
        "encoder_int8_mb": (MODEL_DIR / "text_encoder_int8.onnx").stat().st_size / 1024 / 1024,
        "decoder_int8_mb": (MODEL_DIR / "fm_decoder_int8.onnx").stat().st_size / 1024 / 1024,
        "vocoder_mb": vocoder.stat().st_size / 1024 / 1024,
        "total_mb": (
            (MODEL_DIR / "text_encoder_int8.onnx").stat().st_size
            + (MODEL_DIR / "fm_decoder_int8.onnx").stat().st_size
            + vocoder.stat().st_size
        )
        / 1024
        / 1024,
    }


def split_text(text: str, max_chars: int = 34) -> list[str]:
    pieces = [p.strip() for p in re.split(r"(?<=[。！？!?，,])", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current += piece
    if current:
        chunks.append(current)
    return chunks


def make_tts(num_threads: int, vocoder: Path) -> sherpa_onnx.OfflineTts:
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=str(MODEL_DIR / "tokens.txt"),
                encoder=str(MODEL_DIR / "text_encoder_int8.onnx"),
                decoder=str(MODEL_DIR / "fm_decoder_int8.onnx"),
                data_dir=str(PACKAGED / "espeak-ng-data"),
                lexicon=str(PACKAGED / "lexicon.txt"),
                vocoder=str(vocoder),
            ),
            debug=False,
            num_threads=num_threads,
            provider="cpu",
        )
    )
    if not config.validate():
        raise RuntimeError(f"Invalid sherpa ZipVoice config: {MODEL_DIR}")
    return sherpa_onnx.OfflineTts(config)


def load_reference() -> tuple[Any, int]:
    audio, sample_rate = sf.read(str(PROMPT_WAV), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def generate(
    tts: sherpa_onnx.OfflineTts,
    text: str,
    reference_audio: Any,
    reference_sample_rate: int,
    steps: int,
    speed: float,
) -> tuple[np.ndarray, int, dict[str, float]]:
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.reference_audio = reference_audio
    gen_config.reference_sample_rate = reference_sample_rate
    gen_config.reference_text = PROMPT_TEXT
    gen_config.num_steps = steps
    gen_config.speed = speed
    gen_config.extra["min_char_in_sentence"] = "20"

    started = time.perf_counter()
    audio = tts.generate(text, gen_config)
    wall_s = time.perf_counter() - started
    if len(audio.samples) == 0:
        raise RuntimeError("sherpa-onnx returned empty audio")
    samples = np.asarray(audio.samples, dtype=np.float32)
    audio_s = len(samples) / audio.sample_rate
    return samples, audio.sample_rate, {
        "wall_s": wall_s,
        "audio_s": audio_s,
        "rtf": wall_s / audio_s if audio_s else 0.0,
    }


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, samplerate=sample_rate, subtype="PCM_16")


def concat_chunks(chunks: list[np.ndarray], sample_rate: int) -> np.ndarray:
    silence = np.zeros(int(sample_rate * 0.05), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        if index:
            parts.append(silence)
        parts.append(chunk)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--max-chars", type=int, default=34)
    parser.add_argument("--vocoder", type=Path, default=VOCODER)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    tts = make_tts(args.threads, args.vocoder)
    load_seconds = time.perf_counter() - load_started
    reference_audio, reference_sample_rate = load_reference()

    rows: list[dict[str, Any]] = []
    for steps in args.steps:
        for sample_id, text in LONG_TESTS:
            full_samples, sr, full_metrics = generate(
                tts, text, reference_audio, reference_sample_rate, steps, args.speed
            )
            speed_suffix = "" if args.speed == 1.0 else f"_speed{str(args.speed).replace('.', 'p')}"
            full_output = out_dir / "audio" / f"full_step{steps}{speed_suffix}_{sample_id}.wav"
            write_wav(full_output, full_samples, sr)
            rows.append(
                {
                    "mode": "full_sentence",
                    "steps": steps,
                    "sample_id": sample_id,
                    "text": text,
                    "chunks": [text],
                    "ttfa_s": full_metrics["wall_s"],
                    "total_wall_s": full_metrics["wall_s"],
                    "audio_s": full_metrics["audio_s"],
                    "rtf": full_metrics["rtf"],
                    "output": str(full_output),
                }
            )

            chunk_texts = split_text(text, max_chars=args.max_chars)
            chunk_samples: list[np.ndarray] = []
            chunk_metrics: list[dict[str, float]] = []
            for chunk_index, chunk in enumerate(chunk_texts, start=1):
                samples, sr, metrics = generate(
                    tts, chunk, reference_audio, reference_sample_rate, steps, args.speed
                )
                chunk_output = out_dir / "audio" / f"chunk_step{steps}{speed_suffix}_{sample_id}_{chunk_index:02d}.wav"
                write_wav(chunk_output, samples, sr)
                chunk_samples.append(samples)
                chunk_metrics.append(metrics)
            combined = concat_chunks(chunk_samples, sr)
            combined_output = out_dir / "audio" / f"chunked_step{steps}{speed_suffix}_{sample_id}.wav"
            write_wav(combined_output, combined, sr)
            total_wall_s = sum(metric["wall_s"] for metric in chunk_metrics)
            audio_s = len(combined) / sr
            rows.append(
                {
                    "mode": "chunked_playback",
                    "steps": steps,
                    "sample_id": sample_id,
                    "text": text,
                    "chunks": chunk_texts,
                    "chunk_wall_s": [metric["wall_s"] for metric in chunk_metrics],
                    "ttfa_s": chunk_metrics[0]["wall_s"],
                    "total_wall_s": total_wall_s,
                    "audio_s": audio_s,
                    "rtf": total_wall_s / audio_s if audio_s else 0.0,
                    "output": str(combined_output),
                }
            )

    summary = {
        "model_dir": str(MODEL_DIR),
        "vocoder": str(args.vocoder),
        "threads": args.threads,
        "steps": args.steps,
        "max_chars": args.max_chars,
        "speed": args.speed,
        "load_seconds": load_seconds,
        "peak_rss_mb": peak_rss_mb(),
        "model_size_mb": model_size_mb(args.vocoder),
        "rows": rows,
    }
    out_json = out_dir / "benchmark.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_json)
    for row in rows:
        print(
            row["mode"],
            "step",
            row["steps"],
            row["sample_id"],
            "ttfa",
            f"{row['ttfa_s']:.2f}s",
            "total",
            f"{row['total_wall_s']:.2f}s",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
