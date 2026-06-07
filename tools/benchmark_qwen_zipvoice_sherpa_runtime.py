#!/usr/bin/env python3
"""Benchmark Qwen->ZipVoice few-step inference through sherpa-onnx runtime."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any

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
DEFAULT_OUT = BASE / "benchmarks" / "qwen_zipvoice_speedup_v1" / "sherpa_runtime"
TESTS = [
    ("s01", "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。"),
    ("s02", "我知道你现在有点紧张，可是先不要急着证明自己，慢慢说，我会听完。"),
    ("s03", "这件事我们先放在同一个地方整理，等线索够清楚，再决定下一步要怎么做。"),
    ("s04", "你先不要急，我们慢慢来，把事情一件一件处理好。"),
    ("s05", "等一下我先看一下讯息，晚一点再跟你说。"),
]


def peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if raw > 10_000_000:
        return raw / 1024 / 1024
    return raw / 1024


def resolve_zipvoice_files(model_dir: Path) -> dict[str, Path]:
    encoder = model_dir / "text_encoder_int8.onnx"
    decoder = model_dir / "fm_decoder_int8.onnx"
    if not encoder.exists():
        encoder = model_dir / "encoder.int8.onnx"
    if not decoder.exists():
        decoder = model_dir / "decoder.int8.onnx"
    tokens = model_dir / "tokens.txt"
    if not tokens.exists():
        tokens = MODEL_DIR / "tokens.txt"
    for label, path in {"encoder": encoder, "decoder": decoder, "tokens": tokens}.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    return {"encoder": encoder, "decoder": decoder, "tokens": tokens}


def model_size_mb(model_dir: Path, vocoder: Path) -> dict[str, float]:
    files = resolve_zipvoice_files(model_dir)
    return {
        "encoder_int8_mb": files["encoder"].stat().st_size / 1024 / 1024,
        "decoder_int8_mb": files["decoder"].stat().st_size / 1024 / 1024,
        "vocoder_mb": vocoder.stat().st_size / 1024 / 1024,
        "total_mb": (
            files["encoder"].stat().st_size
            + files["decoder"].stat().st_size
            + vocoder.stat().st_size
        )
        / 1024
        / 1024,
    }


def make_tts(num_threads: int, model_dir: Path, vocoder: Path) -> sherpa_onnx.OfflineTts:
    files = resolve_zipvoice_files(model_dir)
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=str(files["tokens"]),
                encoder=str(files["encoder"]),
                decoder=str(files["decoder"]),
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
        raise RuntimeError(f"Invalid sherpa ZipVoice config: {model_dir}")
    return sherpa_onnx.OfflineTts(config)


def load_reference() -> tuple[Any, int]:
    audio, sample_rate = sf.read(str(PROMPT_WAV), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def synthesize(
    tts: sherpa_onnx.OfflineTts,
    text: str,
    output: Path,
    reference_audio: Any,
    reference_sample_rate: int,
    num_steps: int,
    speed: float,
) -> dict[str, float]:
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.reference_audio = reference_audio
    gen_config.reference_sample_rate = reference_sample_rate
    gen_config.reference_text = PROMPT_TEXT
    gen_config.num_steps = num_steps
    gen_config.speed = speed
    gen_config.extra["min_char_in_sentence"] = "20"

    started = time.perf_counter()
    audio = tts.generate(text, gen_config)
    wall_s = time.perf_counter() - started
    if len(audio.samples) == 0:
        raise RuntimeError("sherpa-onnx returned empty audio")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
    audio_s = len(audio.samples) / audio.sample_rate
    return {
        "wall_s": wall_s,
        "audio_s": audio_s,
        "rtf": wall_s / audio_s if audio_s else 0.0,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows)}
    for key in ["wall_s", "audio_s", "rtf"]:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            out[f"avg_{key}"] = sum(vals) / len(vals)
            out[f"sum_{key}"] = sum(vals)
    out["avg_chars"] = sum(len(str(row["text"])) for row in rows) / len(rows)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--vocoder", type=Path, default=VOCODER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    tts = make_tts(args.threads, args.model_dir, args.vocoder)
    load_seconds = time.perf_counter() - load_start
    load_peak_rss_mb = peak_rss_mb()

    reference_start = time.perf_counter()
    reference_audio, reference_sample_rate = load_reference()
    reference_load_seconds = time.perf_counter() - reference_start

    rows: list[dict[str, Any]] = []
    for step in args.steps:
        speed_suffix = "" if args.speed == 1.0 else f"_speed{str(args.speed).replace('.', 'p')}"
        synthesize(
            tts,
            TESTS[0][1],
            out_dir / "audio" / f"warm_step{step}{speed_suffix}.wav",
            reference_audio,
            reference_sample_rate,
            step,
            args.speed,
        )
        for sample_id, text in TESTS:
            output = out_dir / "audio" / f"sherpa_step{step}{speed_suffix}_{sample_id}.wav"
            metrics = synthesize(
                tts,
                text,
                output,
                reference_audio,
                reference_sample_rate,
                step,
                args.speed,
            )
            rows.append(
                {
                    "runtime": "sherpa_onnx_python_binding",
                    "step": step,
                    "sample_id": sample_id,
                    "text": text,
                    "output": str(output),
                    **metrics,
                }
            )

    summary: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "prompt_wav": str(PROMPT_WAV),
        "prompt_text": PROMPT_TEXT,
        "threads": args.threads,
        "speed": args.speed,
        "vocoder": str(args.vocoder),
        "load_seconds": load_seconds,
        "load_peak_rss_mb": load_peak_rss_mb,
        "reference_load_seconds": reference_load_seconds,
        "final_peak_rss_mb": peak_rss_mb(),
        "model_size_mb": model_size_mb(args.model_dir, args.vocoder),
        "rows": rows,
        "aggregates": {},
    }
    for step in args.steps:
        subset = [row for row in rows if row["step"] == step]
        summary["aggregates"][f"sherpa_step{step}"] = aggregate(subset)

    out_json = out_dir / "benchmark.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_json)
    print(json.dumps(summary["aggregates"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
