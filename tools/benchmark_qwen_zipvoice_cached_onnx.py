#!/usr/bin/env python3
"""Benchmark cached vs uncached Qwen->ZipVoice few-step ONNX inference."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
ZIPVOICE_ROOT = ROOT / "external" / "ZipVoice"
ZIPVOICE_EGS = ZIPVOICE_ROOT / "egs" / "zipvoice"
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
OUT = BASE / "benchmarks" / "qwen_zipvoice_speedup_v1" / "zipvoice_cached_onnx"
MODEL_DIR = ZIPVOICE_EGS / "exp" / "zipvoice_distill_qwen_teacher_stage2_fewstep10_onnx_epoch10"
PROMPT_WAV = BASE / "teacher_qwen3_1p7b_distill_v1" / "audio" / "distill_0158.wav"
PROMPT_TEXT = "这句话听起来很重要，刚刚那个细节可能不是巧合，你先冷静一点，我有在听。"
TESTS = [
    ("s01", "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。"),
    ("s02", "我知道你现在有点紧张，可是先不要急着证明自己，慢慢说，我会听完。"),
    ("s03", "这件事我们先放在同一个地方整理，等线索够清楚，再决定下一步要怎么做。"),
    ("s04", "你先不要急，我们慢慢来，把事情一件一件处理好。"),
    ("s05", "等一下我先看一下讯息，晚一点再跟你说。"),
]


sys.path.insert(0, str(ZIPVOICE_ROOT))

from zipvoice.bin.infer_zipvoice import get_vocoder  # noqa: E402
from zipvoice.bin.infer_zipvoice_onnx import OnnxModel, generate_sentence, save_wav  # noqa: E402
from zipvoice.models.modules.solver import get_time_steps  # noqa: E402
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer  # noqa: E402
from zipvoice.utils.feature import VocosFbank  # noqa: E402
from zipvoice.utils.infer import (  # noqa: E402
    add_punctuation,
    chunk_tokens_punctuation,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)


def peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if raw > 10_000_000:
        return raw / 1024 / 1024
    return raw / 1024


def model_size_mb() -> dict[str, float]:
    return {
        "text_encoder_int8_mb": (MODEL_DIR / "text_encoder_int8.onnx").stat().st_size / 1024 / 1024,
        "fm_decoder_int8_mb": (MODEL_DIR / "fm_decoder_int8.onnx").stat().st_size / 1024 / 1024,
    }


def wav_seconds(wav: torch.Tensor, sample_rate: int) -> float:
    return float(wav.shape[-1]) / sample_rate


class CachedOnnxEngine:
    def __init__(self, num_threads: int, feat_scale: float = 0.1, target_rms: float = 0.1) -> None:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        load_start = time.perf_counter()
        self.model = OnnxModel(
            str(MODEL_DIR / "text_encoder_int8.onnx"),
            str(MODEL_DIR / "fm_decoder_int8.onnx"),
            num_thread=num_threads,
        )
        self.vocoder = get_vocoder(None)
        self.vocoder.eval()
        self.tokenizer = EmiliaTokenizer(token_file=str(MODEL_DIR / "tokens.txt"))
        self.feature_extractor = VocosFbank()
        self.sample_rate = 24000
        self.feat_scale = feat_scale
        self.target_rms = target_rms
        self.load_seconds = time.perf_counter() - load_start
        self.load_peak_rss_mb = peak_rss_mb()

        cache_start = time.perf_counter()
        prompt_wav = load_prompt_wav(str(PROMPT_WAV), sampling_rate=self.sample_rate)
        prompt_wav = remove_silence(prompt_wav, self.sample_rate, only_edge=False, trail_sil=200)
        prompt_wav, self.prompt_rms = rms_norm(prompt_wav, target_rms)
        self.prompt_duration_s = wav_seconds(prompt_wav, self.sample_rate)
        prompt_features = self.feature_extractor.extract(prompt_wav, sampling_rate=self.sample_rate)
        self.prompt_features = prompt_features.unsqueeze(0) * feat_scale
        prompt_text = add_punctuation(PROMPT_TEXT)
        self.prompt_tokens_str = self.tokenizer.texts_to_tokens([prompt_text])[0]
        self.prompt_tokens = self.tokenizer.tokens_to_token_ids([self.prompt_tokens_str])
        self.prompt_cache_seconds = time.perf_counter() - cache_start

    def timed_sample(
        self,
        tokens: list[int],
        num_step: int,
        guidance_scale: float = 3.0,
        speed: float = 1.0,
        t_shift: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        started = time.perf_counter()
        tokens_t = torch.tensor([tokens], dtype=torch.int64)
        prompt_tokens_t = torch.tensor(self.prompt_tokens, dtype=torch.int64)
        prompt_features_len = torch.tensor(self.prompt_features.size(1), dtype=torch.int64)
        speed_t = torch.tensor(speed, dtype=torch.float32)

        text_encoder_start = time.perf_counter()
        text_condition = self.model.run_text_encoder(
            tokens_t, prompt_tokens_t, prompt_features_len, speed_t
        )
        text_encoder_s = time.perf_counter() - text_encoder_start

        batch_size, num_frames, _ = text_condition.shape
        feat_dim = self.model.feat_dim
        timesteps = get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=num_step,
            t_shift=t_shift,
        )
        x = torch.randn(batch_size, num_frames, feat_dim)
        speech_condition = torch.nn.functional.pad(
            self.prompt_features,
            (0, 0, 0, num_frames - self.prompt_features.shape[1]),
        )
        guidance_scale_t = torch.tensor(guidance_scale, dtype=torch.float32)

        decoder_start = time.perf_counter()
        for step in range(num_step):
            v = self.model.run_fm_decoder(
                t=timesteps[step],
                x=x,
                text_condition=text_condition,
                speech_condition=speech_condition,
                guidance_scale=guidance_scale_t,
            )
            x = x + v * (timesteps[step + 1] - timesteps[step])
        decoder_s = time.perf_counter() - decoder_start
        pred_features = x[:, prompt_features_len.item() :, :]
        return pred_features, {
            "sample_s": time.perf_counter() - started,
            "text_encoder_s": text_encoder_s,
            "decoder_s": decoder_s,
        }

    @torch.inference_mode()
    def synthesize_cached(self, text: str, output: Path, num_step: int) -> dict[str, float]:
        started = time.perf_counter()
        text = add_punctuation(text)
        tokens_start = time.perf_counter()
        tokens_str = self.tokenizer.texts_to_tokens([text])[0]
        token_duration = self.prompt_duration_s / len(self.prompt_tokens_str)
        max_tokens = int((25 - self.prompt_duration_s) / token_duration)
        chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)
        chunked_tokens = self.tokenizer.tokens_to_token_ids(chunked_tokens_str)
        tokenize_s = time.perf_counter() - tokens_start

        features = []
        text_encoder_s = 0.0
        decoder_s = 0.0
        sample_s = 0.0
        for tokens in chunked_tokens:
            pred_features, timing = self.timed_sample(tokens=tokens, num_step=num_step)
            pred_features = pred_features.permute(0, 2, 1) / self.feat_scale
            features.append(pred_features)
            text_encoder_s += timing["text_encoder_s"]
            decoder_s += timing["decoder_s"]
            sample_s += timing["sample_s"]

        vocoder_start = time.perf_counter()
        wavs = []
        for pred_features in features:
            wav = self.vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
            if self.prompt_rms < self.target_rms:
                wav = wav * self.prompt_rms / self.target_rms
            wavs.append(wav)
        final_wav = cross_fade_concat(wavs, fade_duration=0.1, sample_rate=self.sample_rate)
        final_wav = remove_silence(final_wav, self.sample_rate, only_edge=True, trail_sil=0)
        vocoder_s = time.perf_counter() - vocoder_start

        output.parent.mkdir(parents=True, exist_ok=True)
        save_wav(str(output), final_wav, self.sample_rate)
        wall_s = time.perf_counter() - started
        audio_s = wav_seconds(final_wav, self.sample_rate)
        return {
            "wall_s": wall_s,
            "audio_s": audio_s,
            "rtf": wall_s / audio_s if audio_s else 0.0,
            "tokenize_s": tokenize_s,
            "sample_s": sample_s,
            "text_encoder_s": text_encoder_s,
            "decoder_s": decoder_s,
            "vocoder_s": vocoder_s,
            "chunks": float(len(chunked_tokens)),
        }

    @torch.inference_mode()
    def synthesize_uncached(self, text: str, output: Path, num_step: int) -> dict[str, float]:
        started = time.perf_counter()
        metrics = generate_sentence(
            save_path=str(output),
            prompt_text=PROMPT_TEXT,
            prompt_wav=str(PROMPT_WAV),
            text=text,
            model=self.model,
            vocoder=self.vocoder,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            num_step=num_step,
            guidance_scale=3.0,
            speed=1.0,
            t_shift=0.5,
            target_rms=self.target_rms,
            feat_scale=self.feat_scale,
            sampling_rate=self.sample_rate,
            remove_long_sil=False,
        )
        wall_s = time.perf_counter() - started
        return {
            "wall_s": wall_s,
            "audio_s": float(metrics["wav_seconds"]),
            "rtf": wall_s / float(metrics["wav_seconds"]) if metrics["wav_seconds"] else 0.0,
            "model_reported_s": float(metrics["t"]),
            "model_reported_no_vocoder_s": float(metrics["t_no_vocoder"]),
            "model_reported_vocoder_s": float(metrics["t_vocoder"]),
        }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [
        "wall_s",
        "audio_s",
        "rtf",
        "tokenize_s",
        "sample_s",
        "text_encoder_s",
        "decoder_s",
        "vocoder_s",
        "model_reported_s",
    ]
    out: dict[str, Any] = {"count": len(rows)}
    for key in keys:
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
    parser.add_argument("--include-uncached", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    engine = CachedOnnxEngine(num_threads=args.threads)

    rows: list[dict[str, Any]] = []
    for step in args.steps:
        # warmup keeps first-call overhead out of the per-sentence average while
        # still reporting model load and prompt-cache costs separately.
        warm = OUT / "audio" / f"warm_step{step}.wav"
        engine.synthesize_cached(TESTS[0][1], warm, step)
        for sample_id, text in TESTS:
            output = OUT / "audio" / f"cached_step{step}_{sample_id}.wav"
            metrics = engine.synthesize_cached(text, output, step)
            rows.append(
                {
                    "runtime": "zipvoice_repo_cached",
                    "step": step,
                    "sample_id": sample_id,
                    "text": text,
                    "output": str(output),
                    **metrics,
                }
            )
        if args.include_uncached:
            for sample_id, text in TESTS:
                output = OUT / "audio" / f"uncached_step{step}_{sample_id}.wav"
                metrics = engine.synthesize_uncached(text, output, step)
                rows.append(
                    {
                        "runtime": "zipvoice_repo_uncached",
                        "step": step,
                        "sample_id": sample_id,
                        "text": text,
                        "output": str(output),
                        **metrics,
                    }
                )

    summary: dict[str, Any] = {
        "model_dir": str(MODEL_DIR),
        "prompt_wav": str(PROMPT_WAV),
        "prompt_text": PROMPT_TEXT,
        "threads": args.threads,
        "load_seconds": engine.load_seconds,
        "load_peak_rss_mb": engine.load_peak_rss_mb,
        "prompt_cache_seconds": engine.prompt_cache_seconds,
        "final_peak_rss_mb": peak_rss_mb(),
        "model_size_mb": model_size_mb(),
        "rows": rows,
        "aggregates": {},
    }
    for runtime in sorted({row["runtime"] for row in rows}):
        for step in args.steps:
            subset = [row for row in rows if row["runtime"] == runtime and row["step"] == step]
            summary["aggregates"][f"{runtime}_step{step}"] = aggregate(subset)

    out_json = OUT / "benchmark.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_json)
    print(json.dumps(summary["aggregates"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
