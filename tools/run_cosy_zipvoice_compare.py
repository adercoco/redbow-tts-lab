#!/usr/bin/env python3
"""Generate ZipVoice original/distilled outputs using Cosy teacher references."""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

import sherpa_onnx
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
OUT = BASE / "cosy_zipvoice_distill_v1"
AUDIO = OUT / "audio"

ZIP_EGS = ROOT / "external" / "ZipVoice" / "egs" / "zipvoice"
ORIGINAL_EXP = ZIP_EGS / "exp" / "zipvoice_original_onnx_int8"
DISTILL_EXP = ZIP_EGS / "exp" / "zipvoice_distill_qwen_teacher_stage2_fewstep10_onnx_epoch10"
PACKAGED = ROOT / "models" / "sherpa" / "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia"
VOCODER = ROOT / "models" / "sherpa" / "vocos_24khz.onnx"

TESTS = [
    ("line_01", "欸，你先别急啦，我们慢慢看，应该可以找到线索。"),
    ("line_02", "没关系，你慢慢说，我在这边听，真的不用紧张。"),
    ("line_03", "我觉得这件事情可以慢慢来，不需要马上决定。"),
    ("line_04", "如果你愿意的话，我们等一下再一起确认一次。"),
]

TEACHER_PACKS = ["raw_best2_7s", "clear_best2_7s"]


def peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if raw > 10_000_000:
        return raw / 1024 / 1024
    return raw / 1024


def model_size(exp: Path) -> int:
    total = 0
    for name in ["text_encoder_int8.onnx", "fm_decoder_int8.onnx"]:
        total += (exp / name).stat().st_size
    total += VOCODER.stat().st_size
    return total


def fmt_mb(value: int) -> str:
    return f"{value / 1024 / 1024:.1f}MB"


def make_tts(exp: Path):
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            zipvoice=sherpa_onnx.OfflineTtsZipvoiceModelConfig(
                tokens=str(exp / "tokens.txt"),
                encoder=str(exp / "text_encoder_int8.onnx"),
                decoder=str(exp / "fm_decoder_int8.onnx"),
                data_dir=str(PACKAGED / "espeak-ng-data"),
                lexicon=str(PACKAGED / "lexicon.txt"),
                vocoder=str(VOCODER),
            ),
            debug=False,
            num_threads=4,
            provider="cpu",
        )
    )
    if not config.validate():
        raise RuntimeError(f"Invalid ZipVoice config: {exp}")
    return sherpa_onnx.OfflineTts(config)


def generate(tts, text: str, ref_audio: Path, ref_text: str, output: Path, steps: int) -> float:
    reference_audio, sample_rate = sf.read(str(ref_audio), dtype="float32", always_2d=False)
    if getattr(reference_audio, "ndim", 1) > 1:
        reference_audio = reference_audio.mean(axis=1)
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.reference_audio = reference_audio
    gen_config.reference_sample_rate = sample_rate
    gen_config.reference_text = ref_text
    gen_config.num_steps = steps
    gen_config.extra["min_char_in_sentence"] = "20"
    started = time.perf_counter()
    audio = tts.generate(text, gen_config)
    elapsed = time.perf_counter() - started
    if len(audio.samples) == 0:
        raise RuntimeError("ZipVoice returned empty audio")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
    return elapsed


def main() -> int:
    cosy_rows = json.loads((OUT / "cosy_sweep_results.json").read_text(encoding="utf-8"))
    cosy_by_pack_text = {(row["pack_id"], row["text_id"]): row for row in cosy_rows if row.get("status") == "ok"}
    variants = [
        {
            "family": "ZipVoice original 16-step",
            "variant_id": "zipvoice_original_step16",
            "exp": ORIGINAL_EXP,
            "steps": 16,
            "model_size": fmt_mb(model_size(ORIGINAL_EXP)),
            "note": "官方原始 ZipVoice ONNX int8；同 Cosy reference zero-shot 模仿。",
        },
        {
            "family": "ZipVoice-Distill stage2 4-step",
            "variant_id": "zipvoice_distill_stage2_step4",
            "exp": DISTILL_EXP,
            "steps": 4,
            "model_size": fmt_mb(model_size(DISTILL_EXP)),
            "note": "已蒸餾 few-step ZipVoice ONNX int8；目前模型是上一輪 Qwen teacher 蒸餾權重，但這次 inference 使用 Cosy reference。",
        },
    ]

    rows: list[dict] = []
    for variant in variants:
        print(f"loading {variant['family']}")
        load_started = time.perf_counter()
        tts = make_tts(variant["exp"])
        load_seconds = time.perf_counter() - load_started
        load_rss = peak_rss_mb()
        for pack_id in TEACHER_PACKS:
            teacher = cosy_by_pack_text[(pack_id, "line_01")]
            ref_audio = Path(teacher["output"])
            ref_text = teacher["text"]
            for text_id, text in TESTS:
                output = AUDIO / variant["variant_id"] / pack_id / f"{text_id}.wav"
                status = "ok"
                error = ""
                elapsed = 0.0
                try:
                    if not output.exists():
                        elapsed = generate(tts, text, ref_audio, ref_text, output, variant["steps"])
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
                rows.append(
                    {
                        "family": variant["family"],
                        "variant_id": variant["variant_id"],
                        "pack_id": pack_id,
                        "text_id": text_id,
                        "text": text,
                        "teacher_audio": cosy_by_pack_text[(pack_id, text_id)]["output"],
                        "teacher_ref_audio": str(ref_audio),
                        "teacher_ref_text": ref_text,
                        "output": str(output) if status == "ok" else "",
                        "seconds": elapsed,
                        "load_seconds": load_seconds,
                        "peak_rss_mb_after_load": round(load_rss, 1),
                        "peak_rss_mb_after_generate": round(peak_rss_mb(), 1),
                        "steps": variant["steps"],
                        "model_size": variant["model_size"],
                        "note": variant["note"],
                        "status": status,
                        "error": error,
                    }
                )
                print(
                    variant["variant_id"],
                    pack_id,
                    text_id,
                    f"{elapsed:.2f}s",
                    status,
                    f"rss={rows[-1]['peak_rss_mb_after_generate']}MB",
                )

    path = OUT / "zipvoice_compare_results.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
