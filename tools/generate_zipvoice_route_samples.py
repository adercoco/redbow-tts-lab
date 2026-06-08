from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/Users/ader/Documents/App")
ZIPVOICE_ROOT = ROOT / "external/ZipVoice"
ZIP_EGS = ZIPVOICE_ROOT / "egs/zipvoice"
OUT = ROOT / "distillation/taiwan_mandarin_low_r/reports/teacher_to_zipvoice_route_v1/generated"
REF_AUDIO = ROOT / "distillation/taiwan_mandarin_low_r/datasets/downloads_female_voice/reference_packs_v1/pack_best2_7s.wav"
REF_TEXT = "所以我当时就说,我想要做一张疗愈人的专辑。 开始当然就是我们的提案会议,我就提出了因为多年"
QWEN_PROMPT_WAV = ROOT / "distillation/taiwan_mandarin_low_r/teacher_qwen3_1p7b_distill_v1/audio/distill_0158.wav"
QWEN_PROMPT_TEXT = "这句话听起来很重要，刚刚那个细节可能不是巧合，你先冷静一点，我有在听。"

TEXTS = [
    ("daily_01", "等一下我先把资料整理好，晚点再跟你确认一次。"),
    ("daily_02", "今天先不要想太多，回家路上买杯热的，慢慢来就好。"),
    ("daily_03", "你刚刚那句我有听到，我觉得可以再温柔一点说。"),
]

VARIANTS = [
    {
        "key": "zipvoice_direct_original_16step",
        "label": "Direct ZipVoice original 16-step",
        "model_dir": ZIP_EGS / "exp/zipvoice_original_onnx_int8",
        "steps": 16,
        "prompt_wav": REF_AUDIO,
        "prompt_text": REF_TEXT,
    },
    {
        "key": "zipvoice_cosy_student_16step",
        "label": "ZipVoice Cosy student 16-step",
        "model_dir": ZIP_EGS / "exp/zipvoice_cosy_teacher_smoke_onnx_ckpt60",
        "steps": 16,
        "prompt_wav": REF_AUDIO,
        "prompt_text": REF_TEXT,
    },
    {
        "key": "zipvoice_cosy_student_8step",
        "label": "ZipVoice Cosy student 8-step",
        "model_dir": ZIP_EGS / "exp/zipvoice_cosy_teacher_smoke_onnx_ckpt60",
        "steps": 8,
        "prompt_wav": REF_AUDIO,
        "prompt_text": REF_TEXT,
    },
    {
        "key": "zipvoice_cosy_student_4step",
        "label": "ZipVoice Cosy student 4-step",
        "model_dir": ZIP_EGS / "exp/zipvoice_cosy_teacher_smoke_onnx_ckpt60",
        "steps": 4,
        "prompt_wav": REF_AUDIO,
        "prompt_text": REF_TEXT,
    },
    {
        "key": "zipvoice_qwen_student_stage1_16step",
        "label": "ZipVoice Qwen student stage1 16-step",
        "model_dir": ZIP_EGS / "exp/zipvoice_distill_qwen_teacher_stage1_20_onnx_ckpt10",
        "steps": 16,
        "prompt_wav": QWEN_PROMPT_WAV,
        "prompt_text": QWEN_PROMPT_TEXT,
    },
    {
        "key": "zipvoice_qwen_student_stage2_8step",
        "label": "ZipVoice Qwen student stage2 8-step",
        "model_dir": ZIP_EGS / "exp/zipvoice_distill_qwen_teacher_stage2_fewstep10_onnx_epoch10",
        "steps": 8,
        "prompt_wav": QWEN_PROMPT_WAV,
        "prompt_text": QWEN_PROMPT_TEXT,
    },
    {
        "key": "zipvoice_qwen_fewstep_distilled_4step",
        "label": "ZipVoice distilled 4-step",
        "model_dir": ZIP_EGS / "exp/zipvoice_distill_qwen_teacher_stage2_fewstep10_onnx_epoch10",
        "steps": 4,
        "prompt_wav": QWEN_PROMPT_WAV,
        "prompt_text": QWEN_PROMPT_TEXT,
    },
]

sys.path.insert(0, str(ZIPVOICE_ROOT))

from zipvoice.bin.infer_zipvoice import get_vocoder  # noqa: E402
from zipvoice.bin.infer_zipvoice_onnx import OnnxModel, generate_sentence  # noqa: E402
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer  # noqa: E402
from zipvoice.utils.feature import VocosFbank  # noqa: E402


def peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if raw > 10_000_000:
        return raw / 1024 / 1024
    return raw / 1024


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024 if path.exists() else 0.0


def model_size(model_dir: Path) -> dict[str, float]:
    return {
        "text_encoder_int8_mb": size_mb(model_dir / "text_encoder_int8.onnx"),
        "fm_decoder_int8_mb": size_mb(model_dir / "fm_decoder_int8.onnx"),
        "vocoder_mb": size_mb(ROOT / "models/sherpa/vocos_24khz.onnx"),
    }


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        out_dir = OUT / variant["key"]
        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir = variant["model_dir"]
        load_start = time.perf_counter()
        model = OnnxModel(
            str(model_dir / "text_encoder_int8.onnx"),
            str(model_dir / "fm_decoder_int8.onnx"),
            num_thread=4,
        )
        vocoder = get_vocoder(None)
        vocoder.eval()
        tokenizer = EmiliaTokenizer(token_file=str(model_dir / "tokens.txt"))
        feature_extractor = VocosFbank()
        load_s = time.perf_counter() - load_start
        load_peak = peak_rss_mb()
        for sample_id, text in TEXTS:
            output = out_dir / f"{sample_id}.wav"
            started = time.perf_counter()
            metrics = generate_sentence(
                save_path=str(output),
                prompt_text=variant["prompt_text"],
                prompt_wav=str(variant["prompt_wav"]),
                text=text,
                model=model,
                vocoder=vocoder,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                num_step=variant["steps"],
                guidance_scale=3.0,
                speed=1.0,
                t_shift=0.5,
                target_rms=0.1,
                feat_scale=0.1,
                sampling_rate=24000,
                remove_long_sil=False,
            )
            wall_s = time.perf_counter() - started
            audio_s = float(metrics.get("wav_seconds", 0.0))
            rows.append(
                {
                    "family": variant["key"],
                    "label": variant["label"],
                    "model_dir": str(model_dir),
                    "sample_id": sample_id,
                    "text": text,
                    "output": str(output),
                    "steps": variant["steps"],
                    "load_s": load_s,
                    "load_peak_rss_mb": load_peak,
                    "wall_s": wall_s,
                    "audio_s": audio_s,
                    "rtf": wall_s / audio_s if audio_s else 0.0,
                    "peak_rss_mb": peak_rss_mb(),
                    "model_size_mb": model_size(model_dir),
                    "status": "ok",
                }
            )
            print(variant["key"], sample_id, f"{wall_s:.2f}s", output)
        del model, vocoder, tokenizer, feature_extractor
    (OUT / "zipvoice_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT / "zipvoice_results.json")


if __name__ == "__main__":
    main()
