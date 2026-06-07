#!/usr/bin/env python3
"""Generate CosyVoice2 zero-shot audition samples."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import torchaudio


ROOT = Path(__file__).resolve().parents[1]
COSY_ROOT = ROOT / "external" / "CosyVoice"
OUT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports" / "teacher_model_audition_v2"
REF_WAV = ROOT / "distillation" / "taiwan_mandarin_low_r" / "teacher_qwen3_1p7b_distill_v1" / "audio" / "distill_0158.wav"
REF_TEXT = "这句话听起来很重要，刚刚那个细节可能不是巧合，你先冷静一点，我有在听。"
TESTS = [
    ("calm", "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。"),
    ("soft", "我知道你现在有点紧张，慢慢说就好，我会听完。"),
]

sys.path.insert(0, str(COSY_ROOT))
sys.path.insert(0, str(COSY_ROOT / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402


def main() -> int:
    audio_dir = OUT / "audio" / "cosyvoice2_qwen_ref"
    audio_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel(model_dir=str(COSY_ROOT / "pretrained_models" / "CosyVoice2-0.5B"))
    rows = []
    for text_id, text in TESTS:
        started = perf_counter()
        final = audio_dir / f"{text_id}.wav"
        status = "ok"
        error = ""
        try:
            for index, item in enumerate(model.inference_zero_shot(text, REF_TEXT, str(REF_WAV), stream=False)):
                if index == 0:
                    torchaudio.save(str(final), item["tts_speech"], model.sample_rate)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            final = None
        rows.append(
            {
                "family": "CosyVoice2",
                "candidate_id": "cosyvoice2_qwen_ref",
                "label": "CosyVoice2 zero-shot Qwen teacher reference",
                "prompt": f"ref={REF_WAV.name}; ref_text={REF_TEXT}",
                "text_id": text_id,
                "text": text,
                "output": str(final) if final else None,
                "seconds": perf_counter() - started,
                "status": status,
                "error": error,
            }
        )
        print(rows[-1])
    path = OUT / "cosyvoice_results.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
