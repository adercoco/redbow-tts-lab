#!/usr/bin/env python3
"""Generate CosyVoice-300M-Instruct no-external-reference samples."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import torchaudio

from no_reference_voice_design_shared import TESTS, VOICE_DESIGNS


ROOT = Path(__file__).resolve().parents[1]
COSY_ROOT = ROOT / "external" / "CosyVoice"
OUT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports" / "teacher_model_audition_v4_no_reference"
MODEL_DIR = COSY_ROOT / "pretrained_models" / "CosyVoice-300M-Instruct"
SPK_ID = "中文女"

sys.path.insert(0, str(COSY_ROOT))
sys.path.insert(0, str(COSY_ROOT / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402


def main() -> int:
    audio_root = OUT / "audio" / "cosyvoice_300m_instruct_short"
    audio_root.mkdir(parents=True, exist_ok=True)
    model = AutoModel(model_dir=str(MODEL_DIR))
    spks = model.list_available_spks()
    if SPK_ID not in spks:
        raise RuntimeError(f"{SPK_ID} not found in speakers: {spks}")
    rows = []
    for design in VOICE_DESIGNS:
        model_dir = audio_root / design.id
        model_dir.mkdir(parents=True, exist_ok=True)
        instruct = "You are a helpful assistant. 请用台湾国语口音、温柔年轻女声、低卷舌表达。<|endofprompt|>"
        for text_id, text in TESTS:
            final = model_dir / f"{text_id}.wav"
            started = perf_counter()
            status = "ok"
            error = ""
            if not final.exists():
                try:
                    for index, item in enumerate(
                        model.inference_instruct(text, SPK_ID, instruct, stream=False, speed=1.0)
                    ):
                        if index == 0:
                            torchaudio.save(str(final), item["tts_speech"], model.sample_rate)
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
            rows.append(
                {
                    "family": "CosyVoice-300M-Instruct",
                    "candidate_id": f"cosy_instruct_{design.id}",
                    "design_id": design.id,
                    "label": design.label,
                    "prompt": instruct,
                    "text_id": text_id,
                    "text": text,
                    "seconds": perf_counter() - started,
                    "status": status,
                    "error": error,
                    "output": str(final) if final.exists() else "",
                    "mode": f"short instruct + built-in speaker {SPK_ID}; no external reference audio",
                }
            )
            print(f"CosyInstruct {design.id} {text_id}: {rows[-1]['seconds']:.2f}s {status}")
    path = OUT / "cosyvoice_instruct_short_no_reference_results.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
