#!/usr/bin/env python3
"""Generate Qwen3 VoiceDesign no-reference Taiwan female samples."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from time import perf_counter

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model

from no_reference_voice_design_shared import TESTS, VOICE_DESIGNS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports" / "teacher_model_audition_v4_no_reference"
MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"


def main() -> int:
    audio_root = OUT / "audio" / "qwen3_voicedesign"
    audio_root.mkdir(parents=True, exist_ok=True)
    model = load_model(MODEL_ID)
    rows = []
    for design in VOICE_DESIGNS:
        model_dir = audio_root / design.id
        model_dir.mkdir(parents=True, exist_ok=True)
        for text_id, text in TESTS:
            final = model_dir / f"{text_id}.wav"
            started = perf_counter()
            status = "ok"
            error = ""
            if not final.exists():
                try:
                    generate_audio(
                        text=text,
                        model=model,
                        instruct=design.prompt,
                        lang_code="zh",
                        output_path=str(model_dir),
                        file_prefix=text_id,
                        audio_format="wav",
                        verbose=False,
                    )
                    generated = model_dir / f"{text_id}_000.wav"
                    if not generated.exists():
                        matches = sorted(model_dir.glob(f"{text_id}*.wav"))
                        if not matches:
                            raise RuntimeError("Qwen generated no wav")
                        generated = matches[-1]
                    shutil.copy2(generated, final)
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
            rows.append(
                {
                    "family": "Qwen3 VoiceDesign",
                    "candidate_id": f"qwen3_{design.id}",
                    "design_id": design.id,
                    "label": design.label,
                    "prompt": design.prompt,
                    "text_id": text_id,
                    "text": text,
                    "seconds": perf_counter() - started,
                    "status": status,
                    "error": error,
                    "output": str(final) if final.exists() else "",
                    "mode": "text voice design; no reference audio",
                }
            )
            print(f"Qwen {design.id} {text_id}: {rows[-1]['seconds']:.2f}s {status}")
    path = OUT / "qwen3_no_reference_results.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
