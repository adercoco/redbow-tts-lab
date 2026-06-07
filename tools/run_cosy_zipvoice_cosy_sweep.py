#!/usr/bin/env python3
"""Generate CosyVoice2 teacher candidates from enhanced reference packs."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
OUT = BASE / "cosy_zipvoice_distill_v1"
REFS = OUT / "references" / "manifest.json"
AUDIO = OUT / "audio" / "cosyvoice2"
COSY_ROOT = ROOT / "external" / "CosyVoice"

TESTS = [
    ("line_01", "欸，你先别急啦，我们慢慢看，应该可以找到线索。"),
    ("line_02", "没关系，你慢慢说，我在这边听，真的不用紧张。"),
    ("line_03", "我觉得这件事情可以慢慢来，不需要马上决定。"),
    ("line_04", "如果你愿意的话，我们等一下再一起确认一次。"),
]


def normalize_output(wav: torch.Tensor) -> torch.Tensor:
    wav = wav - wav.mean()
    rms = torch.sqrt(torch.mean(wav**2)).clamp_min(1e-8)
    target = 10 ** (-20.0 / 20.0)
    wav = wav * (target / rms)
    peak = wav.abs().max().clamp_min(1e-8)
    if peak > 0.96:
        wav = wav * (0.96 / peak)
    return wav.clamp(-0.98, 0.98)


def main() -> int:
    sys.path.insert(0, str(COSY_ROOT))
    sys.path.insert(0, str(COSY_ROOT / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    packs = json.loads(REFS.read_text(encoding="utf-8"))
    model = AutoModel(model_dir=str(COSY_ROOT / "pretrained_models" / "CosyVoice2-0.5B"))
    rows = []
    for pack in packs:
        for text_id, text in TESTS:
            output = AUDIO / pack["pack_id"] / f"{text_id}.wav"
            raw_output = AUDIO / pack["pack_id"] / f"{text_id}.raw.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            status = "ok"
            error = ""
            try:
                if not output.exists():
                    first = None
                    for index, item in enumerate(
                        model.inference_zero_shot(text, pack["ref_text"], pack["pack_audio"], stream=False)
                    ):
                        if index == 0:
                            first = item["tts_speech"]
                            torchaudio.save(str(raw_output), first, model.sample_rate)
                            torchaudio.save(str(output), normalize_output(first), model.sample_rate)
                            break
                    if first is None:
                        raise RuntimeError("Cosy generated no audio")
            except Exception as exc:
                status = "failed"
                error = str(exc)
            rows.append(
                {
                    "family": "CosyVoice2 tuned teacher",
                    "pack_id": pack["pack_id"],
                    "ref_audio": pack["pack_audio"],
                    "ref_text": pack["ref_text"],
                    "ref_files": pack["ref_files"],
                    "ref_note": pack["note"],
                    "text_id": text_id,
                    "text": text,
                    "output": str(output) if status == "ok" else "",
                    "raw_output": str(raw_output) if status == "ok" else "",
                    "seconds": time.perf_counter() - started,
                    "status": status,
                    "error": error,
                }
            )
            print(f"Cosy {pack['pack_id']} {text_id}: {rows[-1]['seconds']:.2f}s {status}")

    path = OUT / "cosy_sweep_results.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
