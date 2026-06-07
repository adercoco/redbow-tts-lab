#!/usr/bin/env python3
"""Generate CosyVoice teacher samples for daily dialogue comparison reports."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
DOWNLOADS = BASE / "datasets" / "downloads_female_voice"
REF_MANIFEST = DOWNLOADS / "cosy_zipvoice_distill_v1" / "references" / "manifest.json"
OUT = BASE / "teacher_cosy_daily_dialogue_v1"
COSY_ROOT = ROOT / "external" / "CosyVoice"

TEXTS = [
    {
        "id": "daily_01",
        "text": "欸我刚到楼下，你不用急，慢慢来就好，我在这边等你。",
    },
    {
        "id": "daily_02",
        "text": "你要喝什么？我等一下顺路买，珍奶半糖少冰可以吗？",
    },
    {
        "id": "daily_03",
        "text": "今天有点累欸，我们晚餐简单吃就好，不要跑太远。",
    },
    {
        "id": "daily_04",
        "text": "我刚刚在开会没有看到讯息，不是故意不回你啦。",
    },
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

    packs = {item["pack_id"]: item for item in json.loads(REF_MANIFEST.read_text(encoding="utf-8"))}
    pack = packs["raw_best2_7s"]
    audio_dir = OUT / "audio"
    raw_dir = OUT / "raw_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CosyVoice2 model for daily dialogue samples")
    model = AutoModel(model_dir=str(COSY_ROOT / "pretrained_models" / "CosyVoice2-0.5B"))
    manifest = []
    for row in TEXTS:
        output = audio_dir / f"{row['id']}.wav"
        raw_output = raw_dir / f"{row['id']}.wav"
        started = time.perf_counter()
        status = "ok"
        error = ""
        try:
            first = None
            for index, generated in enumerate(
                model.inference_zero_shot(row["text"], pack["ref_text"], pack["pack_audio"], stream=False)
            ):
                if index == 0:
                    first = generated["tts_speech"]
                    torchaudio.save(str(raw_output), first, model.sample_rate)
                    torchaudio.save(str(output), normalize_output(first), model.sample_rate)
                    break
            if first is None:
                raise RuntimeError("Cosy generated no audio")
        except Exception as exc:
            status = "failed"
            error = str(exc)
        seconds = time.perf_counter() - started
        item = {
            **row,
            "audio": str(output) if status == "ok" else None,
            "raw_audio": str(raw_output) if status == "ok" else None,
            "seconds": seconds,
            "status": status,
            "error": error,
            "teacher_model": "CosyVoice2-0.5B",
            "teacher_pack_id": "raw_best2_7s",
            "teacher_ref_audio": pack["pack_audio"],
            "teacher_ref_text": pack["ref_text"],
        }
        manifest.append(item)
        print(row["id"], status, f"{seconds:.2f}s", row["text"])

    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
