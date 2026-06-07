#!/usr/bin/env python3
"""Run GPT-SoVITS v2 with native auxiliary reference audio paths."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external" / "GPT-SoVITS"
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
PACKS = BASE / "reference_packs_v1" / "manifest.json"
OUT = BASE / "clone_audition_v2_large_models"
AUDIO = OUT / "audio" / "gpt_sovits_v2_aux"

TESTS = [
    ("tw_soft", "欸，你先别急啦，我们慢慢看，应该可以找到线索。"),
    ("tw_calm", "没关系，你慢慢说，我在这边听，真的不用紧张。"),
    ("tw_decide", "我觉得这件事情可以慢慢来，不需要马上决定。"),
    ("tw_confirm", "如果你愿意的话，我们等一下再一起确认一次。"),
]


def soundfile_load(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T), sr


def main() -> int:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "GPT_SoVITS"))
    os.environ.setdefault("is_half", "False")
    os.environ.setdefault("version", "v2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torchaudio.load = soundfile_load

    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    config.device = "cpu"
    config.is_half = False
    config.version = "v2"
    pipeline = TTS(config)

    packs = json.loads(PACKS.read_text(encoding="utf-8"))
    pack = next(item for item in packs if item["pack_id"] == "pack_gpt_aux")
    ref_audio = pack["ref_audio_paths"][0]
    aux_refs = pack["ref_audio_paths"][1:]
    prompt_text = " ".join(pack["ref_text"].split()[:80])

    rows = []
    AUDIO.mkdir(parents=True, exist_ok=True)
    for text_id, text in TESTS:
        output = AUDIO / f"{text_id}.wav"
        started = time.perf_counter()
        status = "ok"
        error = ""
        try:
            result = pipeline.run(
                {
                    "text": text,
                    "text_lang": "zh",
                    "ref_audio_path": ref_audio,
                    "aux_ref_audio_paths": aux_refs,
                    "prompt_text": prompt_text,
                    "prompt_lang": "zh",
                    "top_k": 15,
                    "top_p": 1,
                    "temperature": 1,
                    "text_split_method": "cut5",
                    "batch_size": 1,
                    "batch_threshold": 0.75,
                    "split_bucket": True,
                    "speed_factor": 1.0,
                    "fragment_interval": 0.3,
                    "seed": 666,
                    "parallel_infer": True,
                    "repetition_penalty": 1.35,
                    "sample_steps": 32,
                    "super_sampling": False,
                    "streaming_mode": False,
                }
            )
            if hasattr(result, "__iter__") and not isinstance(result, tuple):
                result = list(result)[-1]
            sr, audio = result
            sf.write(output, audio, sr)
        except Exception as exc:
            status = "failed"
            error = str(exc)
        rows.append(
            {
                "family": "GPT-SoVITS v2 aux refs",
                "candidate_id": "gpt_sovits_v2_aux_pack_gpt_aux",
                "pack_id": "pack_gpt_aux",
                "ref_id": "pack_gpt_aux",
                "ref_audio": ref_audio,
                "aux_ref_audio_paths": aux_refs,
                "ref_text": prompt_text,
                "text_id": text_id,
                "text": text,
                "output": str(output) if output.exists() else "",
                "seconds": time.perf_counter() - started,
                "status": status,
                "error": error,
            }
        )
        print(f"GPT aux {text_id}: {rows[-1]['seconds']:.2f}s {status}")

    path = OUT / "gpt_sovits_aux_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
