#!/usr/bin/env python3
"""Run IndexTTS2 auditions with the prepared reference packs."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_ROOT = ROOT / "external" / "index-tts"
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
PACKS = BASE / "reference_packs_v1" / "manifest.json"
OUT = BASE / "clone_audition_v2_large_models"
AUDIO = OUT / "audio" / "indextts2_pack"

TESTS = [
    ("tw_soft", "欸，你先别急啦，我们慢慢看，应该可以找到线索。"),
    ("tw_calm", "没关系，你慢慢说，我在这边听，真的不用紧张。"),
    ("tw_decide", "我觉得这件事情可以慢慢来，不需要马上决定。"),
    ("tw_confirm", "如果你愿意的话，我们等一下再一起确认一次。"),
]


def main() -> int:
    os.chdir(INDEX_ROOT)
    sys.path.insert(0, str(INDEX_ROOT))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from indextts.infer_v2 import IndexTTS2

    packs = [pack for pack in json.loads(PACKS.read_text(encoding="utf-8")) if pack.get("pack_audio")]
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=False,
        device="mps",
        use_cuda_kernel=False,
        use_deepspeed=False,
    )

    rows = []
    AUDIO.mkdir(parents=True, exist_ok=True)
    for pack in packs:
        for text_id, text in TESTS:
            output = AUDIO / pack["pack_id"] / f"{text_id}.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            status = "ok"
            error = ""
            if not output.exists():
                try:
                    tts.infer(
                        spk_audio_prompt=pack["pack_audio"],
                        text=text,
                        output_path=str(output),
                        emo_vector=[0, 0, 0, 0, 0, 0, 0, 0.8],
                        use_random=False,
                        verbose=True,
                    )
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
            rows.append(
                {
                    "family": "IndexTTS2 pack",
                    "candidate_id": f"indextts2_pack_{pack['pack_id']}",
                    "pack_id": pack["pack_id"],
                    "ref_id": pack["pack_id"],
                    "ref_audio": pack["pack_audio"],
                    "ref_text": pack["ref_text"],
                    "ref_files": pack["ref_files"],
                    "text_id": text_id,
                    "text": text,
                    "output": str(output) if output.exists() else "",
                    "seconds": time.perf_counter() - started,
                    "status": status,
                    "error": error,
                }
            )
            print(f"IndexTTS2 {pack['pack_id']} {text_id}: {rows[-1]['seconds']:.2f}s {status}")

    path = OUT / "indextts2_pack_results.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
