#!/usr/bin/env python3
"""Generate samples from the F5 model fine-tuned on the Downloads female voice clips."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
PACKS = BASE / "reference_packs_v1" / "manifest.json"
OUT = BASE / "clone_audition_v2_large_models"
AUDIO = OUT / "audio" / "f5_tts_finetuned_40u"
F5_CLI = ROOT / ".venv-f5" / "bin" / "f5-tts_infer-cli"
CKPT = ROOT / ".venv-f5" / "lib" / "python3.12" / "ckpts" / "downloads_female_voice_finetune_v1" / "model_last.pt"
VOCAB = ROOT / ".venv-f5" / "lib" / "python3.12" / "data" / "downloads_female_voice_finetune_v1_pinyin" / "vocab.txt"

TESTS = [
    ("tw_soft", "欸，你先别急啦，我们慢慢看，应该可以找到线索。"),
    ("tw_calm", "没关系，你慢慢说，我在这边听，真的不用紧张。"),
    ("tw_decide", "我觉得这件事情可以慢慢来，不需要马上决定。"),
    ("tw_confirm", "如果你愿意的话，我们等一下再一起确认一次。"),
]


def main() -> int:
    packs = [pack for pack in json.loads(PACKS.read_text(encoding="utf-8")) if pack.get("pack_id") == "pack_best2_7s"]
    pack = packs[0]
    rows = []
    for text_id, text in TESTS:
        output = AUDIO / pack["pack_id"] / f"{text_id}.wav"
        work = output.parent / f"_work_{text_id}"
        output.parent.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status = "ok"
        error = ""
        if not output.exists():
            try:
                subprocess.run(
                    [
                        str(F5_CLI),
                        "--model",
                        "F5TTS_v1_Base",
                        "--ckpt_file",
                        str(CKPT),
                        "--vocab_file",
                        str(VOCAB),
                        "--ref_audio",
                        pack["pack_audio"],
                        "--ref_text",
                        pack["ref_text"],
                        "--gen_text",
                        text,
                        "--output_dir",
                        str(work),
                        "--output_file",
                        output.name,
                        "--remove_silence",
                        "--nfe_step",
                        "32",
                    ],
                    cwd=work,
                    check=True,
                )
                generated = work / output.name
                if not generated.exists():
                    raise RuntimeError(f"F5 generated no wav at {generated}")
                shutil.copy2(generated, output)
            except Exception as exc:
                status = "failed"
                error = str(exc)
        rows.append(
            {
                "family": "F5-TTS fine-tuned 40 updates",
                "candidate_id": "f5_tts_finetuned_40u_pack_best2_7s",
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
                "train_updates": 40,
                "train_dataset_seconds": 40.0,
                "checkpoint": str(CKPT),
            }
        )
        print(f"F5 finetuned {text_id}: {rows[-1]['seconds']:.2f}s {status}")

    path = OUT / "f5_finetuned_40u_results.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
