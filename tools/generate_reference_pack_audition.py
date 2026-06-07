#!/usr/bin/env python3
"""Generate CosyVoice2 and F5-TTS auditions from multi-utterance reference packs."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torchaudio


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
PACKS = BASE / "reference_packs_v1" / "manifest.json"
OUT = BASE / "clone_audition_v2_large_models"
AUDIO = OUT / "audio"
COSY_ROOT = ROOT / "external" / "CosyVoice"
F5_CLI = ROOT / ".venv-f5" / "bin" / "f5-tts_infer-cli"

TESTS = [
    ("tw_soft", "欸，你先别急啦，我们慢慢看，应该可以找到线索。"),
    ("tw_calm", "没关系，你慢慢说，我在这边听，真的不用紧张。"),
    ("tw_decide", "我觉得这件事情可以慢慢来，不需要马上决定。"),
    ("tw_confirm", "如果你愿意的话，我们等一下再一起确认一次。"),
]


def load_packs() -> list[dict]:
    packs = json.loads(PACKS.read_text(encoding="utf-8"))
    return [pack for pack in packs if pack.get("pack_audio")]


def base_row(family: str, pack: dict, text_id: str, text: str, output: Path) -> dict:
    return {
        "family": family,
        "candidate_id": f"{family.lower().replace(' ', '_').replace('-', '_')}_{pack['pack_id']}",
        "pack_id": pack["pack_id"],
        "ref_id": pack["pack_id"],
        "ref_audio": pack["pack_audio"],
        "ref_text": pack["ref_text"],
        "ref_files": pack["ref_files"],
        "text_id": text_id,
        "text": text,
        "output": str(output),
    }


def run_cosy(packs: list[dict]) -> list[dict]:
    sys.path.insert(0, str(COSY_ROOT))
    sys.path.insert(0, str(COSY_ROOT / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(model_dir=str(COSY_ROOT / "pretrained_models" / "CosyVoice2-0.5B"))
    rows = []
    for pack in packs:
        for text_id, text in TESTS:
            output = AUDIO / "cosyvoice2_pack" / pack["pack_id"] / f"{text_id}.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            status = "ok"
            error = ""
            if not output.exists():
                try:
                    for index, item in enumerate(
                        model.inference_zero_shot(text, pack["ref_text"], pack["pack_audio"], stream=False)
                    ):
                        if index == 0:
                            torchaudio.save(str(output), item["tts_speech"], model.sample_rate)
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
            row = base_row("CosyVoice2-0.5B pack", pack, text_id, text, output)
            row.update({"seconds": time.perf_counter() - started, "status": status, "error": error})
            if status != "ok":
                row["output"] = ""
            rows.append(row)
            print(f"Cosy pack {pack['pack_id']} {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def run_f5(packs: list[dict]) -> list[dict]:
    rows = []
    for pack in packs:
        for text_id, text in TESTS:
            output = AUDIO / "f5_tts_v1_base_pack" / pack["pack_id"] / f"{text_id}.wav"
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
            row = base_row("F5-TTS v1 Base pack", pack, text_id, text, output)
            row.update({"seconds": time.perf_counter() - started, "status": status, "error": error})
            if status != "ok":
                row["output"] = ""
            rows.append(row)
            print(f"F5 pack {pack['pack_id']} {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    packs = load_packs()
    rows = []
    rows.extend(run_cosy(packs))
    rows.extend(run_f5(packs))
    path = OUT / "reference_pack_results.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
