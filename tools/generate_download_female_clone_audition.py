#!/usr/bin/env python3
"""Generate clone auditions from strict-clean Downloads female voice references."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torchaudio


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
REF_DIR = BASE / "clean_strict_24k"
OUT = BASE / "clone_audition_v1"
AUDIO = OUT / "audio"
F5_CLI = ROOT / ".venv-f5" / "bin" / "f5-tts_infer-cli"
COSY_ROOT = ROOT / "external" / "CosyVoice"
GPT_RUNNER = ROOT / "tools" / "run_gpt_sovits_role_grid.py"

TESTS = [
    ("tw_soft", "欸，你先别急啦，我们慢慢看，应该可以找到线索。"),
    ("tw_calm", "没关系，你慢慢说，我在这边听，真的不用紧张。"),
]


def load_refs() -> list[dict[str, str]]:
    rows = list(csv.DictReader((REF_DIR / "metadata.csv").open(encoding="utf-8")))
    refs = []
    for index, row in enumerate(rows, 1):
        transcript = row.get("transcript", "").strip()
        if not transcript:
            continue
        refs.append(
            {
                **row,
                "ref_id": f"ref_{index:02d}",
                "ref_audio": str(REF_DIR / row["file"]),
                "ref_text": transcript,
            }
        )
    return refs


def base_row(family: str, ref: dict[str, str], text_id: str, text: str, output: Path) -> dict[str, object]:
    return {
        "family": family,
        "candidate_id": f"{family.lower().replace(' ', '_').replace('-', '_')}_{ref['ref_id']}",
        "role_id": ref["ref_id"],
        "label": ref["ref_id"],
        "ref_id": ref["ref_id"],
        "ref_file": ref["file"],
        "ref_audio": ref["ref_audio"],
        "ref_text": ref["ref_text"],
        "ref_duration": ref["duration"],
        "ref_median_f0": ref["median_f0"],
        "text_id": text_id,
        "text": text,
        "output": str(output),
    }


def run_cosyvoice(refs: list[dict[str, str]]) -> list[dict[str, object]]:
    sys.path.insert(0, str(COSY_ROOT))
    sys.path.insert(0, str(COSY_ROOT / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    model = AutoModel(model_dir=str(COSY_ROOT / "pretrained_models" / "CosyVoice2-0.5B"))
    rows = []
    for ref in refs:
        for text_id, text in TESTS:
            output = AUDIO / "cosyvoice2" / ref["ref_id"] / f"{text_id}.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            status = "ok"
            error = ""
            if not output.exists():
                try:
                    for index, item in enumerate(
                        model.inference_zero_shot(text, ref["ref_text"], ref["ref_audio"], stream=False)
                    ):
                        if index == 0:
                            torchaudio.save(str(output), item["tts_speech"], model.sample_rate)
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
            row = base_row("CosyVoice2-0.5B", ref, text_id, text, output)
            row.update({"seconds": time.perf_counter() - started, "status": status, "error": error})
            if status != "ok":
                row["output"] = ""
            rows.append(row)
            print(f"Cosy {ref['ref_id']} {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def run_gpt_sovits(refs: list[dict[str, str]]) -> list[dict[str, object]]:
    plan = []
    for ref in refs:
        for text_id, text in TESTS:
            output = AUDIO / "gpt_sovits_v2" / ref["ref_id"] / f"{text_id}.wav"
            plan.append(
                {
                    **base_row("GPT-SoVITS v2", ref, text_id, text, output),
                    "ref_audio": ref["ref_audio"],
                    "ref_text": ref["ref_text"],
                }
            )
    plan_path = OUT / "gpt_sovits_plan.json"
    results_path = OUT / "gpt_sovits_results.json"
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(
        [
            str(ROOT / ".venv-gptsovits" / "bin" / "python"),
            str(GPT_RUNNER),
            "--plan",
            str(plan_path),
            "--results",
            str(results_path),
        ],
        cwd=ROOT,
        check=True,
    )
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    for row in rows:
        row["family"] = "GPT-SoVITS v2"
    return rows


def run_f5(refs: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for ref in refs:
        for text_id, text in TESTS:
            output = AUDIO / "f5_tts_v1_base" / ref["ref_id"] / f"{text_id}.wav"
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
                            ref["ref_audio"],
                            "--ref_text",
                            ref["ref_text"],
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
            row = base_row("F5-TTS v1 Base", ref, text_id, text, output)
            row.update({"seconds": time.perf_counter() - started, "status": status, "error": error})
            if status != "ok":
                row["output"] = ""
            rows.append(row)
            print(f"F5 {ref['ref_id']} {text_id}: {row['seconds']:.2f}s {status}")
    return rows


def run_qwen(refs: list[dict[str, str]]) -> list[dict[str, object]]:
    subprocess.run(
        [
            str(ROOT / ".venv-mlx312" / "bin" / "python"),
            str(ROOT / "tools" / "run_qwen_download_female_clone_audition.py"),
        ],
        cwd=ROOT,
        check=True,
    )
    return json.loads((OUT / "qwen3_ref_attempt_results.json").read_text(encoding="utf-8"))


def main() -> int:
    refs = load_refs()
    print(f"refs={len(refs)}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    rows.extend(run_gpt_sovits(refs))
    rows.extend(run_cosyvoice(refs))
    rows.extend(run_qwen(refs))
    rows.extend(run_f5(refs))
    path = OUT / "clone_audition_results.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
