#!/usr/bin/env python3
"""Generate GPT-SoVITS v2 zero-shot samples for the teacher-model audition."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external" / "GPT-SoVITS"
OUT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "reports" / "teacher_model_audition_v2"
AUDIO_OUT = OUT / "audio" / "gpt_sovits_v2_qwen_ref"
WORK = AUDIO_OUT / "_work"
REF_AUDIO = ROOT / "distillation" / "taiwan_mandarin_low_r" / "teacher_qwen3_1p7b_distill_v1" / "audio" / "distill_0158.wav"
REF_TEXT = "这句话听起来很重要，刚刚那个细节可能不是巧合，你先冷静一点，我有在听。"

GPT_MODEL = REPO / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
SOVITS_MODEL = REPO / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s2G2333k.pth"

TESTS = [
    (
        "calm",
        "如果你真的想继续查下去，我可以陪你，可是你要先冷静一点。",
    ),
    (
        "soft",
        "我知道你现在有点紧张，慢慢说就好，我会听完。",
    ),
]


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def main() -> int:
    require(REF_AUDIO)
    require(GPT_MODEL)
    require(SOVITS_MODEL)
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    (REPO / "GPT_SoVITS" / "pretrained_models" / "fast_langdetect").mkdir(parents=True, exist_ok=True)

    ref_text_file = WORK / "ref.txt"
    ref_text_file.write_text(REF_TEXT, encoding="utf-8")

    rows: list[dict[str, object]] = []
    env = os.environ.copy()
    env.update(
        {
            "is_half": "False",
            "PYTHONPATH": str(REPO),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    for text_id, text in TESTS:
        target_file = WORK / f"{text_id}.txt"
        target_file.write_text(text, encoding="utf-8")
        run_dir = WORK / f"run_{text_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        output = AUDIO_OUT / f"{text_id}.wav"

        cmd = [
            str(ROOT / ".venv-gptsovits" / "bin" / "python"),
            str(ROOT / "tools" / "run_gpt_sovits_once.py"),
            "--gpt_model",
            str(GPT_MODEL),
            "--sovits_model",
            str(SOVITS_MODEL),
            "--ref_audio",
            str(REF_AUDIO),
            "--ref_text",
            str(ref_text_file),
            "--target_text",
            str(target_file),
            "--output_path",
            str(run_dir),
        ]
        started = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
        seconds = time.perf_counter() - started
        generated = run_dir / "output.wav"
        if proc.returncode == 0 and generated.exists():
            shutil.copy2(generated, output)
            error = ""
        else:
            error = (proc.stderr or proc.stdout or "GPT-SoVITS failed").strip()[-2000:]

        rows.append(
            {
                "family": "GPT-SoVITS v2",
                "candidate_id": "gpt_sovits_v2_qwen_ref",
                "label": "GPT-SoVITS v2 zero-shot Qwen ref",
                "prompt": "官方 GPT-SoVITS v2 底模，使用目前 Qwen3 1.7B 台灣低卷舌女聲作為 prompt audio/text；不是社群聲權重。",
                "text_id": text_id,
                "text": text,
                "seconds": seconds,
                "output": str(output) if output.exists() else "",
                "error": error,
            }
        )
        print(f"{text_id}: {seconds:.2f}s output={output.exists()}")
        if error:
            print(error)

    (OUT / "gpt_sovits_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
