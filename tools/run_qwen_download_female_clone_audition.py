#!/usr/bin/env python3
"""Run Qwen3 ref-audio attempt for Downloads female clone audition."""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
REF_DIR = BASE / "clean_strict_24k"
OUT = BASE / "clone_audition_v1"
AUDIO = OUT / "audio"
TESTS = [
    ("tw_soft", "欸，你先别急啦，我们慢慢看，应该可以找到线索。"),
    ("tw_calm", "没关系，你慢慢说，我在这边听，真的不用紧张。"),
]


def load_refs() -> list[dict[str, str]]:
    rows = list(csv.DictReader((REF_DIR / "metadata.csv").open(encoding="utf-8")))
    refs = []
    for index, row in enumerate(rows, 1):
        transcript = row.get("transcript", "").strip()
        if transcript:
            refs.append(
                {
                    **row,
                    "ref_id": f"ref_{index:02d}",
                    "ref_audio": str(REF_DIR / row["file"]),
                    "ref_text": transcript,
                }
            )
    return refs


def base_row(ref: dict[str, str], text_id: str, text: str, output: Path) -> dict[str, object]:
    return {
        "family": "Qwen3 1.7B VoiceDesign ref attempt",
        "candidate_id": f"qwen3_1p7b_ref_attempt_{ref['ref_id']}",
        "ref_id": ref["ref_id"],
        "ref_file": ref["file"],
        "ref_audio": ref["ref_audio"],
        "ref_text": ref["ref_text"],
        "ref_duration": ref["duration"],
        "ref_median_f0": ref["median_f0"],
        "text_id": text_id,
        "text": text,
        "output": str(output),
        "note": "Qwen3 VoiceDesign checkpoint requires instruct; ref_audio/ref_text is passed as an experimental ref attempt, not guaranteed pure clone.",
    }


def main() -> int:
    refs = load_refs()
    model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit")
    instruct = "台湾年轻女生，声音温柔清亮，低卷舌，语气自然，像真人访谈口吻。"
    rows = []
    for ref in refs:
        for text_id, text in TESTS:
            output_dir = AUDIO / "qwen3_1p7b_ref_attempt" / ref["ref_id"]
            output = output_dir / f"{text_id}.wav"
            output_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            status = "ok"
            error = ""
            if not output.exists():
                try:
                    generate_audio(
                        text=text,
                        model=model,
                        instruct=instruct,
                        ref_audio=ref["ref_audio"],
                        ref_text=ref["ref_text"],
                        lang_code="zh",
                        output_path=str(output_dir),
                        file_prefix=text_id,
                        audio_format="wav",
                        verbose=False,
                    )
                    generated = output_dir / f"{text_id}_000.wav"
                    if not generated.exists():
                        matches = sorted(output_dir.glob(f"{text_id}*.wav"))
                        if not matches:
                            raise RuntimeError("Qwen generated no wav")
                        generated = matches[-1]
                    shutil.copy2(generated, output)
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
            row = base_row(ref, text_id, text, output)
            row.update({"seconds": time.perf_counter() - started, "status": status, "error": error})
            if status != "ok":
                row["output"] = ""
            rows.append(row)
            print(f"Qwen {ref['ref_id']} {text_id}: {row['seconds']:.2f}s {status}")
    path = OUT / "qwen3_ref_attempt_results.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
