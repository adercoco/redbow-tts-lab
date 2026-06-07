#!/usr/bin/env python3
"""Generate Qwen3 teacher audio for a selected voice profile."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "voice_profiles" / "taiwan_mandarin_low_r.json"
DEFAULT_TEXTS = ROOT / "distillation" / "taiwan_mandarin_low_r" / "seed_texts.jsonl"
DEFAULT_OUT = ROOT / "distillation" / "taiwan_mandarin_low_r" / "teacher_qwen3_1p7b_seed"


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--texts", type=Path, default=DEFAULT_TEXTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse existing ok manifest rows and wav files.")
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    rows = read_jsonl(args.texts)
    if args.limit is not None:
        rows = rows[: args.limit]

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    existing_by_id: dict[str, dict] = {}
    manifest_path = args.out / "manifest.json"
    if args.resume and manifest_path.exists():
        for item in json.loads(manifest_path.read_text(encoding="utf-8")):
            audio = item.get("audio")
            if item.get("status") == "ok" and audio and Path(audio).exists():
                existing_by_id[item["id"]] = item
    print(f"Loading {profile['teacher_model']}...")
    model = load_model(profile["teacher_model"])

    manifest = []
    for row in rows:
        item_id = row["id"]
        text = row["text"]
        category = row.get("category", "")
        output = audio_dir / f"{item_id}.wav"
        if item_id in existing_by_id:
            item = dict(existing_by_id[item_id])
            item["text"] = text
            item["category"] = category
            manifest.append(item)
            print(item_id, "skip", item.get("audio"))
            continue
        started = perf_counter()
        try:
            generate_audio(
                text=text,
                model=model,
                instruct=profile["voice_design_prompt"],
                lang_code=profile.get("language_code", "zh"),
                output_path=str(audio_dir),
                file_prefix=item_id,
                audio_format="wav",
                verbose=False,
            )
            generated = audio_dir / f"{item_id}_000.wav"
            if not generated.exists():
                matches = sorted(audio_dir.glob(f"{item_id}*.wav"))
                if not matches:
                    raise RuntimeError("no wav generated")
                generated = matches[-1]
            if generated != output:
                shutil.copy2(generated, output)
            status = "ok"
            error = ""
            output_text: str | None = str(output)
        except Exception as exc:
            status = "failed"
            error = str(exc)
            output_text = None
        seconds = perf_counter() - started
        print(item_id, status, f"{seconds:.1f}s", output_text or error)
        manifest.append(
            {
                "id": item_id,
                "text": text,
                "category": category,
                "audio": output_text,
                "seconds": seconds,
                "status": status,
                "error": error,
                "teacher_voice_profile": profile["id"],
                "teacher_model": profile["teacher_model"],
            }
        )

    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
