#!/usr/bin/env python3
"""Generate a CosyVoice teacher corpus for true ZipVoice distillation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
TEXTS = BASE / "distill_texts_v1.jsonl"
DOWNLOADS = BASE / "datasets" / "downloads_female_voice"
REF_MANIFEST = DOWNLOADS / "cosy_zipvoice_distill_v1" / "references" / "manifest.json"
DEFAULT_OUT = BASE / "teacher_cosy_raw_best2_distill_v1"
COSY_ROOT = ROOT / "external" / "CosyVoice"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--texts", type=Path, default=TEXTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pack-id", default="raw_best2_7s")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-skips", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(COSY_ROOT))
    sys.path.insert(0, str(COSY_ROOT / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    packs = {item["pack_id"]: item for item in json.loads(REF_MANIFEST.read_text(encoding="utf-8"))}
    pack = packs[args.pack_id]
    rows = read_jsonl(args.texts)[: args.limit]
    audio_dir = args.out / "audio"
    raw_dir = args.out / "raw_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"

    existing: dict[str, dict] = {}
    if args.resume and manifest_path.exists():
        for item in json.loads(manifest_path.read_text(encoding="utf-8")):
            if item.get("status") == "ok" and item.get("audio") and Path(item["audio"]).exists():
                existing[item["id"]] = item

    print(f"Loading CosyVoice2 model, pack={args.pack_id}")
    model = AutoModel(model_dir=str(COSY_ROOT / "pretrained_models" / "CosyVoice2-0.5B"))
    manifest = []
    for row in rows:
        item_id = row["id"]
        text = row["text"]
        category = row.get("category", "")
        output = audio_dir / f"{item_id}.wav"
        raw_output = raw_dir / f"{item_id}.wav"
        if item_id in existing:
            item = dict(existing[item_id])
            item["text"] = text
            item["category"] = category
            manifest.append(item)
            if not args.quiet_skips:
                print(item_id, "skip")
            continue
        if args.resume and output.exists() and raw_output.exists():
            item = {
                "id": item_id,
                "text": text,
                "category": category,
                "audio": str(output),
                "raw_audio": str(raw_output),
                "seconds": None,
                "status": "ok",
                "error": "",
                "teacher_model": "CosyVoice2-0.5B",
                "teacher_pack_id": args.pack_id,
                "teacher_ref_audio": pack["pack_audio"],
                "teacher_ref_text": pack["ref_text"],
                "resume_note": "Recovered from existing audio/raw_audio files.",
            }
            manifest.append(item)
            if not args.quiet_skips:
                print(item_id, "skip-existing-file")
            continue
        started = time.perf_counter()
        status = "ok"
        error = ""
        try:
            first = None
            for index, generated in enumerate(
                model.inference_zero_shot(text, pack["ref_text"], pack["pack_audio"], stream=False)
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
            "id": item_id,
            "text": text,
            "category": category,
            "audio": str(output) if status == "ok" else None,
            "raw_audio": str(raw_output) if status == "ok" else None,
            "seconds": seconds,
            "status": status,
            "error": error,
            "teacher_model": "CosyVoice2-0.5B",
            "teacher_pack_id": args.pack_id,
            "teacher_ref_audio": pack["pack_audio"],
            "teacher_ref_text": pack["ref_text"],
        }
        manifest.append(item)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(item_id, status, f"{seconds:.2f}s")

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = [item for item in manifest if item["status"] == "ok"]
    print(manifest_path)
    print(f"ok={len(ok)} total={len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
