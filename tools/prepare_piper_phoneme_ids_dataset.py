#!/usr/bin/env python3
"""Prepare a Piper phoneme-id dataset with a lightweight pypinyin frontend."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from pypinyin import Style, pinyin

from piper.phonemize_chinese import (
    PHONEME_TO_ID,
    _normalize_g2pw_syllable,
    _split_initial_final_tone,
    phonemes_to_ids,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
DEFAULT_SOURCE = BASE / "datasets" / "piper_ljspeech_distill_v1"
DEFAULT_OUT = BASE / "datasets" / "piper_phoneme_ids_distill_v1"

PINYIN_RE = re.compile(r"^[a-züv:]+[1-5]$")


def clean_text(text: str) -> str:
    return " ".join(text.replace("|", " ").split())


def text_to_phonemes(text: str) -> list[str]:
    phonemes: list[str] = []
    for char in text:
        if char.isspace():
            continue

        if char in PHONEME_TO_ID:
            phonemes.append(char)
            continue

        py = pinyin(
            char,
            style=Style.TONE3,
            heteronym=False,
            neutral_tone_with_five=True,
            errors=lambda chars: list(chars),
        )[0][0]
        py = _normalize_g2pw_syllable(py)
        if not PINYIN_RE.match(py):
            continue

        initial, final, tone = _split_initial_final_tone(py)
        if not final or tone is None:
            continue
        if not initial:
            initial = "Ø"
        phonemes.extend([initial, final, tone])

    return phonemes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else ROOT / args.source
    out = args.out if args.out.is_absolute() else ROOT / args.out
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in (source / "metadata.csv").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt_id, text = line.split("|", 1)
        rows.append((utt_id, clean_text(text)))
        if args.limit and len(rows) >= args.limit:
            break

    metadata_lines = []
    manifest_rows = []
    max_id = 0
    for utt_id, text in rows:
        src = source / "wavs" / f"{utt_id}.wav"
        dst = wav_dir / f"{utt_id}.wav"
        if not src.exists():
            raise FileNotFoundError(src)
        if not dst.exists():
            shutil.copy2(src, dst)

        phonemes = text_to_phonemes(text)
        ids = phonemes_to_ids(phonemes, PHONEME_TO_ID)
        max_id = max(max_id, max(ids))
        metadata_lines.append(f"{utt_id}|{text}|{' '.join(str(i) for i in ids)}")
        manifest_rows.append(
            {
                "id": utt_id,
                "text": text,
                "wav": f"wavs/{utt_id}.wav",
                "phonemes": phonemes,
                "phoneme_ids": ids,
            }
        )

    (out / "metadata.csv").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    (out / "phonemes.json").write_text(
        json.dumps(PHONEME_TO_ID, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "items": len(manifest_rows),
                "frontend": "pypinyin tone3 -> Piper Chinese pinyin phoneme ids",
                "num_symbols": max_id + 1,
                "rows": manifest_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# Piper Phoneme-ID Distillation Dataset\n\n"
        "Uses a lightweight pypinyin frontend instead of g2pW so the student "
        "can be trained and deployed with direct phoneme ids.\n\n"
        f"- Items: {len(manifest_rows)}\n"
        f"- Num symbols: {max_id + 1}\n",
        encoding="utf-8",
    )
    print(out)
    print(f"items={len(manifest_rows)}")
    print(f"num_symbols={max_id + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
