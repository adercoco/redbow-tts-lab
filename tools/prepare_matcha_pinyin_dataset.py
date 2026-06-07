#!/usr/bin/env python3
"""Prepare a pinyin Matcha-TTS smoke dataset from a teacher manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as taf
from pypinyin import Style, pinyin

from matcha.utils.audio import mel_spectrogram


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r"
DEFAULT_MANIFEST = BASE / "teacher_indextts2_distill_smoke_v1" / "manifest.json"
DEFAULT_OUT = BASE / "datasets" / "matcha_pinyin_indextts2_smoke_v1"
ALLOWED_RE = re.compile(r"[^a-zA-Z0-9;:,.!? ]+")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def text_to_pinyin_ascii(text: str, tones: bool = True) -> str:
    tokens: list[str] = []
    for char in text:
        if char.isspace():
            tokens.append(" ")
            continue
        if char in "，,。.!！?？":
            tokens.append({ "，": ",", "。": ".", "！": "!", "？": "?" }.get(char, char))
            continue
        style = Style.TONE3 if tones else Style.NORMAL
        py = pinyin(
            char,
            style=style,
            heteronym=False,
            neutral_tone_with_five=True,
            errors=lambda chars: list(chars),
        )[0][0]
        py = py.replace("ü", "v")
        py = ALLOWED_RE.sub("", py)
        if py:
            tokens.append(py)
            tokens.append(" ")
    out = " ".join("".join(tokens).split())
    return out.lower()


def wav_to_matcha_mel(path: Path) -> torch.Tensor:
    audio_np, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio_np.T)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != 22050:
        audio = taf.resample(audio, sample_rate, 22050)
    return mel_spectrogram(
        audio,
        n_fft=1024,
        num_mels=80,
        sampling_rate=22050,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=8000,
        center=False,
    ).squeeze(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--valid-items", type=int, default=2)
    parser.add_argument("--no-tones", action="store_true")
    args = parser.parse_args()

    manifest = resolve(args.manifest)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [row for row in json.loads(manifest.read_text(encoding="utf-8")) if row.get("status") == "ok"]
    prepared = []
    for row in rows:
        wav = resolve(row["audio"])
        if not wav.exists():
            raise FileNotFoundError(wav)
        pinyin_text = text_to_pinyin_ascii(row["text"], tones=not args.no_tones)
        prepared.append(
            {
                "id": row["id"],
                "wav": str(wav),
                "text": row["text"],
                "pinyin_text": pinyin_text,
                "teacher_seconds": row.get("seconds"),
            }
        )

    valid_count = min(args.valid_items, max(1, len(prepared) // 8))
    train_rows = prepared[:-valid_count]
    valid_rows = prepared[-valid_count:]

    (out / "train.txt").write_text(
        "\n".join(f"{row['wav']}|{row['pinyin_text']}" for row in train_rows) + "\n",
        encoding="utf-8",
    )
    (out / "valid.txt").write_text(
        "\n".join(f"{row['wav']}|{row['pinyin_text']}" for row in valid_rows) + "\n",
        encoding="utf-8",
    )

    total_sum = torch.tensor(0.0)
    total_sq_sum = torch.tensor(0.0)
    total_len = torch.tensor(0.0)
    for row in prepared:
        mel = wav_to_matcha_mel(Path(row["wav"]))
        total_sum += mel.sum()
        total_sq_sum += torch.pow(mel, 2).sum()
        total_len += mel.shape[0] * mel.shape[1]
    mel_mean = total_sum / total_len
    mel_std = torch.sqrt(total_sq_sum / total_len - torch.pow(mel_mean, 2))

    info = {
        "source_manifest": str(manifest),
        "items": len(prepared),
        "train_items": len(train_rows),
        "valid_items": len(valid_rows),
        "text_frontend": (
            "Chinese text -> pypinyin Style.TONE3 -> ASCII letters plus tone digits"
            if not args.no_tones
            else "Chinese text -> pypinyin Style.NORMAL -> ASCII letters, no tones"
        ),
        "sample_rate": 22050,
        "data_statistics": {"mel_mean": float(mel_mean), "mel_std": float(mel_std)},
        "rows": prepared,
    }
    (out / "dataset_manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Matcha Pinyin IndexTTS2 Smoke Dataset\n\n"
        "This is a smoke dataset for testing Matcha/FastSpeech-style non-AR students. "
        "It intentionally romanizes Chinese text to pinyin because the upstream Matcha "
        "symbol table is ASCII/IPA oriented.\n\n"
        f"- Items: {len(prepared)}\n"
        f"- Train: {len(train_rows)}\n"
        f"- Valid: {len(valid_rows)}\n"
        f"- Mel mean/std: {float(mel_mean):.6f} / {float(mel_std):.6f}\n",
        encoding="utf-8",
    )
    print(out)
    print(json.dumps(info["data_statistics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
