#!/usr/bin/env python3
"""Prepare louder/clearer reference packs for Cosy -> ZipVoice comparison."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
REF_DIR = BASE / "clean_strict_24k"
OUT = BASE / "cosy_zipvoice_distill_v1"
REF_OUT = OUT / "references"

PACK_DEFS = [
    ("raw_best2_7s", [0, 1], "原始 best2 reference，未做音色處理"),
    ("clear_best2_7s", [0, 1], "best2 reference，做高通、清晰頻段微增益、RMS 放大"),
    ("clear_best3_11s", [0, 1, 5], "best3 reference，做同樣清晰化；測試更長 reference 是否更穩"),
]


def load_rows() -> list[dict[str, str]]:
    return list(csv.DictReader((REF_DIR / "metadata.csv").open(encoding="utf-8")))


def load_mono(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.transforms.Resample(sr, 24000)(wav)
        sr = 24000
    return wav, sr


def enhance(wav: torch.Tensor, sr: int) -> torch.Tensor:
    wav = wav - wav.mean()
    wav = F.highpass_biquad(wav, sr, cutoff_freq=75)
    wav = F.lowpass_biquad(wav, sr, cutoff_freq=8600)

    spec = torch.fft.rfft(wav)
    freqs = torch.fft.rfftfreq(wav.shape[-1], d=1 / sr).to(wav.device)
    presence = torch.exp(-0.5 * ((freqs - 3200) / 1450) ** 2)
    mud_cut = torch.exp(-0.5 * ((freqs - 180) / 120) ** 2)
    weight = 1.0 + 0.22 * presence - 0.10 * mud_cut
    wav = torch.fft.irfft(spec * weight, n=wav.shape[-1])

    rms = torch.sqrt(torch.mean(wav**2)).clamp_min(1e-8)
    target = 10 ** (-21.0 / 20.0)
    wav = wav * (target / rms)
    peak = wav.abs().max().clamp_min(1e-8)
    if peak > 0.96:
        wav = wav * (0.96 / peak)
    return wav.clamp(-0.98, 0.98)


def stats(wav: torch.Tensor) -> dict[str, float]:
    rms = torch.sqrt(torch.mean(wav**2)).item()
    peak = wav.abs().max().item()
    return {
        "rms_db": round(20 * math.log10(max(rms, 1e-8)), 2),
        "peak_db": round(20 * math.log10(max(peak, 1e-8)), 2),
    }


def build_pack(rows: list[dict[str, str]], pack_id: str, indexes: list[int], note: str) -> dict:
    audio_parts: list[torch.Tensor] = []
    ref_files: list[str] = []
    ref_texts: list[str] = []
    before_stats: list[dict[str, float]] = []
    after_stats: list[dict[str, float]] = []
    silence = torch.zeros(1, int(0.18 * 24000))

    for idx in indexes:
        row = rows[idx]
        path = REF_DIR / row["file"]
        wav, sr = load_mono(path)
        before_stats.append(stats(wav))
        if pack_id.startswith("clear_"):
            wav = enhance(wav, sr)
        after_stats.append(stats(wav))
        audio_parts.append(wav)
        audio_parts.append(silence)
        ref_files.append(row["file"])
        ref_texts.append(row["transcript"].strip())

    combined = torch.cat(audio_parts[:-1], dim=1)
    output = REF_OUT / f"{pack_id}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), combined, 24000)
    return {
        "pack_id": pack_id,
        "pack_audio": str(output),
        "ref_text": " ".join(text for text in ref_texts if text),
        "ref_files": ref_files,
        "note": note,
        "duration": round(combined.shape[-1] / 24000, 2),
        "before_stats": before_stats,
        "after_stats": after_stats,
        "pack_stats": stats(combined),
    }


def main() -> int:
    rows = load_rows()
    REF_OUT.mkdir(parents=True, exist_ok=True)
    manifest = [build_pack(rows, pack_id, indexes, note) for pack_id, indexes, note in PACK_DEFS]
    path = REF_OUT / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    for item in manifest:
        print(item["pack_id"], item["duration"], item["pack_stats"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
