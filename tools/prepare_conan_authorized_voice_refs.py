#!/usr/bin/env python3
"""Prepare authorized Conan character voice references for TTS cloning."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(
    "/Users/ader/Documents/Codex/2026-05-23/youtube/extracted_audio/"
    "conan_official_voice/verified"
)
OUT = ROOT / "distillation" / "conan_authorized_voice_refs_v1"
CLIPS = OUT / "clips_24k"
PACKS = OUT / "reference_packs"

ROLES = ["haibara", "conan", "agasa"]


def rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    return 20 * math.log10(rms + 1e-12)


def peak_db(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return 20 * math.log10(peak + 1e-12)


def silence_ratio(audio: np.ndarray, threshold_db: float = -45.0) -> float:
    if audio.size == 0:
        return 1.0
    frame = 1024
    hop = 512
    values = []
    for start in range(0, max(1, audio.size - frame + 1), hop):
        chunk = audio[start : start + frame]
        values.append(rms_db(chunk))
    if not values:
        return 1.0
    return sum(1 for value in values if value < threshold_db) / len(values)


def normalize(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    audio = audio.astype(np.float32)
    audio = audio - float(np.mean(audio))
    current = rms_db(audio)
    gain = 10 ** ((target_db - current) / 20)
    audio = audio * min(gain, 8.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0
    if peak > 0.96:
        audio = audio * (0.96 / peak)
    return np.clip(audio, -0.98, 0.98).astype(np.float32)


def text_from_whisper_json(path: Path) -> str:
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    return re.sub(r"\s+", "", str(data.get("text", "")).strip())


def transcribe(wavs: list[Path]) -> None:
    if not wavs:
        return
    missing = [wav for wav in wavs if not (wav.with_suffix(".json")).exists()]
    if not missing:
        return
    import whisper

    model = whisper.load_model("small")
    for wav in missing:
        audio, _ = librosa.load(str(wav), sr=16_000, mono=True)
        result = model.transcribe(
            audio.astype(np.float32),
            language="zh",
            task="transcribe",
            fp16=False,
            condition_on_previous_text=False,
            initial_prompt="以下是中文配音的角色台词。",
        )
        text = re.sub(r"\s+", "", str(result.get("text", "")).strip())
        payload = {
            "text": text,
            "language": "zh",
            "model": "openai-whisper-small",
            "source_wav": str(wav),
        }
        wav.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"transcribed {wav.name}: {text}")


def prepare_clips() -> list[dict[str, object]]:
    CLIPS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    wavs: list[Path] = []
    for role in ROLES:
        role_dir = SRC_ROOT / role
        for source in sorted(role_dir.glob("*.wav")):
            audio, sr = librosa.load(str(source), sr=24_000, mono=True)
            duration = float(audio.size / 24_000)
            audio = normalize(audio)
            clip = CLIPS / f"{role}_{source.stem}.wav"
            sf.write(str(clip), audio, 24_000, subtype="PCM_16")
            wavs.append(clip)
            rows.append(
                {
                    "role": role,
                    "source": str(source),
                    "clip": str(clip),
                    "duration": round(duration, 3),
                    "rms_db": round(rms_db(audio), 2),
                    "peak_db": round(peak_db(audio), 2),
                    "silence_ratio": round(silence_ratio(audio), 3),
                }
            )
    transcribe(wavs)
    for row in rows:
        clip = Path(str(row["clip"]))
        row["transcript"] = text_from_whisper_json(clip.with_suffix(".json"))
    return rows


def make_pack_audio(role: str, rows: list[dict[str, object]]) -> dict[str, object]:
    role_rows = [
        row
        for row in rows
        if row["role"] == role
        and float(row["duration"]) >= 2.0
        and float(row["duration"]) <= 9.5
        and str(row.get("transcript", "")).strip()
    ]
    role_rows.sort(
        key=lambda row: (
            abs(float(row["duration"]) - 4.5),
            float(row["silence_ratio"]),
        )
    )
    selected = role_rows[:3]
    if not selected:
        raise RuntimeError(f"No usable reference clips for {role}")

    chunks = []
    texts = []
    files = []
    silence = np.zeros(int(0.22 * 24_000), dtype=np.float32)
    for row in selected:
        audio, sr = sf.read(str(row["clip"]), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        chunks.extend([audio.astype(np.float32), silence])
        texts.append(str(row["transcript"]))
        files.append(str(row["clip"]))
    pack_audio = np.concatenate(chunks).astype(np.float32)
    pack_audio = normalize(pack_audio, target_db=-20.0)
    PACKS.mkdir(parents=True, exist_ok=True)
    pack_path = PACKS / f"{role}_pack_best3_24k.wav"
    sf.write(str(pack_path), pack_audio, 24_000, subtype="PCM_16")
    return {
        "role": role,
        "pack_id": f"{role}_best3",
        "pack_audio": str(pack_path),
        "ref_text": " ".join(texts),
        "ref_files": files,
        "duration": round(float(pack_audio.size / 24_000), 3),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = prepare_clips()
    packs = [make_pack_audio(role, rows) for role in ROLES]
    (OUT / "clip_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (PACKS / "manifest.json").write_text(
        json.dumps(packs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (OUT / "clip_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT / "clip_manifest.json")
    print(PACKS / "manifest.json")
    for pack in packs:
        print(pack["role"], pack["duration"], pack["ref_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
