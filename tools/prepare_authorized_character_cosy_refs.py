#!/usr/bin/env python3
"""Prepare CosyVoice zero-shot reference packs for authorized character clips."""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import whisper


ROOT = Path("/Users/ader/Documents/App")
SRC_ROOT = Path(
    "/Users/ader/Documents/Codex/2026-05-23/youtube/extracted_audio/character_voice_collection/verified"
)
OUT = ROOT / "distillation/character_voice_collection_refs_v1"
CLIPS = OUT / "clips_24k"
PACKS = OUT / "reference_packs"
SAMPLE_RATE = 24_000

ROLES = {
    "ryotsu": {
        "label": "兩津勘吉",
        "src": SRC_ROOT / "ryotsu",
    },
    "shinchan": {
        "label": "野原新之助",
        "src": SRC_ROOT / "shinchan",
    },
    "misae": {
        "label": "野原美牙",
        "src": SRC_ROOT / "misae",
    },
}


def rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    return 20 * np.log10(rms + 1e-12)


def normalize(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    audio = audio.astype(np.float32)
    audio = audio - float(np.mean(audio))
    gain = 10 ** ((target_db - rms_db(audio)) / 20)
    audio = audio * min(gain, 8.0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.96:
        audio = audio * (0.96 / peak)
    return np.clip(audio, -0.98, 0.98).astype(np.float32)


def transcribe(model, path: Path) -> dict:
    audio_16k, _ = librosa.load(path, sr=16_000, mono=True)
    result = model.transcribe(
        audio_16k.astype(np.float32),
        language="zh",
        task="transcribe",
        fp16=False,
        verbose=False,
    )
    text = " ".join(seg.get("text", "").strip() for seg in result.get("segments", []))
    segments = result.get("segments", [])
    avg_logprob = (
        float(np.mean([seg.get("avg_logprob", -10.0) for seg in segments]))
        if segments
        else -10.0
    )
    no_speech_prob = (
        float(np.mean([seg.get("no_speech_prob", 1.0) for seg in segments]))
        if segments
        else 1.0
    )
    return {
        "transcript": text.strip(),
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
    }


def quality_score(row: dict) -> float:
    duration = row["duration"]
    duration_score = 1.0 - min(abs(duration - 4.0) / 6.0, 1.0)
    speech_score = 1.0 - min(row["no_speech_prob"], 1.0)
    logprob_score = max(min((row["avg_logprob"] + 1.2) / 1.2, 1.0), 0.0)
    return 0.45 * logprob_score + 0.35 * speech_score + 0.20 * duration_score


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CLIPS.mkdir(parents=True, exist_ok=True)
    PACKS.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model("small")
    manifest = []
    packs = []

    for role, info in ROLES.items():
        role_rows = []
        for src in sorted(info["src"].glob("*.wav")):
            audio, _ = librosa.load(src, sr=SAMPLE_RATE, mono=True)
            audio = normalize(audio, -20.0)
            clip_name = f"{role}_{src.stem}.wav"
            clip_path = CLIPS / clip_name
            sf.write(clip_path, audio, SAMPLE_RATE, subtype="PCM_16")
            duration = round(float(audio.size / SAMPLE_RATE), 3)
            asr = transcribe(model, clip_path)
            row = {
                "role": role,
                "label": info["label"],
                "source": str(src),
                "clip": str(clip_path),
                "duration": duration,
                "rms_db": round(rms_db(audio), 2),
                "peak_db": round(20 * np.log10(float(np.max(np.abs(audio))) + 1e-12), 2),
                **asr,
            }
            row["quality_score"] = round(quality_score(row), 4)
            role_rows.append(row)
            manifest.append(row)

        candidates = [
            row
            for row in role_rows
            if 2.0 <= row["duration"] <= 6.5 and len(row["transcript"]) >= 2
        ]
        selected = sorted(candidates, key=lambda row: row["quality_score"], reverse=True)[:3]
        selected = sorted(selected, key=lambda row: row["source"])
        if not selected:
            selected = role_rows[:1]
        chunks = []
        ref_texts = []
        for row in selected:
            audio, _ = librosa.load(row["clip"], sr=SAMPLE_RATE, mono=True)
            chunks.append(audio)
            chunks.append(np.zeros(int(0.18 * SAMPLE_RATE), dtype=np.float32))
            ref_texts.append(row["transcript"])
        pack_audio = normalize(np.concatenate(chunks), -20.0)
        pack_path = PACKS / f"{role}_best3_24k.wav"
        sf.write(pack_path, pack_audio, SAMPLE_RATE, subtype="PCM_16")
        packs.append(
            {
                "role": role,
                "label": info["label"],
                "pack_id": f"{role}_best3",
                "pack_audio": str(pack_path),
                "ref_text": " ".join(ref_texts).strip(),
                "ref_files": [row["clip"] for row in selected],
                "source_files": [row["source"] for row in selected],
                "duration": round(float(pack_audio.size / SAMPLE_RATE), 3),
                "selected_rows": selected,
            }
        )

    (OUT / "clip_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (PACKS / "manifest.json").write_text(
        json.dumps(packs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"clips": len(manifest), "packs": packs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
