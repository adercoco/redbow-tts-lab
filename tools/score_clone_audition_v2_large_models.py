#!/usr/bin/env python3
"""Score v2 clone audition outputs with CAMPPlus speaker embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
OUT = BASE / "clone_audition_v2_large_models"
ONNX = ROOT / "external" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B" / "campplus.onnx"

RESULT_FILES = [
    OUT / "reference_pack_results.json",
    OUT / "gpt_sovits_aux_results.json",
    OUT / "indextts2_pack_results.json",
    OUT / "f5_finetuned_40u_results.json",
]


def load_audio(path: Path) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    return audio


def embed(session: onnxruntime.InferenceSession, path: Path) -> torch.Tensor:
    audio = load_audio(path)
    feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = session.run(None, {session.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()
    vector = torch.tensor(embedding, dtype=torch.float32)
    return torch.nn.functional.normalize(vector, dim=0)


def main() -> int:
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(ONNX), sess_options=option, providers=["CPUExecutionProvider"])

    cache: dict[str, torch.Tensor] = {}

    def get(path_value: str) -> torch.Tensor | None:
        if not path_value:
            return None
        path = Path(path_value)
        if not path.exists():
            return None
        key = str(path)
        if key not in cache:
            cache[key] = embed(session, path)
        return cache[key]

    total = 0
    scored = 0
    for result_file in RESULT_FILES:
        rows = json.loads(result_file.read_text(encoding="utf-8"))
        for row in rows:
            total += 1
            row["speaker_cosine"] = None
            row["speaker_cosine_primary"] = None
            row["speaker_cosine_avg"] = None
            row["speaker_cosine_max"] = None
            row["speaker_score_note"] = "not_scored"
            if row.get("status") != "ok":
                continue
            out = get(str(row.get("output", "")))
            ref_values = [str(row.get("ref_audio", ""))]
            ref_values.extend(str(path) for path in row.get("aux_ref_audio_paths", []))
            refs = [ref for ref in (get(path) for path in ref_values) if ref is not None]
            if not refs or out is None:
                continue
            scores = [torch.dot(ref, out).item() for ref in refs]
            row["speaker_cosine_primary"] = round(scores[0], 4)
            row["speaker_cosine_avg"] = round(sum(scores) / len(scores), 4)
            row["speaker_cosine_max"] = round(max(scores), 4)
            row["speaker_cosine"] = row["speaker_cosine_max"]
            row["speaker_score_note"] = "campplus_cosine_max_over_reference_pool"
            scored += 1
        result_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    merged: list[dict] = []
    for result_file in RESULT_FILES:
        merged.extend(json.loads(result_file.read_text(encoding="utf-8")))
    merged_path = OUT / "all_results_scored.json"
    merged_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"scored={scored}/{total}")
    print(merged_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
