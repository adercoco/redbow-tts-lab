#!/usr/bin/env python3
"""Score Cosy/ZipVoice comparison outputs with CAMPPlus speaker embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "distillation" / "taiwan_mandarin_low_r" / "datasets" / "downloads_female_voice"
OUT = BASE / "cosy_zipvoice_distill_v1"
ONNX = ROOT / "external" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B" / "campplus.onnx"


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


def score_file(result_path: Path, reference_key: str = "ref_audio") -> list[dict]:
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(str(ONNX), sess_options=option, providers=["CPUExecutionProvider"])
    cache: dict[str, torch.Tensor] = {}

    def get(path_value: str) -> torch.Tensor | None:
        path = Path(path_value)
        if not path.exists():
            return None
        key = str(path)
        if key not in cache:
            cache[key] = embed(session, path)
        return cache[key]

    rows = json.loads(result_path.read_text(encoding="utf-8"))
    for row in rows:
        row["speaker_cosine"] = None
        row["speaker_score_note"] = "not_scored"
        if row.get("status") != "ok":
            continue
        ref = get(str(row.get(reference_key, "")))
        out = get(str(row.get("output", "")))
        if ref is None or out is None:
            continue
        row["speaker_cosine"] = round(torch.dot(ref, out).item(), 4)
        row["speaker_score_note"] = f"campplus_cosine_{reference_key}_to_output"
    result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main() -> int:
    all_rows: list[dict] = []
    cosy = OUT / "cosy_sweep_results.json"
    if cosy.exists():
        all_rows.extend(score_file(cosy, "ref_audio"))
    zipvoice = OUT / "zipvoice_compare_results.json"
    if zipvoice.exists():
        all_rows.extend(score_file(zipvoice, "teacher_audio"))
    merged = OUT / "all_results_scored.json"
    merged.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(merged)
    print(f"rows={len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
