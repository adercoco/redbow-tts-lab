#!/usr/bin/env python3
"""CPU-compatible CosyVoice speech-token extraction.

The upstream helper hard-codes CUDAExecutionProvider. This local copy keeps the
same output format but runs on macOS CPU for dataset preparation.
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torchaudio
import whisper
from tqdm import tqdm


def read_wav_scp(data_dir: Path) -> dict[str, str]:
    utt2wav = {}
    for line in (data_dir / "wav.scp").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt, wav = line.split(maxsplit=1)
        utt2wav[utt] = wav
    return utt2wav


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--num_thread", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(args.dir)
    utt2wav = read_wav_scp(data_dir)

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(
        args.onnx_path,
        sess_options=option,
        providers=["CPUExecutionProvider"],
    )

    def single_job(utt: str) -> tuple[str, list[int]]:
        audio, sample_rate = torchaudio.load(utt2wav[utt], backend="soundfile")
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.shape[1] / 16000 > 30:
            logging.warning("skip %s: audio longer than 30s", utt)
            return utt, []
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        token = ort_session.run(
            None,
            {
                ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
            },
        )[0].flatten().tolist()
        return utt, token

    utt2speech_token = {}
    with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
        futures = [executor.submit(single_job, utt) for utt in utt2wav]
        for future in tqdm(as_completed(futures), total=len(futures)):
            utt, speech_token = future.result()
            utt2speech_token[utt] = speech_token

    torch.save(utt2speech_token, data_dir / "utt2speech_token.pt")
    print(f"wrote {data_dir / 'utt2speech_token.pt'}")


if __name__ == "__main__":
    main()
