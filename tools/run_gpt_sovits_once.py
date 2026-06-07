#!/usr/bin/env python3
"""Run one GPT-SoVITS inference with a macOS-safe wav loader."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external" / "GPT-SoVITS"


def soundfile_load(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T), sr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_model", required=True)
    parser.add_argument("--sovits_model", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--target_text", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "GPT_SoVITS"))
    os.environ.setdefault("is_half", "False")
    os.environ.setdefault("version", "v2")

    torchaudio.load = soundfile_load

    from tools.i18n.i18n import I18nAuto
    from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

    i18n = I18nAuto()
    ref_text = Path(args.ref_text).read_text(encoding="utf-8")
    target_text = Path(args.target_text).read_text(encoding="utf-8")

    change_gpt_weights(gpt_path=args.gpt_model)
    change_sovits_weights(sovits_path=args.sovits_model)
    result = list(
        get_tts_wav(
            ref_wav_path=args.ref_audio,
            prompt_text=ref_text,
            prompt_language=i18n("中文"),
            text=target_text,
            text_language=i18n("中文"),
            top_p=1,
            temperature=1,
        )
    )
    if not result:
        raise RuntimeError("GPT-SoVITS returned no audio")

    sr, audio = result[-1]
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(output_dir / "output.wav", audio, sr)
    print(output_dir / "output.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
