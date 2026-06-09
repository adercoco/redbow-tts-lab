#!/usr/bin/env python3
"""Generate one Qwen3-TTS 0.6B Base clone wav for the web voice changer."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_audio.tts.generate import generate_audio, load_model


MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output.parent / f".{args.output.stem}_qwen_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(MODEL_ID)
    generate_audio(
        text=args.text,
        model=model,
        lang_code="zh",
        ref_audio=str(args.ref_audio),
        ref_text=args.ref_text,
        output_path=str(temp_dir),
        file_prefix=args.output.stem,
        audio_format="wav",
        verbose=False,
    )
    generated = temp_dir / f"{args.output.stem}_000.wav"
    if not generated.exists():
        raise FileNotFoundError(f"Qwen finished but did not create {generated}")
    generated.replace(args.output)
    for item in temp_dir.glob("*"):
        item.unlink(missing_ok=True)
    temp_dir.rmdir()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
