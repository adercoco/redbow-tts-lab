#!/usr/bin/env python3
"""Build a sherpa-onnx compatible Matcha pack from our fine-tuned model.

The app needs a self-contained directory with:
- Matcha acoustic ONNX carrying sherpa metadata
- HiFi-GAN vocoder ONNX carrying sherpa metadata
- tokens.txt matching Matcha's symbol IDs
- lexicon.txt that maps pinyin characters to those token IDs
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import onnx
import torch

from matcha.cli import load_vocoder
from matcha.text.symbols import symbols


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDENT = (
    ROOT
    / "distillation"
    / "taiwan_mandarin_low_r"
    / "students"
    / "matcha_cosy_golden_pinyin_v1_step15000"
)
DEFAULT_VOCODER = ROOT / "models" / "matcha" / "hifigan_T2_v1"


def add_meta_data(filename: Path, meta_data: dict[str, Any]) -> None:
    model = onnx.load(str(filename))
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, str(filename))


def write_tokens(path: Path) -> None:
    lines: list[str] = []
    for idx, token in enumerate(symbols):
        if token == " ":
            lines.append(str(idx))
        else:
            lines.append(f"{token} {idx}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pua_for_ascii(ch: str) -> str:
    """Map one ASCII pinyin character to a private-use UTF-8 character.

    sherpa's CharacterLexicon groups contiguous ASCII letters into a word. Our
    Matcha checkpoint was trained on compact pinyin characters, so we need the
    frontend to split every pinyin character independently without adding space
    tokens. Private-use Unicode characters give us that split while staying
    reversible in the app.
    """

    return chr(0xE000 + ord(ch))


def write_pinyin_lexicon(path: Path) -> None:
    supported = set("abcdefghijklmnopqrstuvwxyz12345")
    lines = [f"{pua_for_ascii(ch)} {ch}" for ch in sorted(supported)]
    lines.extend(f"{ch} {ch}" for ch in [",", ".", ":", ";", "!", "?"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_hifigan(vocoder_path: Path, output: Path, dummy_frames: int) -> None:
    vocoder, _denoiser = load_vocoder("hifigan_T2_v1", str(vocoder_path), torch.device("cpu"))
    vocoder.eval()
    dummy_mel = torch.randn(1, 80, dummy_frames, dtype=torch.float32)
    torch.onnx.export(
        vocoder,
        dummy_mel,
        str(output),
        input_names=["mel"],
        output_names=["audio"],
        dynamic_axes={
            "mel": {0: "batch_size", 2: "mel_time"},
            "audio": {0: "batch_size", 2: "audio_time"},
        },
        opset_version=15,
        export_params=True,
        do_constant_folding=True,
        dynamo=False,
    )
    add_meta_data(
        output,
        {
            "model_type": "hifigan",
            "sample_rate": 22050,
            "hop_length": 256,
            "n_fft": 1024,
            "win_length": 1024,
            "num_mels": 80,
            "version": 1,
            "maintainer": "redbow-local",
            "source": str(vocoder_path),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-dir", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--steps", type=int, default=8, choices=(8, 16))
    parser.add_argument("--vocoder", type=Path, default=DEFAULT_VOCODER)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "RedBowVoice" / "RedBowAssets" / "Models" / "MatchaCosyGolden",
    )
    parser.add_argument("--dummy-vocoder-frames", type=int, default=160)
    args = parser.parse_args()

    source_acoustic = args.student_dir / "onnx_mobile" / f"matcha_cosy_golden_steps{args.steps}.onnx"
    if not source_acoustic.exists():
        raise FileNotFoundError(source_acoustic)
    if not args.vocoder.exists():
        raise FileNotFoundError(args.vocoder)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    acoustic = args.out_dir / f"matcha_cosy_golden_steps{args.steps}.onnx"
    vocoder = args.out_dir / "hifigan_T2_v1.onnx"

    shutil.copy2(source_acoustic, acoustic)
    add_meta_data(
        acoustic,
        {
            "model_type": "matcha-tts",
            "language": "pinyin Mandarin",
            "voice": "zh",
            "jieba": 1,
            "has_espeak": 0,
            "n_speakers": 1,
            "sample_rate": 22050,
            "version": 1,
            "pad_id": 0,
            "use_eos_bos": 0,
            "num_ode_steps": args.steps,
            "frontend": "Chinese text must be converted to compact pinyin before calling sherpa",
        },
    )

    write_tokens(args.out_dir / "tokens.txt")
    write_pinyin_lexicon(args.out_dir / "lexicon.txt")
    export_hifigan(args.vocoder, vocoder, args.dummy_vocoder_frames)

    pack = {
        "id": "matcha-cosy-golden-pinyin-v1",
        "displayName": f"CosyVoice2 Golden Teacher -> Matcha {args.steps}-step",
        "backend": "sherpa-onnx matcha",
        "available": True,
        "acousticModel": acoustic.name,
        "vocoder": vocoder.name,
        "tokens": "tokens.txt",
        "lexicon": "lexicon.txt",
        "numSteps": args.steps,
        "sampleRate": 22050,
        "temperature": 0.5,
        "speed": 1.18,
        "textFrontend": "compact-pinyin",
        "notes": "Fine-tuned Matcha student from CosyVoice2 golden teacher corpus. iOS converts Chinese text to compact pinyin, then sherpa-onnx Matcha generates mel and HiFi-GAN vocodes it.",
        "pinyinPrivateUseBase": "U+E000",
    }
    (args.out_dir / "model-pack.json").write_text(
        json.dumps(pack, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(args.out_dir)
    print(json.dumps(pack, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
