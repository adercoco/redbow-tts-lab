#!/usr/bin/env python3
"""Convert a Piper ONNX model to float16 for mobile runtime experiments."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16


def human_size(path: Path) -> str:
    size = float(path.stat().st_size)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{path.stat().st_size}B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-config", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(args.model)
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, args.output)

    print(f"input={args.model}")
    print(f"input_size={human_size(args.model)}")
    print(f"output={args.output}")
    print(f"output_size={human_size(args.output)}")

    if args.copy_config:
        source_config = args.model.with_suffix(args.model.suffix + ".json")
        output_config = args.output.with_suffix(args.output.suffix + ".json")
        shutil.copy2(source_config, output_config)
        print(f"config={output_config}")
        print(f"config_size={human_size(output_config)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
