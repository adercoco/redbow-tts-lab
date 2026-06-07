#!/usr/bin/env python3
"""Quantize a Piper ONNX model for mobile runtime experiments."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


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
    parser.add_argument(
        "--weight-type",
        choices=("qint8", "quint8"),
        default="qint8",
        help="Dynamic weight quantization type.",
    )
    parser.add_argument(
        "--copy-config",
        action="store_true",
        help="Copy <model>.json next to the quantized model as <output>.json.",
    )
    parser.add_argument(
        "--op-types",
        default=None,
        help="Comma-separated ONNX op types to quantize, for example MatMul,Gemm. "
        "Leave unset to use onnxruntime's default dynamic quantization set.",
    )
    args = parser.parse_args()

    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8
    args.output.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(args.model),
        model_output=str(args.output),
        weight_type=weight_type,
        op_types_to_quantize=(
            [op.strip() for op in args.op_types.split(",") if op.strip()]
            if args.op_types
            else None
        ),
    )

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
