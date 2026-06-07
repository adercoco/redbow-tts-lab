#!/usr/bin/env python3
"""Static QDQ quantization for Piper ONNX using phoneme-id calibration rows."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


def human_size(path: Path) -> str:
    size = float(path.stat().st_size)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{path.stat().st_size}B"


class PiperCalibrationReader(CalibrationDataReader):
    def __init__(self, metadata_csv: Path, limit: int) -> None:
        self.items: list[dict[str, np.ndarray]] = []
        for line in metadata_csv.read_text().splitlines():
            if not line.strip():
                continue

            parts = line.split("|")
            if len(parts) != 3:
                continue

            ids = [int(item) for item in parts[2].split()]
            ids_array = np.expand_dims(np.array(ids, dtype=np.int64), 0)
            self.items.append(
                {
                    "input": ids_array,
                    "input_lengths": np.array([ids_array.shape[1]], dtype=np.int64),
                    "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
                }
            )
            if len(self.items) >= limit:
                break

        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self.items):
            return None
        item = self.items[self._index]
        self._index += 1
        return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--copy-config", action="store_true")
    parser.add_argument(
        "--op-types",
        default="Conv,MatMul,Gemm",
        help="Comma-separated ONNX op types to quantize.",
    )
    parser.add_argument(
        "--exclude-output-op-types",
        default="Softmax",
        help="Comma-separated op types whose outputs should not be quantized.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reader = PiperCalibrationReader(args.metadata_csv, args.limit)
    op_types = [op.strip() for op in args.op_types.split(",") if op.strip()]
    exclude_output_op_types = [
        op.strip() for op in args.exclude_output_op_types.split(",") if op.strip()
    ]

    quantize_static(
        model_input=str(args.model),
        model_output=str(args.output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types,
        per_channel=True,
        extra_options={
            "OpTypesToExcludeOutputQuantization": exclude_output_op_types,
        },
    )

    print(f"input={args.model}")
    print(f"input_size={human_size(args.model)}")
    print(f"output={args.output}")
    print(f"output_size={human_size(args.output)}")
    print(f"calibration_items={len(reader.items)}")
    print(f"op_types={','.join(op_types)}")
    print(f"exclude_output_op_types={','.join(exclude_output_op_types)}")

    if args.copy_config:
        source_config = args.model.with_suffix(args.model.suffix + ".json")
        output_config = args.output.with_suffix(args.output.suffix + ".json")
        shutil.copy2(source_config, output_config)
        print(f"config={output_config}")
        print(f"config_size={human_size(output_config)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
