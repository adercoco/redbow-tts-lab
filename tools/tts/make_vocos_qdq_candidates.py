#!/usr/bin/env python3
"""Create iOS-friendlier Vocos quantization candidates.

The report-favorite dynamic int8 Vocos uses ConvInteger, which crashes on the
iOS/simulator ONNX Runtime build. These candidates keep the graph in QDQ form
or only quantize MatMul so we can test what the mobile runtime accepts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)


class RandomMelReader(CalibrationDataReader):
    def __init__(self, batches: int = 16, frames: int = 180) -> None:
        rng = np.random.default_rng(20260607)
        self._items = []
        for _ in range(batches):
            # ZipVoice/Vocos mel-like activations are centered and moderate.
            # Include a small drift so min/max calibration sees voiced variation.
            mels = rng.normal(0.0, 1.2, size=(1, 100, frames)).astype(np.float32)
            drift = rng.normal(0.0, 0.25, size=(1, 100, 1)).astype(np.float32)
            self._items.append({"mels": mels + drift})
        self._iter = iter(self._items)

    def get_next(self):
        return next(self._iter, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    fp32 = Path(args.fp32)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OMP_NUM_THREADS", "4")

    qdq_all = out_dir / "vocos_24khz_qdq_all_int8.onnx"
    quantize_static(
        fp32,
        qdq_all,
        RandomMelReader(batches=24, frames=220),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "MatMul"],
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )

    qdq_matmul = out_dir / "vocos_24khz_qdq_matmul_int8.onnx"
    quantize_static(
        fp32,
        qdq_matmul,
        RandomMelReader(batches=24, frames=220),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["MatMul"],
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )

    dynamic_matmul = out_dir / "vocos_24khz_dynamic_matmul_qint8.onnx"
    quantize_dynamic(
        fp32,
        dynamic_matmul,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    for path in [qdq_all, qdq_matmul, dynamic_matmul]:
        print(f"{path}\t{path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
