#!/usr/bin/env python3
"""Export our fine-tuned Matcha checkpoint with the legacy ONNX tracer.

PyTorch 2.12 defaults to the dynamo exporter, which cannot currently trace
Matcha's data-dependent y_max_length slicing. The legacy tracer is acceptable
for the mobile experiment because we export one fixed ODE-step count per model
and keep the text length axis dynamic.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from lightning import LightningModule

from matcha.cli import load_matcha


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class ExportableMatcha(LightningModule):
    def __init__(self, checkpoint: Path, n_timesteps: int):
        super().__init__()
        self.matcha = load_matcha(checkpoint.stem, checkpoint, "cpu")
        self.n_timesteps = n_timesteps

    def forward(self, x, x_lengths, scales, spks=None):
        temperature = scales[0]
        length_scale = scales[1]
        output = self.matcha.synthesise(
            x,
            x_lengths,
            self.n_timesteps,
            temperature,
            spks,
            length_scale,
        )
        return output["mel"], output["mel_lengths"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-timesteps", type=int, required=True)
    parser.add_argument("--opset", type=int, default=15)
    parser.add_argument("--dummy-length", type=int, default=80)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = ExportableMatcha(args.checkpoint, args.n_timesteps).eval()

    x = torch.randint(0, 20, size=(1, args.dummy_length), dtype=torch.long)
    x_lengths = torch.tensor([args.dummy_length], dtype=torch.long)
    scales = torch.tensor([0.5, 0.85], dtype=torch.float32)
    inputs = (x, x_lengths, scales)

    torch.onnx.export(
        model,
        inputs,
        str(args.output),
        input_names=["x", "x_lengths", "scales"],
        output_names=["mel", "mel_lengths"],
        dynamic_axes={
            "x": {0: "batch_size", 1: "time"},
            "x_lengths": {0: "batch_size"},
            "mel": {0: "batch_size", 2: "mel_time"},
            "mel_lengths": {0: "batch_size"},
        },
        opset_version=args.opset,
        export_params=True,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"exported {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
