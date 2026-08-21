#!/usr/bin/env python3
"""Scaling benchmark for matched standalone sequence mixers."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtace.mamba3 import Mamba3SequenceMixer
from mtace.mixers import (
    AttentionSequenceMixer,
    DeepSetsSequenceMixer,
    DenseRadialSequenceMixer,
)
from mtace.ssm import MambaSequenceMixer


def measure(module, sample, iterations):
    for _ in range(2):
        module(sample)
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        module(sample)
        timings.append(1000.0 * (time.perf_counter() - start))
    return statistics.median(timings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[16, 64, 256, 1024])
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    mamba3_siso = Mamba3SequenceMixer(
        args.dimension, d_state=16, headdim=16, mimo_rank=1, backend="torch"
    ).eval()
    mamba3_mimo = Mamba3SequenceMixer(
        args.dimension, d_state=16, headdim=16, mimo_rank=4, backend="torch"
    ).eval()
    mamba1 = MambaSequenceMixer(args.dimension, d_state=16, backend="torch").eval()
    attention = AttentionSequenceMixer(args.dimension, 4).eval()
    deepsets = DeepSetsSequenceMixer(args.dimension).eval()

    for length in args.lengths:
        sample = torch.randn(args.batch, length, args.dimension)
        dense = DenseRadialSequenceMixer(args.dimension, length).eval()
        print(
            f"length={length:5d} "
            f"mamba3_siso_ms={measure(mamba3_siso, sample, args.iterations):9.3f} "
            f"mamba3_mimo_ms={measure(mamba3_mimo, sample, args.iterations):9.3f} "
            f"mamba1_ms={measure(mamba1, sample, args.iterations):9.3f} "
            f"attention_ms={measure(attention, sample, args.iterations):9.3f} "
            f"dense_ms={measure(dense, sample, args.iterations):9.3f} "
            f"deepsets_ms={measure(deepsets, sample, args.iterations):9.3f}"
        )


if __name__ == "__main__":
    main()
