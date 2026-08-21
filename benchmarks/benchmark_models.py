#!/usr/bin/env python3
"""Compare MTACE with the local TRACE Transformer reference."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtace.model import MambaACE


def structure(atom_count: int, cutoff: float, seed: int = 7):
    generator = torch.Generator().manual_seed(seed + atom_count)
    positions = torch.rand(atom_count, 3, generator=generator) * 5.0
    distances = torch.cdist(positions, positions)
    sender, receiver = torch.where((distances < cutoff) & (distances > 0.0))
    species_table = torch.tensor([1, 8, 14, 53])
    return {
        "z": species_table[torch.randint(0, 4, (atom_count,), generator=generator)],
        "pos": positions,
        "cell": torch.eye(3) * 12.0,
        "edge_index": torch.stack((sender, receiver)),
        "edge_shift": torch.zeros((sender.numel(), 3)),
        "volume": torch.tensor(12.0**3),
    }


def latency(model, data, warmup: int, iterations: int):
    for _ in range(warmup):
        model(data, training=False, compute_stress=False)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        model(data, training=False, compute_stress=False)
        samples.append(1000.0 * (time.perf_counter() - start))
    return statistics.median(samples), statistics.mean(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash-ace-root", type=Path, required=True)
    parser.add_argument("--atoms", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/model_benchmark.csv"))
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    sys.path.insert(0, str(args.flash_ace_root.resolve()))
    from flashace.model import TransformersACE

    common = dict(
        r_max=4.5,
        l_max=2,
        num_radial=6,
        hidden_dim=16,
        num_layers=1,
        correlation_order=4,
        correlation_channels=8,
    )
    models = {
        "MTACE-v2-physical-Mamba3-MIMO": MambaACE(
            **common,
            tokenizer_type="physical_shells",
            num_shells=32,
            mixer_type="mamba",
            mamba_dim=4,
            mamba_d_state=4,
            mamba_headdim=8,
            mamba_mimo_rank=4,
            mamba_chunk_size=16,
            ffn_hidden=21,
            mamba_backend="torch",
        ).eval(),
        "TRACE-v2": TransformersACE(
            **common,
            attention_num_heads=2,
            attention_ffn_hidden=32,
        ).eval(),
    }
    rows = []
    for atom_count in args.atoms:
        data = structure(atom_count, common["r_max"])
        for name, model in models.items():
            median, mean, deviation = latency(model, data, args.warmup, args.iterations)
            row = {
                "model": name,
                "atoms": atom_count,
                "edges": int(data["edge_index"].shape[1]),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "median_ms": median,
                "mean_ms": mean,
                "stdev_ms": deviation,
                "device": "cpu",
                "threads": torch.get_num_threads(),
                "torch_version": torch.__version__,
            }
            rows.append(row)
            print(json.dumps(row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
