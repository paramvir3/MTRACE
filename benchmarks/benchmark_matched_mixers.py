#!/usr/bin/env python3
"""Benchmark interchangeable mixers on one fixed ACE/token representation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtace.model import MambaACEV2


def structure(atom_count: int, cutoff: float, seed: int = 23) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + atom_count)
    positions = torch.rand(atom_count, 3, generator=generator) * 7.0
    distances = torch.cdist(positions, positions)
    sender, receiver = torch.where((distances < cutoff) & (distances > 0.0))
    species = torch.tensor([1, 8, 14, 53])
    return {
        "z": species[torch.randint(0, len(species), (atom_count,), generator=generator)],
        "pos": positions,
        "cell": torch.eye(3) * 14.0,
        "edge_index": torch.stack((sender, receiver)),
        "edge_shift": torch.zeros((sender.numel(), 3)),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def latency(model, data, warmup: int, iterations: int) -> tuple[float, float, float]:
    device = data["pos"].device
    for _ in range(warmup):
        model(data, training=False, compute_stress=False)
    synchronize(device)
    samples = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter()
        model(data, training=False, compute_stress=False)
        synchronize(device)
        samples.append(1000.0 * (time.perf_counter() - start))
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return statistics.median(samples), statistics.mean(samples), deviation


def realized_mamba_backend(model, device: torch.device, requested: str) -> str:
    mixer = model.layers[0].mixer
    if not hasattr(mixer, "forward_direction"):
        return "n/a"
    direction = mixer.forward_direction
    fused = (
        requested != "torch"
        and device.type == "cuda"
        and direction.accelerated_backend_available
        and direction.fused_configuration_error is None
    )
    return "fused_bf16" if fused else "portable"


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The TorchScript type system doesn't support instance-level annotations.*",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--shells", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--mamba-backend", choices=("torch", "auto", "cuda"), default="torch"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/matched_mixer_benchmark.csv"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    torch.set_num_threads(max(1, args.threads))

    common = dict(
        r_max=5.0,
        l_max=2,
        num_radial=8,
        num_shells=args.shells,
        hidden_dim=32,
        num_layers=1,
        correlation_order=4,
        correlation_channels=8,
        tokenizer_type="physical_shells",
        shell_coupling_mode="conservative",
        mamba_dim=32,
        mamba_d_state=16,
        mamba_headdim=16,
        mamba_backend=args.mamba_backend,
        attention_heads=4,
        dropout=0.0,
    )
    specifications = {
        "mamba3_siso": {"mixer_type": "mamba", "mamba_mimo_rank": 1},
        "mamba3_mimo_r4": {"mixer_type": "mamba", "mamba_mimo_rank": 4},
        "attention": {"mixer_type": "attention", "mamba_mimo_rank": 1},
        "dense_radial": {"mixer_type": "dense", "mamba_mimo_rank": 1},
        "deepsets_mlp": {"mixer_type": "mlp", "mamba_mimo_rank": 1},
        "identity": {"mixer_type": "identity", "mamba_mimo_rank": 1},
    }
    models = {
        name: MambaACEV2(**common, **options).to(device).eval()
        for name, options in specifications.items()
    }

    rows = []
    for atom_count in args.atoms:
        data = {
            key: value.to(device)
            for key, value in structure(atom_count, common["r_max"]).items()
        }
        for name, model in models.items():
            median, mean, deviation = latency(
                model, data, args.warmup, args.iterations
            )
            row = {
                "model": name,
                "atoms": atom_count,
                "edges": int(data["edge_index"].shape[1]),
                "shells": args.shells,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "median_ms": median,
                "mean_ms": mean,
                "stdev_ms": deviation,
                "device": str(device),
                "mamba_backend": args.mamba_backend if name.startswith("mamba3") else "n/a",
                "realized_backend": realized_mamba_backend(
                    model, device, args.mamba_backend
                ),
                "workload": "energy+forces",
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
