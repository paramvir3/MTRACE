#!/usr/bin/env python3
"""Measure how much of a trained mixer's update needs cross-shell communication.

For every checkpoint and frame this reports, per layer, the two diagnostics of
``MambaACEV2.gate_shell_dependence``:

``residual_fraction``  share of the update a shell-constant gate cannot
                       reproduce -- shell-dependent *gating*, which the identity
                       control has as well;
``mixing_fraction``    share that disappears when the mixer sees each shell
                       token in isolation -- cross-shell *communication*, which
                       is exactly zero for the identity control.

``branch_fraction``    size of the branch as added (after W_O and the layer
                       scale) relative to the features it updates.  The two
                       fractions above are shares of this branch.

Only ``mixing_fraction`` tests whether the sequence model communicates across
shells, and only ``mixing_fraction * branch_fraction`` says how much that
communication moves the features.  Frames are taken
from the validation split that ``train.py`` would build from the same settings,
so a checkpoint is measured on structures it was not trained on.

    PYTHONPATH=. python3 experiments/mixing_fraction.py \\
        --frames results/water_h100_partial/water_bulk.xyz \\
        --checkpoints results/water_h100_partial/checkpoints/h100_*.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from mtace.checkpoint import restore_model
from mtace.data import build_neighbor_tensors, split_frames


def validation_frames(frames, args):
    _, validation, _ = split_frames(
        frames,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        mode=args.split_mode,
        block_size=args.split_block_size,
        gap=args.split_gap,
    )
    if args.max_atoms:
        validation = [atoms for atoms in validation if len(atoms) <= args.max_atoms]
    if args.max_frames and len(validation) > args.max_frames:
        # Evenly spaced through the split rather than the first few, which in a
        # blocked split are consecutive trajectory frames.
        picks = np.linspace(0, len(validation) - 1, args.max_frames).round().astype(int)
        validation = [validation[int(index)] for index in picks]
    return validation


def measure(path: Path, frames, initialise_seed: int | None = None) -> dict:
    model, checkpoint = restore_model(path, "cpu", mamba_backend="torch")
    if initialise_seed is not None:
        # Same configuration, same migrated settings, fresh weights.
        from mtace.checkpoint import migrated_model_config
        from mtace.model import MambaACEV2

        torch.manual_seed(initialise_seed)
        model = MambaACEV2(**migrated_model_config(
            checkpoint, mamba_backend="torch"
        )).to(next(model.parameters()).dtype).eval()
        checkpoint = {**checkpoint, "epoch": 0}
    dtype = next(model.parameters()).dtype
    per_layer: dict[int, dict[str, list[float]]] = {}
    for atoms in frames:
        edge_index, edge_shift = build_neighbor_tensors(atoms, model.r_max, dtype)
        report = model.gate_shell_dependence(
            torch.as_tensor(atoms.numbers, dtype=torch.long),
            torch.as_tensor(atoms.positions, dtype=dtype),
            torch.as_tensor(atoms.cell.array, dtype=dtype),
            edge_index,
            edge_shift,
        )
        for entry in report:
            store = per_layer.setdefault(int(entry["layer"]), {})
            for key, value in entry.items():
                if key != "layer":
                    store.setdefault(key, []).append(float(value))
    summary = {
        "checkpoint": str(path),
        "architecture_version": int(checkpoint.get("architecture_version", 0)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "mixers": [layer.mixer_type for layer in model.layers],
        "frames": len(frames),
        "layers": {},
    }
    for layer, values in sorted(per_layer.items()):
        summary["layers"][layer] = {
            key: {
                "mean": float(np.mean(series)),
                "std": float(np.std(series, ddof=1)) if len(series) > 1 else 0.0,
            }
            for key, series in values.items()
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frames", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", default="blocked")
    parser.add_argument("--split-block-size", type=int, default=25)
    parser.add_argument("--split-gap", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--max-atoms", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--initialised", type=int, default=0, metavar="SEEDS",
        help="also measure freshly initialised models with each checkpoint's "
        "configuration, over this many seeds, to separate learned mixing from "
        "mixing present at initialisation",
    )
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    frames = validation_frames(read(args.frames, index=":"), args)
    print(f"{len(frames)} held-out frames, {sorted({len(a) for a in frames})} atoms\n")
    print(f"{'checkpoint':>26} {'ep':>4} {'layer':>5} {'mixer':>10} "
          f"{'residual':>17} {'mixing':>17} {'branch':>8} {'|g|':>7}")
    results = []
    runs = [(path, None) for path in args.checkpoints]
    runs += [(path, seed) for path in args.checkpoints for seed in range(args.initialised)]
    for path, seed in runs:
        summary = measure(Path(path), frames, seed)
        summary["initialise_seed"] = seed
        results.append(summary)
        for layer, stats in summary["layers"].items():
            res, mix = stats["residual_fraction"], stats["mixing_fraction"]
            mixing = (
                "nan" if math.isnan(mix["mean"])
                else f"{mix['mean']:.4f} +- {mix['std']:.4f}"
            )
            print(f"{Path(path).name:>26} {summary['epoch']:>4} {layer:>5} "
                  f"{summary['mixers'][layer]:>10} "
                  f"{res['mean']:.4f} +- {res['std']:.4f} {mixing:>17} "
                  f"{stats['branch_fraction']['mean']:8.4f} "
                  f"{stats['gate_abs_mean']['mean']:7.3f}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
