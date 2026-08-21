#!/usr/bin/env python3
"""Regenerate every measured number quoted in the manuscript.

Draft 5 quoted a density-normalization figure with no stated protocol, and it is
not reproducible as written: the answer depends on the coordination sweep, on
l_max, and on the correlation order.  Every number the paper reports should come
out of this script, with the protocol printed alongside it.

    python experiments/reproduce_manuscript_numbers.py
    python experiments/reproduce_manuscript_numbers.py --json results/numbers.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk
from ase.io import read

from mtace.data import (
    average_num_neighbors,
    build_neighbor_tensors,
    minimum_edge_distance,
)
from mtace.model import MambaACEV2
from mtace.physics import CompactRadialShellBasis

DOUBLE = torch.float64


def _model(**overrides):
    settings = dict(
        r_max=6.0, l_max=2, num_radial=8, hidden_dim=32, num_layers=1,
        correlation_order=4, correlation_channels=8, num_shells=16,
        mamba_dim=32, mamba_d_state=8, mamba_headdim=8, readout_hidden=32,
        mamba_backend="torch",
    )
    settings.update(overrides)
    torch.manual_seed(0)
    return MambaACEV2(**settings).double().eval()


def density_normalization(results: dict) -> None:
    """How much does the node feature move when coordination changes?

    Protocol, stated because the number depends on it: fcc copper conventional
    cells repeated 2x2x2, lattice constants 4.2 / 3.6 / 3.1 / 2.8 Angstrom,
    r_max = 6 A, l_max = 2, correlation_order = 4, seed 0, float64, untrained
    weights.  The reported figure is
    ``max(rms node feature) / min(rms node feature)`` over the sweep.
    """

    lattice_constants = (4.2, 3.6, 3.1, 2.8)
    frames = [
        bulk("Cu", "fcc", a=constant, cubic=True).repeat(2)
        for constant in lattice_constants
    ]
    neighbors = average_num_neighbors(frames, 6.0)
    coordination = []
    entry = {
        "protocol": density_normalization.__doc__.strip(),
        "lattice_constants_angstrom": list(lattice_constants),
        "avg_num_neighbors": neighbors,
    }
    for label, value in (("off", 1.0), ("on", neighbors)):
        model = _model(avg_num_neighbors=value)
        magnitudes = []
        for atoms in frames:
            edge_index, edge_shift = build_neighbor_tensors(atoms, 6.0, DOUBLE)
            positions = torch.tensor(atoms.positions, dtype=DOUBLE)
            cell = torch.tensor(atoms.cell.array, dtype=DOUBLE)
            z = torch.tensor(atoms.numbers, dtype=torch.long)
            with torch.no_grad():
                vectors = positions[edge_index[0]] - positions[edge_index[1]]
                vectors = vectors + edge_shift @ cell
                lengths = vectors.norm(dim=-1)
                features, _, _, _ = model.ace(
                    model.species_embedding(z), edge_index, vectors, lengths
                )
            magnitudes.append(float(features.pow(2).mean().sqrt()))
            if label == "off":
                coordination.append(edge_index.shape[1] / len(atoms))
        entry[f"normalization_{label}"] = {
            "node_feature_rms": magnitudes,
            "spread": max(magnitudes) / min(magnitudes),
        }
    entry["neighbours_per_atom"] = coordination
    entry["coordination_sweep"] = max(coordination) / min(coordination)
    results["density_normalization"] = entry

    print("== density normalization ==")
    print(f"   coordination sweep {min(coordination):.0f} -> {max(coordination):.0f} "
          f"neighbours/atom ({entry['coordination_sweep']:.2f}x)")
    for label in ("off", "on"):
        print(f"   normalization {label:3s}: node-feature rms spread = "
              f"{entry[f'normalization_{label}']['spread']:.2f}x")


def derivative_contract(results: dict) -> None:
    """Third-derivative jump of the shell weights, analytic and measured."""

    rows = []
    for num_shells, span in ((16, 4.0), (32, 4.0), (32, 6.0), (24, 4.5)):
        basis = CompactRadialShellBasis(6.0, num_shells, r_min=6.0 - span, degree=3)
        spacing = span / (num_shells - 1)
        knot = (6.0 - span) + 7 * spacing
        step = spacing / 60.0
        weights = lambda x: basis.dense(torch.tensor([x], dtype=DOUBLE))[0]
        left = (weights(knot) - 3 * weights(knot - step)
                + 3 * weights(knot - 2 * step) - weights(knot - 3 * step)) / step**3
        right = (weights(knot + 3 * step) - 3 * weights(knot + 2 * step)
                 + 3 * weights(knot + step) - weights(knot)) / step**3
        rows.append({
            "num_shells": num_shells,
            "span_angstrom": span,
            "measured_jump": float((left - right).abs().max()),
            "predicted_jump": 6.0 * ((num_shells - 1) / span) ** 3,
        })
    results["derivative_contract"] = rows
    print("\n== derivative-order contract (cubic shells) ==")
    print("   L  span   measured      6((L-1)/span)^3")
    for row in rows:
        print(f"   {row['num_shells']:2d}  {row['span_angstrom']:.1f}   "
              f"{row['measured_jump']:10.2f}   {row['predicted_jump']:10.2f}")


def shell_occupancy(results: dict, dataset: Path) -> None:
    """Fraction of the shell sequence that is identically zero on real data."""

    if not dataset.is_file():
        print(f"\n== shell occupancy ==\n   skipped, {dataset} not found")
        return
    frames = read(str(dataset), index="0:20")
    shortest = minimum_edge_distance(frames, 6.0)
    entry = {"shortest_distance_angstrom": shortest, "cases": []}
    for num_shells, r_min in ((32, 0.0), (32, 2.0), (16, 2.0)):
        basis = CompactRadialShellBasis(6.0, num_shells, r_min=r_min)
        touched = torch.zeros(num_shells, dtype=DOUBLE)
        for atoms in frames:
            edge_index, edge_shift = build_neighbor_tensors(atoms, 6.0, DOUBLE)
            positions = torch.tensor(atoms.positions, dtype=DOUBLE)
            cell = torch.tensor(atoms.cell.array, dtype=DOUBLE)
            vectors = positions[edge_index[0]] - positions[edge_index[1]]
            lengths = (vectors + edge_shift @ cell).norm(dim=-1)
            touched += (basis.dense(lengths) > 1e-12).to(DOUBLE).sum(0)
        dead = int((touched == 0).sum())
        entry["cases"].append({
            "num_shells": num_shells, "shell_r_min": r_min,
            "dead_shells": dead, "dead_fraction": dead / num_shells,
        })
    results["shell_occupancy"] = entry
    print("\n== shell occupancy ==")
    print(f"   shortest interatomic distance in the data: {shortest:.3f} A")
    for case in entry["cases"]:
        print(f"   L={case['num_shells']:2d} r_min={case['shell_r_min']:.1f} A -> "
              f"{case['dead_shells']:2d} dead shells "
              f"({100 * case['dead_fraction']:.0f}% of the sequence)")


def invariant_homogeneity(results: dict) -> None:
    """Scaling degree of each invariant block under a global feature rescaling."""

    entry = {}
    for mode in ("squared", "homogeneous"):
        model = _model(invariant_norm=mode)
        layer = model.layers[0]
        torch.manual_seed(1)
        tokens = torch.randn(4, 16, model.ace.irreps_correlation.dim, dtype=DOUBLE) * 10.0
        width = layer.token_scalar_dim
        reference = layer._token_invariants(tokens)
        ratios = {}
        for scale in (2.0, 4.0):
            scaled = layer._token_invariants(tokens * scale)
            ratios[scale] = {
                "scalar_block": float(
                    scaled[..., :width].norm() / reference[..., :width].norm()
                ),
                "norm_block": float(
                    scaled[..., width:].norm() / reference[..., width:].norm()
                ),
            }
        entry[mode] = ratios
    results["invariant_homogeneity"] = entry
    print("\n== invariant-map homogeneity ==")
    for mode, ratios in entry.items():
        for scale, blocks in ratios.items():
            print(f"   {mode:12s} s={scale:.0f}:  scalar x{blocks['scalar_block']:.3f}"
                  f"   norms x{blocks['norm_block']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None)
    parser.add_argument("--dataset", default="examples/cspbi3/train.extxyz")
    arguments = parser.parse_args()

    results: dict = {}
    density_normalization(results)
    derivative_contract(results)
    shell_occupancy(results, Path(arguments.dataset))
    invariant_homogeneity(results)

    if arguments.json:
        destination = Path(arguments.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
