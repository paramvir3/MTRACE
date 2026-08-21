#!/usr/bin/env python3
"""Controlled mixer study: does state-space mixing help an ACE potential?

This is the experiment the manuscript's central claim depends on.  It holds the
ACE frontend, shell tokenizer, equivariant coupling, readout, data split, loss,
optimizer and schedule fixed and varies only ``mixer_type``.

Two things the architecture-v8 protocol got wrong and this driver fixes:

1. *Capacity.*  The mixers are not automatically parameter matched -- at the
   production configuration Mamba-3 MIMO carries about 4.4 times the mixer
   parameters of attention.  ``--match-parameters`` widens the cheaper mixers
   until the total parameter count agrees to within a tolerance, and the report
   always prints the counts so a reader can judge for themselves.
2. *Seeds.*  A single seed cannot separate a mixer effect from initialization
   noise.  ``--seeds`` runs each arm several times and reports mean and spread.

The driver also records ``gate_shell_dependence``: because ``sum_k T_ik = A_i``,
a gate that does not vary across shells reproduces the direct ACE path exactly,
so a residual fraction that stays near zero means the mixer is decorative
whatever the error numbers say.

Example
-------
    python experiments/matched_mixer_study.py \
        --train examples/cspbi3/train.extxyz --frames 200 \
        --epochs 60 --seeds 3 --match-parameters \
        --output results/cspbi3_mixers.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from mtace.data import AtomisticDataset, average_num_neighbors, split_frames
from mtace.model import MambaACEV2
from mtace.optim import build_optimizer

MIXERS = ("identity", "mlp", "dense", "attention", "mamba")


def mixer_settings(name: str, width: int, rank: int) -> dict:
    """Constructor overrides for one arm of the study."""

    if name == "mamba":
        return {"mixer_type": "mamba", "mamba_mimo_rank": rank, "mamba_dim": width}
    if name == "attention":
        return {"mixer_type": "attention", "attention_heads": 4, "mamba_dim": width}
    return {"mixer_type": name, "mamba_dim": width}


def build_model(name: str, base: dict, width: int, rank: int) -> MambaACEV2:
    settings = dict(base)
    settings.update(mixer_settings(name, width, rank))
    return MambaACEV2(**settings)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def mixer_parameter_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".mixer." in name
    )


def matched_width(
    name: str, base: dict, target: int, rank: int, tolerance: float
) -> int:
    """Smallest mixer width whose total parameter count reaches ``target``.

    The identity control has no mixer parameters and therefore cannot be
    matched; it is reported at its natural size as a capacity lower bound.
    """

    if name == "identity":
        return int(base["mamba_dim"])
    width = int(base["mamba_dim"])
    best = width
    for candidate in range(8, 8 * 64, 8):
        try:
            count = parameter_count(build_model(name, base, candidate, rank))
        except ValueError:
            continue
        best = candidate
        if count >= target * (1.0 - tolerance):
            return candidate
    return best


def evaluate(model, dataset, reference, device) -> tuple[float, float]:
    model.eval()
    energy_squared = 0.0
    force_squared = 0.0
    components = 0
    for index in range(len(dataset)):
        item = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in dataset[index].items()
        }
        energy, forces, _, _ = model(item, training=False, compute_stress=False)
        target = item["target_energy"] - reference[item["z"]].sum()
        energy_squared += float(((energy - target) / item["z"].numel()) ** 2)
        force_squared += float((forces - item["target_forces"]).square().sum())
        components += forces.numel()
    return (
        1000.0 * math.sqrt(energy_squared / len(dataset)),
        math.sqrt(force_squared / components),
    )


def run_arm(
    name: str,
    base: dict,
    width: int,
    rank: int,
    datasets: tuple,
    reference: torch.Tensor,
    arguments,
    seed: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(name, base, width, rank).to(device)
    training, validation = datasets
    optimizer = build_optimizer(
        model,
        {
            "optimizer": arguments.optimizer,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
        },
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=arguments.epochs, eta_min=arguments.learning_rate / 100.0
    )
    best_force = float("inf")
    best_energy = float("inf")
    history = []
    start = time.perf_counter()
    for epoch in range(arguments.epochs):
        model.train()
        order = torch.randperm(len(training)).tolist()
        for offset in range(0, len(order), arguments.batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch = order[offset : offset + arguments.batch_size]
            total = torch.zeros((), device=device)
            for index in batch:
                item = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in training[index].items()
                }
                energy, forces, _, _ = model(
                    item, training=True, compute_stress=False
                )
                target = item["target_energy"] - reference[item["z"]].sum()
                loss = arguments.energy_weight * (
                    (energy - target) / item["z"].numel()
                ) ** 2 + arguments.forces_weight * (
                    forces - item["target_forces"]
                ).square().mean()
                total = total + loss / len(batch)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        scheduler.step()
        energy_rmse, force_rmse = evaluate(model, validation, reference, device)
        history.append({"epoch": epoch + 1, "energy": energy_rmse, "force": force_rmse})
        if force_rmse < best_force:
            best_force, best_energy = force_rmse, energy_rmse
        if arguments.verbose:
            print(
                f"    [{name} seed={seed}] epoch {epoch + 1:03d} "
                f"E={energy_rmse:8.3f} meV/atom  F={force_rmse:.4f} eV/A",
                flush=True,
            )
    elapsed = time.perf_counter() - start

    probe = training[0]
    gate = model.gate_shell_dependence(
        probe["z"].to(device),
        probe["pos"].to(device),
        probe["cell"].to(device),
        probe["edge_index"].to(device),
        probe["edge_shift"].to(device),
    )
    return {
        "mixer": name,
        "seed": seed,
        "mamba_dim": width,
        "parameters": parameter_count(model),
        "mixer_parameters": mixer_parameter_count(model),
        "best_force_rmse_ev_a": best_force,
        "energy_rmse_at_best_force_mev_atom": best_energy,
        "seconds": elapsed,
        "gate_shell_dependence": gate,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--mixers", nargs="+", default=list(MIXERS))
    parser.add_argument("--mimo-rank", type=int, default=1,
                        help="hold at 1 for the strict SISO comparison")
    parser.add_argument("--match-parameters", action="store_true")
    parser.add_argument("--match-tolerance", type=float, default=0.05)
    parser.add_argument("--r-max", type=float, default=6.0)
    parser.add_argument("--l-max", type=int, default=2)
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--correlation-order", type=int, default=3)
    parser.add_argument("--num-shells", type=int, default=16)
    parser.add_argument("--shell-r-min", type=float, default=0.0)
    parser.add_argument("--mamba-dim", type=int, default=32)
    parser.add_argument("--optimizer", default="muon")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--forces-weight", type=float, default=10.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    frames = read(arguments.train, index=f"0:{arguments.frames}:{arguments.stride}")
    training_frames, validation_frames, _ = split_frames(
        frames, arguments.validation_fraction, seed=0, mode="blocked", block_size=10
    )
    neighbors = average_num_neighbors(training_frames, arguments.r_max)
    base = {
        "r_max": arguments.r_max,
        "l_max": arguments.l_max,
        "num_radial": arguments.num_radial,
        "hidden_dim": arguments.hidden_dim,
        "num_layers": arguments.num_layers,
        "correlation_order": arguments.correlation_order,
        "correlation_channels": 8,
        "num_shells": arguments.num_shells,
        "shell_r_min": arguments.shell_r_min,
        "avg_num_neighbors": neighbors,
        "mamba_dim": arguments.mamba_dim,
        "mamba_d_state": 8,
        "mamba_headdim": 8,
        "readout_hidden": arguments.hidden_dim,
        "mamba_backend": "torch",
        "dropout": 0.0,
        "attention_dropout": 0.0,
        # Hold the scalar residual block fixed so only the mixer varies.
        "ffn_type": "swiglu",
    }

    training = AtomisticDataset(training_frames, arguments.r_max)
    validation = AtomisticDataset(validation_frames, arguments.r_max)
    species = sorted({int(n) for atoms in training_frames for n in atoms.numbers})
    counts = np.array(
        [[int(np.count_nonzero(a.numbers == z)) for z in species] for a in training_frames],
        dtype=float,
    )
    energies = np.array([a.get_potential_energy() for a in training_frames])
    if np.linalg.matrix_rank(counts) < len(species):
        # Fixed composition: a single per-atom offset is the only identifiable
        # reference, exactly as in examples/cspbi3.
        offset = float(energies.sum() / counts.sum())
        solution = {z: offset for z in species}
    else:
        coefficients, *_ = np.linalg.lstsq(counts, energies, rcond=None)
        solution = {z: float(v) for z, v in zip(species, coefficients)}
    reference = torch.zeros(119, device=device)
    for number, value in solution.items():
        reference[number] = value

    print(
        f"frames: train={len(training_frames)} valid={len(validation_frames)} "
        f"atoms/frame={len(training_frames[0])} avg_num_neighbors={neighbors:.2f}"
    )
    target = parameter_count(
        build_model("mamba", base, arguments.mamba_dim, arguments.mimo_rank)
    )
    widths = {}
    for name in arguments.mixers:
        widths[name] = (
            matched_width(
                name, base, target, arguments.mimo_rank, arguments.match_tolerance
            )
            if arguments.match_parameters
            else arguments.mamba_dim
        )

    records = []
    for name in arguments.mixers:
        for seed in range(arguments.seeds):
            print(f"  running mixer={name} width={widths[name]} seed={seed}", flush=True)
            records.append(
                run_arm(
                    name,
                    base,
                    widths[name],
                    arguments.mimo_rank,
                    (training, validation),
                    reference,
                    arguments,
                    seed,
                    device,
                )
            )

    print("\n=== matched mixer study ===")
    print(
        f"{'mixer':>10} {'params':>9} {'mixer p.':>9} "
        f"{'F_RMSE mean+-sd':>20} {'E_RMSE mean':>12} {'gate resid':>11} {'s/run':>8}"
    )
    summary = []
    for name in arguments.mixers:
        arm = [record for record in records if record["mixer"] == name]
        forces = [record["best_force_rmse_ev_a"] for record in arm]
        energies_ = [record["energy_rmse_at_best_force_mev_atom"] for record in arm]
        residual = [
            record["gate_shell_dependence"][0]["residual_fraction"] for record in arm
        ]
        spread = statistics.stdev(forces) if len(forces) > 1 else 0.0
        entry = {
            "mixer": name,
            "parameters": arm[0]["parameters"],
            "mixer_parameters": arm[0]["mixer_parameters"],
            "force_rmse_mean": statistics.mean(forces),
            "force_rmse_stdev": spread,
            "energy_rmse_mean": statistics.mean(energies_),
            "gate_residual_fraction_mean": statistics.mean(residual),
            "seconds_mean": statistics.mean(r["seconds"] for r in arm),
        }
        summary.append(entry)
        print(
            f"{name:>10} {entry['parameters']:9d} {entry['mixer_parameters']:9d} "
            f"{entry['force_rmse_mean']:11.4f}+-{spread:<7.4f} "
            f"{entry['energy_rmse_mean']:12.2f} "
            f"{entry['gate_residual_fraction_mean']:11.4f} "
            f"{entry['seconds_mean']:8.1f}"
        )
    print(
        "\ngate resid is the share of the equivariant update a shell-constant gate\n"
        "cannot reproduce.  Near zero means the mixer has collapsed onto plain ACE."
    )

    if arguments.output:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "configuration": vars(arguments),
                    "base_model": base,
                    "widths": widths,
                    "summary": summary,
                    "records": records,
                },
                indent=2,
            )
        )
        print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
