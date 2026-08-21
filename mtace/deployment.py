"""TorchScript export shared by the LAMMPS deployment interfaces."""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Sequence

import torch
from ase.data import atomic_numbers, chemical_symbols

from .checkpoint import restore_model

LAMMPS_FORMAT_VERSION = 1


class AtomicEnergyDeployment(torch.nn.Module):
    """Strictly local atomic energies on unwrapped local/ghost coordinates."""

    def __init__(self, model: torch.nn.Module, references: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("references", references)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        cell = positions.new_zeros((3, 3))
        shifts = positions.new_zeros((edge_index.shape[1], 3))
        learned = self.model.atomic_energies(
            atomic_numbers, positions, cell, edge_index, shifts
        )
        return learned + self.references[atomic_numbers]


def _normalize_elements(elements: Sequence[str | int]) -> tuple[list[str], list[int]]:
    names: list[str] = []
    numbers: list[int] = []
    for value in elements:
        if isinstance(value, int) or str(value).isdigit():
            number = int(value)
            if not 1 <= number <= 118:
                raise ValueError(f"Atomic number must satisfy 1 <= Z <= 118, got {number}")
            name = chemical_symbols[number]
        else:
            name = str(value)
            if name not in atomic_numbers:
                raise ValueError(f"Unknown chemical element {name!r}")
            number = atomic_numbers[name]
        if number in numbers:
            raise ValueError(f"Duplicate deployment element {name}")
        names.append(name)
        numbers.append(number)
    if not numbers:
        raise ValueError("At least one deployment element is required")
    return names, numbers


def _complete_graph(count: int, device: torch.device) -> torch.Tensor:
    index = torch.arange(count, device=device, dtype=torch.long)
    receiver = index.repeat_interleave(count)
    sender = index.repeat(count)
    keep = sender != receiver
    return torch.stack((sender[keep], receiver[keep]))


def _example_inputs(
    numbers: Sequence[int], r_max: float, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = max(4, min(8, len(numbers)))
    z = torch.tensor(
        [numbers[index % len(numbers)] for index in range(count)],
        dtype=torch.long,
        device=device,
    )
    scale = 0.18 * r_max
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [0.2, 1.1, 0.1],
            [0.1, 0.3, 1.2],
            [0.8, 0.7, 0.2],
            [0.6, 0.2, 0.8],
            [0.1, 0.8, 0.7],
            [0.7, 0.6, 0.7],
        ],
        dtype=dtype,
        device=device,
    )[:count]
    return z, coordinates * scale, _complete_graph(count, device)


def _validate_export(
    original: AtomicEnergyDeployment,
    exported: torch.jit.ScriptModule,
    numbers: Sequence[int],
    r_max: float,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    z, positions, edges = _example_inputs(numbers, r_max, dtype, device)
    # Validate a graph shape different from the trace example and its input gradient.
    z = torch.cat((z, z[:1]))
    positions = torch.cat(
        (positions, positions[:1] + positions.new_tensor([[0.07, 0.05, 0.03]])), dim=0
    ).requires_grad_(True)
    edges = _complete_graph(z.numel(), device)
    expected = original(z, positions, edges)
    actual = exported(z, positions, edges)
    tolerance = 2.0e-5 if dtype == torch.float32 else 2.0e-10
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    expected_force = -torch.autograd.grad(expected.sum(), positions, retain_graph=True)[0]
    actual_force = -torch.autograd.grad(actual.sum(), positions)[0]
    torch.testing.assert_close(actual_force, expected_force, rtol=5 * tolerance, atol=5 * tolerance)

    def virial(module: torch.nn.Module) -> torch.Tensor:
        strain = torch.zeros((3, 3), dtype=dtype, device=device, requires_grad=True)
        deformed = positions.detach() @ (
            torch.eye(3, dtype=dtype, device=device) + strain
        )
        energy = module(z, deformed, edges).sum()
        gradient = torch.autograd.grad(energy, strain)[0]
        return -0.5 * (gradient + gradient.transpose(0, 1))

    torch.testing.assert_close(
        virial(exported), virial(original), rtol=5 * tolerance, atol=5 * tolerance
    )
    isolated_z = z[:1]
    isolated_positions = positions[:1].detach()
    isolated_edges = torch.empty((2, 0), dtype=torch.long, device=device)
    torch.testing.assert_close(
        exported(isolated_z, isolated_positions, isolated_edges),
        original(isolated_z, isolated_positions, isolated_edges),
        rtol=tolerance,
        atol=tolerance,
    )


def export_lammps_model(
    checkpoint_path: str | Path,
    output_path: str | Path,
    elements: Sequence[str | int],
) -> Path:
    """Export dynamic atomic energies plus validated deployment metadata."""

    names, numbers = _normalize_elements(elements)
    device = torch.device("cpu")
    model, checkpoint = restore_model(
        checkpoint_path, device, mamba_backend="torch"
    )
    r_max = float(model.r_max)
    if not math.isfinite(r_max) or r_max <= 0.0:
        raise ValueError("LAMMPS export requires a positive finite model cutoff")
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.get_default_dtype()
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError(f"LAMMPS export requires float32 or float64, got {dtype}")
    reference_values = {
        int(key): float(value)
        for key, value in checkpoint.get("atomic_energies", {}).items()
    }
    invalid_reference_keys = sorted(
        number for number in reference_values if not 1 <= number <= 118
    )
    if invalid_reference_keys:
        raise ValueError(f"Invalid atomic-reference keys Z={invalid_reference_keys}")
    if any(not math.isfinite(value) for value in reference_values.values()):
        raise ValueError("Atomic reference energies must be finite")
    trained_numbers = {
        int(number) for number in checkpoint.get("atomic_numbers", [])
    }
    if not trained_numbers and reference_values:
        trained_numbers = set(reference_values)
    untrained = sorted(set(numbers) - trained_numbers) if trained_numbers else []
    if untrained:
        raise ValueError(f"Deployment elements were not present during training: Z={untrained}")
    missing = sorted(set(numbers) - set(reference_values))
    if reference_values and missing:
        raise ValueError(f"Missing atomic reference energies for Z={missing}")
    references = torch.zeros(119, dtype=dtype, device=device)
    for number, value in reference_values.items():
        references[number] = value
    deployment = AtomicEnergyDeployment(model, references).eval()
    example = _example_inputs(numbers, r_max, dtype, device)
    with warnings.catch_warnings(), torch.jit.optimized_execution(False):
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings(
            "ignore",
            message="The TorchScript type system doesn't support instance-level annotations.*",
        )
        traced = torch.jit.trace(deployment, example, check_trace=False, strict=True)
    traced = torch.jit.freeze(traced.eval())
    _validate_export(
        deployment, traced, numbers, r_max, dtype, device
    )
    metadata = {
        "mamba_ace_format_version": str(LAMMPS_FORMAT_VERSION),
        "architecture": str(checkpoint.get("architecture", "mtace_v2")),
        "architecture_version": str(checkpoint.get("architecture_version", 1)),
        "r_max": format(r_max, ".17g"),
        "elements": " ".join(names),
        "atomic_numbers": " ".join(map(str, numbers)),
        "dtype": "float32" if dtype == torch.float32 else "float64",
        "energy_units": "eV",
        "length_units": "Angstrom",
        "output": "atomic_energy",
        "strictly_local": "1",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output), _extra_files=metadata)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export a MTACE checkpoint for pair_style mamba"
    )
    parser.add_argument("checkpoint", help="MTACE training checkpoint")
    parser.add_argument("output", help="Output .mtace.pt TorchScript model")
    parser.add_argument(
        "--elements",
        nargs="+",
        required=True,
        help="Allowed element symbols or atomic numbers, in model mapping order",
    )
    args = parser.parse_args(argv)
    output = export_lammps_model(args.checkpoint, args.output, args.elements)
    print(output)


if __name__ == "__main__":
    main()
