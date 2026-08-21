"""Production ASE calculator for conservative MTACE checkpoints."""

from __future__ import annotations

import math
import threading
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from ase.calculators.calculator import (
    CalculationFailed,
    Calculator,
    CalculatorSetupError,
    PropertyNotImplementedError,
    all_changes,
)
from ase.data import atomic_numbers as ase_atomic_numbers
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList

from .checkpoint import restore_model


def _automatic_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_elements(elements: Sequence[str | int]) -> frozenset[int]:
    if isinstance(elements, (str, int)):
        elements = [elements]
    numbers: set[int] = set()
    for value in elements:
        text = str(value)
        if isinstance(value, int) or text.isdigit():
            number = int(value)
        else:
            try:
                number = ase_atomic_numbers[text]
            except KeyError as exception:
                raise CalculatorSetupError(f"Unknown chemical element {text!r}") from exception
        if not 1 <= number <= 118:
            raise CalculatorSetupError(
                f"Atomic numbers must satisfy 1 <= Z <= 118, got {number}"
            )
        numbers.add(number)
    if not numbers:
        raise CalculatorSetupError("elements must contain at least one species")
    return frozenset(numbers)


class MambaACECalculator(Calculator):
    """ASE calculator with cached Verlet neighbors and conservative derivatives.

    Parameters
    ----------
    checkpoint_path
        MTACE training checkpoint.
    device
        PyTorch device. ``None`` chooses CUDA, then MPS, then CPU.
    neighbor_skin
        ASE per-atom neighbor-list skin in Angstrom. Candidate edges are always
        filtered back to the model cutoff before evaluation.
    elements
        Optional allowed element symbols or atomic numbers. This is required
        only for legacy checkpoints that contain neither species metadata nor
        atomic reference energies.
    mamba_backend
        Optional runtime override: ``torch`` is the portable reference,
        ``auto`` selects an eligible fused CUDA kernel and otherwise falls back,
        and ``cuda`` requires fused CUDA execution.
    """

    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]
    nolabel = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        neighbor_skin: float = 0.3,
        elements: Sequence[str | int] | None = None,
        mamba_backend: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not math.isfinite(float(neighbor_skin)) or float(neighbor_skin) < 0.0:
            raise ValueError("neighbor_skin must be finite and nonnegative")
        if mamba_backend not in {None, "auto", "torch", "cuda"}:
            raise ValueError("mamba_backend must be None, 'auto', 'torch', or 'cuda'")
        self.device = torch.device(device) if device is not None else _automatic_device()
        if mamba_backend == "cuda" and self.device.type != "cuda":
            raise CalculatorSetupError("mamba_backend='cuda' requires a CUDA device")
        self.model, checkpoint = restore_model(
            checkpoint_path, self.device, mamba_backend=mamba_backend
        )
        self.r_max = float(self.model.r_max)
        if not math.isfinite(self.r_max) or self.r_max <= 0.0:
            raise CalculatorSetupError("The model cutoff must be positive and finite")
        self.atomic_energies = {
            int(key): float(value)
            for key, value in checkpoint.get("atomic_energies", {}).items()
        }
        invalid_references = sorted(
            number for number in self.atomic_energies if not 1 <= number <= 118
        )
        if invalid_references:
            raise CalculatorSetupError(
                f"Invalid atomic reference keys Z={invalid_references}"
            )
        if any(not math.isfinite(value) for value in self.atomic_energies.values()):
            raise CalculatorSetupError("Atomic reference energies must be finite")
        checkpoint_species = checkpoint.get("atomic_numbers")
        if checkpoint_species:
            trained_species = _normalize_elements(checkpoint_species)
        elif self.atomic_energies:
            trained_species = frozenset(self.atomic_energies)
        else:
            trained_species = None
        if elements is not None:
            requested_species = _normalize_elements(elements)
            if trained_species is not None:
                unsupported = sorted(requested_species - trained_species)
                if unsupported:
                    raise CalculatorSetupError(
                        f"Requested elements were not present during training: Z={unsupported}"
                    )
            self.allowed_atomic_numbers = requested_species
        else:
            self.allowed_atomic_numbers = trained_species
        if self.atomic_energies and self.allowed_atomic_numbers is not None:
            missing_references = sorted(
                self.allowed_atomic_numbers - set(self.atomic_energies)
            )
            if missing_references:
                raise CalculatorSetupError(
                    f"Missing atomic reference energies for Z={missing_references}"
                )
        try:
            self.dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self.dtype = torch.get_default_dtype()
        if self.dtype not in {torch.float32, torch.float64}:
            raise CalculatorSetupError(
                f"ASE deployment requires float32 or float64 model weights, got {self.dtype}"
            )
        # Pairs closer than the inner shell radius all fold onto shell 0, so the
        # mixer loses every bit of radial resolution below it -- precisely where
        # the potential is steepest and where molecular dynamics goes unstable.
        # The direct ACE path still resolves them, so the degradation is silent
        # unless something says so.  Training validates this against the dataset;
        # at run time the sampled geometry is whatever the dynamics produces.
        self.shell_r_min = float(getattr(self.model.ace, "shell_r_min", 0.0) or 0.0)
        self._warned_below_shell_r_min = False
        self.neighbor_skin = float(neighbor_skin)
        self._neighbor_list: NeighborList | None = None
        self._neighbor_count = -1
        self._lock = threading.RLock()
        references = torch.zeros(119, device=self.device, dtype=self.dtype)
        for atomic_number, value in self.atomic_energies.items():
            if not 1 <= atomic_number <= 118:
                raise CalculatorSetupError(
                    f"Invalid atomic reference key Z={atomic_number}"
                )
            references[atomic_number] = value
        self._reference_energies = references

    def _ensure_neighbor_list(self, atoms) -> NeighborList:
        if self._neighbor_list is None or self._neighbor_count != len(atoms):
            self._neighbor_list = NeighborList(
                np.full(len(atoms), 0.5 * self.r_max),
                skin=self.neighbor_skin,
                self_interaction=False,
                bothways=True,
                sorted=True,
                primitive=NewPrimitiveNeighborList,
            )
            self._neighbor_count = len(atoms)
        self._neighbor_list.update(atoms)
        return self._neighbor_list

    def _neighbor_tensors(self, atoms) -> tuple[torch.Tensor, torch.Tensor]:
        neighbor_list = self._ensure_neighbor_list(atoms)
        senders: list[int] = []
        receivers: list[int] = []
        shifts: list[np.ndarray] = []
        for receiver in range(len(atoms)):
            neighbors, offsets = neighbor_list.get_neighbors(receiver)
            for sender, offset in zip(neighbors, offsets):
                senders.append(int(sender))
                receivers.append(receiver)
                shifts.append(np.asarray(offset, dtype=np.int64))
        if senders:
            edge_index = torch.tensor(
                [senders, receivers], dtype=torch.long, device=self.device
            )
            edge_shift = torch.as_tensor(
                np.asarray(shifts), dtype=self.dtype, device=self.device
            )
            positions = torch.as_tensor(
                atoms.positions, dtype=self.dtype, device=self.device
            )
            cell = torch.as_tensor(
                atoms.cell.array, dtype=self.dtype, device=self.device
            )
            edge_vector = (
                positions[edge_index[0]]
                - positions[edge_index[1]]
                + edge_shift @ cell
            )
            # Filter the Verlet candidates using exactly the dtype and arithmetic
            # used by the model, including at a floating-point cutoff boundary.
            lengths = torch.linalg.vector_norm(edge_vector, dim=-1)
            inside = lengths < self.r_max
            edge_index = edge_index[:, inside]
            edge_shift = edge_shift[inside]
            self._check_shell_resolution(lengths[inside])
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_shift = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        return edge_index, edge_shift

    def _check_shell_resolution(self, lengths: torch.Tensor) -> None:
        """Warn once if the geometry drops below the model's inner shell radius."""

        if self.shell_r_min <= 0.0 or self._warned_below_shell_r_min:
            return
        if lengths.numel() == 0:
            return
        shortest = float(lengths.min())
        if shortest < self.shell_r_min:
            self._warned_below_shell_r_min = True
            warnings.warn(
                f"Interatomic distance {shortest:.3f} A is below the model's "
                f"shell_r_min={self.shell_r_min:.3f} A. Every pair closer than "
                "that folds onto the innermost radial shell, so the sequence "
                "mixer has no radial resolution there and the model is "
                "extrapolating in exactly the region where the potential is "
                "steepest. Check the trajectory for close contacts, or retrain "
                "with a smaller shell_r_min. This warning is issued once.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _validate_atoms(self, atoms, properties: Sequence[str]) -> None:
        if len(atoms) == 0:
            raise CalculatorSetupError("MTACE cannot evaluate an empty Atoms object")
        numbers = np.asarray(atoms.numbers)
        if np.any(numbers < 1) or np.any(numbers > 118):
            raise CalculatorSetupError("Atomic numbers must satisfy 1 <= Z <= 118")
        if self.allowed_atomic_numbers is None:
            raise CalculatorSetupError(
                "This legacy checkpoint has no trained-species metadata; "
                "construct MambaACECalculator(..., elements=[...]) explicitly"
            )
        missing = sorted(set(map(int, numbers)) - self.allowed_atomic_numbers)
        if missing:
            raise CalculatorSetupError(
                f"Atomic species are outside the checkpoint training set: Z={missing}"
            )
        if not np.isfinite(atoms.positions).all():
            raise CalculatorSetupError("Atomic positions must be finite")
        cell = np.asarray(atoms.cell.array)
        if not np.isfinite(cell).all():
            raise CalculatorSetupError("Cell vectors must be finite")
        periodic_lengths = np.linalg.norm(cell, axis=1)[np.asarray(atoms.pbc, dtype=bool)]
        if periodic_lengths.size and np.any(periodic_lengths <= 0.0):
            raise CalculatorSetupError("Every periodic direction requires a nonzero cell vector")
        if "stress" in properties and atoms.cell.rank != 3:
            raise PropertyNotImplementedError(
                "Stress requires a full-rank three-dimensional cell"
            )
        if "stress" in properties:
            volume = float(atoms.get_volume())
            if not math.isfinite(volume) or volume <= 0.0:
                raise CalculatorSetupError("Stress requires a positive finite cell volume")

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        with self._lock:
            super().calculate(atoms, properties, system_changes)
            self._validate_atoms(self.atoms, properties)
            edge_index, edge_shift = self._neighbor_tensors(self.atoms)
            z = torch.as_tensor(
                self.atoms.numbers, dtype=torch.long, device=self.device
            )
            pos = torch.as_tensor(
                self.atoms.positions, dtype=self.dtype, device=self.device
            )
            cell = torch.as_tensor(
                self.atoms.cell.array, dtype=self.dtype, device=self.device
            )
            compute_stress = "stress" in properties
            data = {
                "z": z,
                "pos": pos,
                "cell": cell,
                "edge_index": edge_index,
                "edge_shift": edge_shift,
                "volume": torch.as_tensor(
                    self.atoms.get_volume() if self.atoms.cell.rank == 3 else 1.0,
                    dtype=self.dtype,
                    device=self.device,
                ),
            }
            compute_forces = "forces" in properties or compute_stress
            if compute_forces:
                with torch.enable_grad():
                    _, forces, stress, extra = self.model(
                        data, training=False, compute_stress=compute_stress
                    )
                learned_atomic_energy = extra["atomic_energy"]
            else:
                with torch.no_grad():
                    learned_atomic_energy = self.model.atomic_energies(
                        z, pos, cell, edge_index, edge_shift
                    )
                forces = None
                stress = None
            atomic_energy = learned_atomic_energy + self._reference_energies[z]
            total_energy = atomic_energy.sum()
            if not bool(torch.isfinite(atomic_energy).all()) or not bool(
                torch.isfinite(total_energy)
            ):
                raise CalculationFailed("MTACE produced a nonfinite atomic energy")
            if forces is not None and not bool(torch.isfinite(forces).all()):
                raise CalculationFailed("MTACE produced a nonfinite force")
            if stress is not None and not bool(torch.isfinite(stress).all()):
                raise CalculationFailed("MTACE produced a nonfinite stress")
            energy_value = float(total_energy.detach().cpu())
            results = {
                "energy": energy_value,
                "free_energy": energy_value,
                "energies": atomic_energy.detach().cpu().numpy().astype(
                    np.float64, copy=False
                ),
            }
            if forces is not None:
                results["forces"] = forces.detach().cpu().numpy().astype(
                    np.float64, copy=False
                )
            if compute_stress:
                matrix = stress.detach().cpu().numpy()
                results["stress"] = np.asarray(
                    [
                        matrix[0, 0],
                        matrix[1, 1],
                        matrix[2, 2],
                        matrix[1, 2],
                        matrix[0, 2],
                        matrix[0, 1],
                    ],
                    dtype=np.float64,
                )
            self.results = results
