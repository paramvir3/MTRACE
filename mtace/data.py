"""ASE data conversion and trajectory splitting utilities."""

from __future__ import annotations

import math

import numpy as np
import torch
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset


def build_neighbor_tensors(
    atoms,
    r_max: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build directed ASE edges and enforce the cutoff in model arithmetic."""

    if not math.isfinite(float(r_max)) or float(r_max) <= 0.0:
        raise ValueError("r_max must be positive and finite")
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("AtomisticDataset supports float32 or float64 tensors")
    center, neighbor, shifts = neighbor_list("ijS", atoms, r_max)
    edge_index = torch.stack(
        (torch.as_tensor(neighbor, dtype=torch.long), torch.as_tensor(center, dtype=torch.long))
    )
    edge_shift = torch.as_tensor(shifts, dtype=dtype)
    if edge_index.shape[1]:
        positions = torch.as_tensor(np.asarray(atoms.positions), dtype=dtype)
        cell = torch.as_tensor(np.asarray(atoms.cell.array), dtype=dtype)
        edge_vector = (
            positions[edge_index[0]]
            - positions[edge_index[1]]
            + edge_shift @ cell
        )
        inside = torch.linalg.vector_norm(edge_vector, dim=-1) < float(r_max)
        edge_index = edge_index[:, inside]
        edge_shift = edge_shift[inside]
    return edge_index, edge_shift


def average_num_neighbors(frames, r_max: float) -> float:
    """Mean number of directed in-cutoff neighbors per atom over ``frames``.

    Used to normalize the ACE density.  Because the correlation of order ``nu``
    is a product of ``nu - 1`` copies of the density, an unnormalized model has
    activations that grow polynomially with coordination, which degrades transfer
    between phases, surfaces and pressures.
    """

    if not frames:
        raise ValueError("cannot estimate avg_num_neighbors from an empty set")
    edges = 0
    atoms_total = 0
    for atoms in frames:
        edge_index, _ = build_neighbor_tensors(atoms, r_max, torch.float64)
        edges += int(edge_index.shape[1])
        atoms_total += len(atoms)
    if atoms_total == 0:
        raise ValueError("cannot estimate avg_num_neighbors without atoms")
    # An isolated-atom dataset would otherwise produce a zero normalization.
    return max(1.0, edges / atoms_total)


def shell_occupancy(frames, r_max: float, shell_basis) -> np.ndarray:
    """Fraction of directed edges that place nonzero weight on each shell.

    Uniform shells on ``[0, r_max]`` waste a large part of the sequence: no pair
    of atoms sits below the repulsive wall, so the innermost shells are
    identically zero for every structure and the mixer scans hard zeros.  This
    helper quantifies that and is the basis for choosing ``shell_r_min``.
    """

    occupancy = np.zeros(int(shell_basis.num_shells))
    total = 0
    for atoms in frames:
        edge_index, edge_shift = build_neighbor_tensors(atoms, r_max, torch.float64)
        if edge_index.shape[1] == 0:
            continue
        positions = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float64)
        cell = torch.as_tensor(np.asarray(atoms.cell.array), dtype=torch.float64)
        vector = (
            positions[edge_index[0]] - positions[edge_index[1]] + edge_shift @ cell
        )
        distance = torch.linalg.vector_norm(vector, dim=-1)
        weights = shell_basis.dense(distance)
        occupancy += (weights.abs() > 1.0e-12).double().sum(0).numpy()
        total += int(distance.numel())
    return occupancy / max(1, total)


def minimum_edge_distance(frames, r_max: float) -> float:
    """Shortest interatomic distance inside the cutoff over a set of frames.

    Used to validate ``shell_r_min``.  Every pair closer than the inner shell
    radius is folded onto shell zero, so the mixer loses all radial resolution
    below it -- exactly where the potential is steepest and where molecular
    dynamics goes unstable.  The direct ACE path still resolves those pairs, so
    the failure is silent rather than catastrophic, which is why it needs an
    explicit check.
    """

    shortest = math.inf
    for atoms in frames:
        if len(atoms) < 2:
            continue
        _, distances = neighbor_list("id", atoms, float(r_max))
        if distances.size:
            shortest = min(shortest, float(distances.min()))
    if not math.isfinite(shortest):
        raise ValueError(
            "no interatomic distance below r_max; cannot validate shell_r_min"
        )
    return shortest


def target_statistics(frames) -> dict[str, float]:
    """Dataset spread of each supervised quantity, for loss normalization.

    Raw loss weights are dimensionful: with the values shipped in earlier
    examples the energy term sat four orders of magnitude below the force term,
    so energies were effectively unconstrained.  Dividing each term by the
    dataset standard deviation of its target makes the weights dimensionless and
    comparable, and therefore transferable between systems.

    The energy scale is the standard deviation of the *per-atom* energy after
    removing a per-species least-squares reference, which is the quantity the
    loss actually penalizes.  Force and stress scales are root-mean-square
    component magnitudes, which for a mean-zero quantity is its spread.
    """

    per_atom, force_square, force_count = [], 0.0, 0
    stress_square, stress_count = 0.0, 0
    species = sorted({int(z) for atoms in frames for z in atoms.numbers})
    counts, energies = [], []
    for atoms in frames:
        counts.append([int(np.count_nonzero(atoms.numbers == z)) for z in species])
        energies.append(float(atoms.get_potential_energy()))
        forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
        force_square += float(np.square(forces).sum())
        force_count += forces.size
        stress, has_stress = stress_target(atoms)
        if has_stress:
            stress_square += float(np.square(stress).sum())
            stress_count += stress.size
    count_matrix = np.asarray(counts, dtype=float)
    energy_vector = np.asarray(energies, dtype=float)
    if np.linalg.matrix_rank(count_matrix) < len(species):
        # Fixed composition: only a single per-atom offset is identifiable.
        offsets = count_matrix.sum(axis=1)
        residual = energy_vector - offsets * (energy_vector.sum() / offsets.sum())
    else:
        coefficients, *_ = np.linalg.lstsq(count_matrix, energy_vector, rcond=None)
        residual = energy_vector - count_matrix @ coefficients
    per_atom = residual / count_matrix.sum(axis=1)
    energy_scale = float(np.std(per_atom))
    return {
        "energy_ev_per_atom": energy_scale if energy_scale > 0.0 else 1.0,
        "forces_ev_per_angstrom": (
            math.sqrt(force_square / force_count) if force_count else 1.0
        ),
        "stress_ev_per_angstrom3": (
            math.sqrt(stress_square / stress_count) if stress_count else 1.0
        ),
        "stress_structures": stress_count // 9,
    }


def stress_matrix(value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (6,):
        xx, yy, zz, yz, xz, xy = array
        array = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    elif array.size == 9:
        array = array.reshape(3, 3)
    else:
        raise ValueError(f"Stress must contain 6 or 9 components, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Stress and virial labels must be finite")
    antisymmetric = 0.5 * (array - array.T)
    symmetry_tolerance = 1.0e-8 + 1.0e-7 * float(np.abs(array).max(initial=0.0))
    if float(np.abs(antisymmetric).max(initial=0.0)) > symmetry_tolerance:
        raise ValueError(
            "Stress and virial labels must be symmetric; maximum antisymmetric "
            f"component is {np.abs(antisymmetric).max():.6g}"
        )
    return 0.5 * (array + array.T)


def stress_volume(atoms) -> float:
    if atoms.cell.rank != 3:
        raise ValueError("Stress and virial labels require a full-rank cell")
    volume = float(atoms.get_volume())
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("Stress and virial labels require a positive finite cell volume")
    return volume


def stress_target(atoms) -> tuple[np.ndarray, bool]:
    results = atoms.calc.results if atoms.calc is not None else {}
    candidates = []
    for source_name, source in (("calculator", results), ("atoms.info", atoms.info)):
        if "stress" in source:
            stress_volume(atoms)
            candidates.append((f"{source_name} stress", stress_matrix(source["stress"])))
        if "virial" in source:
            candidates.append(
                (
                    f"{source_name} virial",
                    -stress_matrix(source["virial"]) / stress_volume(atoms),
                )
            )
    if candidates:
        name, target = candidates[0]
        for other_name, other in candidates[1:]:
            if not np.allclose(target, other, rtol=1.0e-6, atol=1.0e-8):
                difference = float(np.max(np.abs(target - other)))
                raise ValueError(
                    f"Inconsistent {name} and {other_name} labels; maximum stress "
                    f"difference is {difference:.6g} eV/Angstrom^3"
                )
        return target, True
    return np.zeros((3, 3)), False


def stress_to_voigt(stress: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (stress[0, 0], stress[1, 1], stress[2, 2], stress[1, 2], stress[0, 2], stress[0, 1])
    )


def stress_to_mandel(stress: torch.Tensor) -> torch.Tensor:
    """Map a symmetric tensor to a norm-preserving six-vector."""

    shear_scale = stress.new_tensor(2.0).sqrt()
    return torch.stack(
        (
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            shear_scale * stress[1, 2],
            shear_scale * stress[0, 2],
            shear_scale * stress[0, 1],
        )
    )


def stress_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rotationally invariant mean-square error for symmetric stress tensors."""

    return stress_to_mandel(prediction - target).square().mean()


class AtomisticDataset(Dataset):
    def __init__(
        self,
        atoms_list,
        r_max: float,
        precompute_neighbors: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        self.atoms_list = list(atoms_list)
        if not self.atoms_list:
            raise ValueError("AtomisticDataset requires at least one structure")
        self.r_max = float(r_max)
        if not math.isfinite(self.r_max) or self.r_max <= 0.0:
            raise ValueError("r_max must be positive and finite")
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("AtomisticDataset supports float32 or float64 tensors")
        self.dtype = dtype
        self.targets = [
            self._validated_target(atoms, index)
            for index, atoms in enumerate(self.atoms_list)
        ]
        self.has_stress_count = sum(target["has_stress"] for target in self.targets)
        self.edge_cache = (
            [build_neighbor_tensors(atoms, r_max, dtype) for atoms in self.atoms_list]
            if precompute_neighbors
            else None
        )

    @staticmethod
    def _validated_target(atoms, index: int) -> dict:
        context = f"structure {index}"
        if len(atoms) < 1:
            raise ValueError(f"{context} contains no atoms")
        numbers = np.asarray(atoms.numbers)
        if numbers.shape != (len(atoms),) or np.any(numbers < 1) or np.any(numbers > 118):
            raise ValueError(f"{context} has atomic numbers outside 1 <= Z <= 118")
        positions = np.asarray(atoms.positions, dtype=float)
        if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
            raise ValueError(f"{context} positions must have shape (N, 3) and be finite")
        cell = np.asarray(atoms.cell.array, dtype=float)
        if cell.shape != (3, 3) or not np.isfinite(cell).all():
            raise ValueError(f"{context} cell must have shape (3, 3) and be finite")
        periodic_lengths = np.linalg.norm(cell, axis=1)[np.asarray(atoms.pbc, dtype=bool)]
        if periodic_lengths.size and np.any(periodic_lengths <= 0.0):
            raise ValueError(f"{context} has a periodic direction without a cell vector")

        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
        except Exception as exception:
            raise ValueError(
                f"{context} must provide potential-energy and force labels"
            ) from exception
        if not math.isfinite(energy):
            raise ValueError(f"{context} energy label must be finite")
        if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise ValueError(f"{context} forces must have shape (N, 3) and be finite")
        try:
            stress, has_stress = stress_target(atoms)
        except ValueError as exception:
            raise ValueError(f"{context}: {exception}") from exception
        volume = (
            stress_volume(atoms)
            if has_stress
            else (float(atoms.get_volume()) if atoms.cell.rank == 3 else 1.0)
        )
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError(f"{context} volume must be positive and finite")
        return {
            "energy": energy,
            "forces": forces.copy(),
            "stress": stress.copy(),
            "has_stress": bool(has_stress),
            "volume": volume,
        }

    def __len__(self) -> int:
        return len(self.atoms_list)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        atoms = self.atoms_list[index]
        edge_index, edge_shift = (
            self.edge_cache[index]
            if self.edge_cache is not None
            else build_neighbor_tensors(atoms, self.r_max, self.dtype)
        )
        target = self.targets[index]
        return {
            "z": torch.as_tensor(atoms.numbers, dtype=torch.long),
            "pos": torch.as_tensor(atoms.positions, dtype=self.dtype),
            "cell": torch.as_tensor(atoms.cell.array, dtype=self.dtype),
            "edge_index": edge_index,
            "edge_shift": edge_shift,
            "volume": torch.tensor(target["volume"], dtype=self.dtype),
            "target_energy": torch.tensor(target["energy"], dtype=self.dtype),
            "target_forces": torch.as_tensor(target["forces"], dtype=self.dtype),
            "target_stress": torch.as_tensor(target["stress"], dtype=self.dtype),
            "has_stress": torch.tensor(target["has_stress"], dtype=torch.bool),
        }

    @staticmethod
    def collate(batch):
        return batch


def split_frames(
    frames,
    validation_fraction: float = 0.1,
    seed: int = 42,
    mode: str = "blocked",
    block_size: int = 25,
    gap: int = 0,
):
    n_frames = len(frames)
    if n_frames < 2:
        raise ValueError("At least two frames are required")
    if not math.isfinite(float(validation_fraction)) or not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    if int(block_size) < 1:
        raise ValueError("block_size must be positive")
    if int(gap) < 0:
        raise ValueError("gap must be nonnegative")
    n_validation = max(1, min(n_frames - 1, round(validation_fraction * n_frames)))
    generator = torch.Generator().manual_seed(seed)
    if mode == "random":
        validation_indices = torch.randperm(n_frames, generator=generator)[:n_validation].tolist()
    elif mode == "blocked":
        block_size = min(max(1, int(block_size)), max(1, n_frames // 2))
        blocks = [list(range(i, min(i + block_size, n_frames))) for i in range(0, n_frames, block_size)]
        validation_indices = []
        for block_index in torch.randperm(len(blocks), generator=generator).tolist():
            if len(validation_indices) >= n_validation:
                break
            if len(validation_indices) + len(blocks[block_index]) >= n_frames:
                # Skip a block that would leave no training frames; do not abandon
                # the remaining blocks, which may still fit.
                continue
            validation_indices.extend(blocks[block_index])
    else:
        raise ValueError("split mode must be 'blocked' or 'random'")
    validation_set = set(validation_indices)
    excluded = set()
    for index in validation_indices:
        excluded.update(range(max(0, index - gap), min(n_frames, index + gap + 1)))
    excluded.difference_update(validation_set)
    training_indices = [i for i in range(n_frames) if i not in validation_set and i not in excluded]
    if not training_indices or not validation_indices:
        raise ValueError("Split produced an empty training or validation set")
    return (
        [frames[i] for i in training_indices],
        [frames[i] for i in sorted(validation_indices)],
        sorted(excluded),
    )


def solve_atomic_energies(frames) -> dict[int, float]:
    if not frames:
        raise ValueError("Cannot solve atomic energies from an empty training set")
    species = sorted({int(z) for atoms in frames for z in atoms.numbers})
    counts = np.array(
        [[np.count_nonzero(atoms.numbers == z) for z in species] for atoms in frames], dtype=float
    )
    energies = np.array([atoms.get_potential_energy() for atoms in frames], dtype=float)
    if not np.isfinite(energies).all():
        raise ValueError("Atomic reference fitting requires finite training energies")
    rank = int(np.linalg.matrix_rank(counts))
    if rank < len(species):
        raise ValueError(
            "Per-species atomic reference energies are not identifiable: the "
            f"species-count matrix has rank {rank} for {len(species)} species. "
            "Provide atomic_energies explicitly or set solve_atomic_energies: false."
        )
    coefficients, *_ = np.linalg.lstsq(counts, energies, rcond=None)
    return {z: float(value) for z, value in zip(species, coefficients)}


def parse_atomic_energies(table) -> dict[int, float]:
    parsed = {}
    for key, value in (table or {}).items():
        z = atomic_numbers[key] if isinstance(key, str) and not key.isdigit() else int(key)
        energy = float(value)
        if not 1 <= z <= 118:
            raise ValueError(f"Atomic reference number must satisfy 1 <= Z <= 118, got {z}")
        if not math.isfinite(energy):
            raise ValueError(f"Atomic reference energy for Z={z} must be finite")
        parsed[z] = energy
    return parsed


def baseline_energy(
    z: torch.Tensor,
    table: dict[int, float],
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    if not table:
        return torch.zeros((), device=z.device, dtype=dtype)
    missing = sorted(set(int(value) for value in z.detach().cpu().tolist()) - set(table))
    if missing:
        raise ValueError(f"Missing atomic reference energies for Z={missing}")
    maximum = max(table)
    values = torch.zeros(maximum + 1, device=z.device, dtype=dtype)
    for atomic_number, energy in table.items():
        values[atomic_number] = energy
    return values[z].sum()


def collate_structures(items, device=None):
    """Concatenate structures into one disconnected graph for a batched forward.

    Atom indices in ``edge_index`` are offset per structure so no edge crosses a
    structure boundary.  Because nothing in the model mixes atoms except through
    ``edge_index``, the batched evaluation is mathematically identical to
    evaluating each structure separately -- the same property verified by the
    additivity check ``E(A u B) = E(A) + E(B)``.

    Returns a dict carrying ``batch`` (atom -> structure), a stacked
    ``(num_graphs, 3, 3)`` cell, and per-structure ``volume``.
    """

    import torch as _torch

    if not items:
        raise ValueError("collate_structures needs at least one structure")
    z, pos, cells, shifts, edges, batch, volumes = [], [], [], [], [], [], []
    offset = 0
    for graph, item in enumerate(items):
        n = int(item["z"].numel())
        z.append(item["z"])
        pos.append(item["pos"])
        cells.append(item["cell"].reshape(1, 3, 3))
        edges.append(item["edge_index"] + offset)
        shifts.append(item["edge_shift"])
        batch.append(_torch.full((n,), graph, dtype=_torch.long))
        volume = item.get("volume")
        volumes.append(
            _torch.as_tensor(float(volume)) if volume is not None
            else _torch.det(item["cell"]).abs().reshape(())
        )
        offset += n
    out = {
        "z": _torch.cat(z),
        "pos": _torch.cat(pos),
        "cell": _torch.cat(cells),
        "edge_index": _torch.cat(edges, dim=1),
        "edge_shift": _torch.cat(shifts),
        "batch": _torch.cat(batch),
        "volume": _torch.stack(volumes).to(_torch.cat(pos).dtype),
    }
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out
