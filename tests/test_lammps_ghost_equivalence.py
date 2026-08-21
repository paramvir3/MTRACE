"""Numerical equivalence of the LAMMPS deployment path and the ASE calculator.

The LAMMPS integration tests in ``test_lammps_interfaces.py`` need a built
``pair_mamba``, so they skip on any machine without one.  This module tests the
same contract *without* LAMMPS, by reproducing exactly what ``pair_mamba.cpp``
does to a periodic cell:

* atomic energies are evaluated on unwrapped local **and ghost** coordinates,
  with no cell and no image shifts passed to the model;
* the differentiated scalar is the sum over **owned** atoms only;
* forces are ``-dE/dx`` over local *and* ghost coordinates, with ghost
  contributions folded back onto their owners, which is what LAMMPS reverse
  communication does;
* the virial is ``-sym(dE/d(strain))`` with the strain applied to every
  coordinate.

If any of that is wrong the model is still self-consistent but LAMMPS and ASE
disagree, which is the failure mode this catches.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms

from mtace.calculator import MambaACECalculator
from mtace.checkpoint import save_checkpoint
from mtace.deployment import export_lammps_model
from mtace.model import MambaACEV2

R_MAX = 5.0


def _cell_and_model(tmp_path, **overrides):
    torch.manual_seed(4)
    config = dict(
        r_max=R_MAX, l_max=2, num_radial=6, hidden_dim=16, num_layers=1,
        correlation_order=3, correlation_channels=8, num_shells=10,
        shell_r_min=1.0, shell_degree=5, avg_num_neighbors=12.0,
        invariant_norm="homogeneous", mamba_dim=16, mamba_d_state=8,
        mamba_headdim=8, readout_hidden=16, mamba_backend="torch",
    )
    config.update(overrides)
    model = MambaACEV2(**config)
    checkpoint = tmp_path / "model.pt"
    references = {1: -0.3, 8: -1.7}
    save_checkpoint(
        checkpoint, model, config, atomic_energies=references,
        atomic_numbers=[1, 8],
    )
    exported = export_lammps_model(
        checkpoint, tmp_path / "model.mtace.pt", ["H", "O"]
    )
    return checkpoint, exported


def _periodic_cell():
    generator = np.random.RandomState(0)
    lattice = 6.4
    positions = generator.uniform(0.4, lattice - 0.4, size=(8, 3))
    atoms = Atoms(
        "H4O4", positions=positions, cell=np.eye(3) * lattice, pbc=True
    )
    return atoms


def _local_and_ghosts(atoms, r_max):
    """Reproduce the local + ghost atom set that LAMMPS hands to the pair style."""

    positions = np.asarray(atoms.positions)
    cell = np.asarray(atoms.cell.array)
    numbers = np.asarray(atoms.numbers)
    n_local = len(atoms)

    ghost_positions, ghost_numbers, owners = [], [], []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                if (a, b, c) == (0, 0, 0):
                    continue
                shifted = positions + np.array([a, b, c], dtype=float) @ cell
                for index in range(n_local):
                    distance = np.linalg.norm(
                        shifted[index][None, :] - positions, axis=1
                    ).min()
                    if distance < r_max:
                        ghost_positions.append(shifted[index])
                        ghost_numbers.append(numbers[index])
                        owners.append(index)

    all_positions = np.vstack([positions, np.asarray(ghost_positions)])
    all_numbers = np.concatenate([numbers, np.asarray(ghost_numbers)])
    owner_index = np.concatenate(
        [np.arange(n_local), np.asarray(owners, dtype=int)]
    )

    senders, receivers = [], []
    for i in range(n_local):
        distance = np.linalg.norm(all_positions - all_positions[i], axis=1)
        for j in np.where((distance < r_max) & (distance > 0.0))[0]:
            senders.append(int(j))
            receivers.append(i)
    edges = torch.tensor([senders, receivers], dtype=torch.long)
    return all_positions, all_numbers, owner_index, edges, n_local


class TestLammpsGhostEquivalence:
    def test_energy_forces_and_virial_match_the_ase_calculator(self, tmp_path):
        checkpoint, exported = _cell_and_model(tmp_path)
        atoms = _periodic_cell()
        calculator = MambaACECalculator(checkpoint, device="cpu")
        atoms.calc = calculator
        reference_energy = float(atoms.get_potential_energy())
        reference_forces = atoms.get_forces()
        reference_stress = atoms.get_stress(voigt=False)
        volume = atoms.get_volume()

        all_positions, all_numbers, owners, edges, n_local = _local_and_ghosts(
            atoms, R_MAX
        )
        assert len(all_positions) > n_local, "the test cell must generate ghosts"

        module = torch.jit.load(str(exported))
        coordinates = torch.tensor(
            all_positions, dtype=torch.float32, requires_grad=True
        )
        species = torch.tensor(all_numbers, dtype=torch.long)
        owned_energy = module(species, coordinates, edges)[:n_local].sum()

        gradient = torch.autograd.grad(owned_energy, coordinates)[0].numpy()
        folded = np.zeros((n_local, 3))
        np.add.at(folded, owners, -gradient)

        strain = torch.zeros((3, 3), dtype=torch.float32, requires_grad=True)
        deformed = torch.tensor(all_positions, dtype=torch.float32) @ (
            torch.eye(3) + strain
        )
        strained_energy = module(species, deformed, edges)[:n_local].sum()
        strain_gradient = torch.autograd.grad(strained_energy, strain)[0]
        virial = (-0.5 * (strain_gradient + strain_gradient.T)).detach().numpy()

        assert float(owned_energy) == pytest.approx(reference_energy, abs=1e-4)
        assert np.abs(folded - reference_forces).max() < 1e-4
        # LAMMPS virial and ASE stress are related by Xi = -V sigma.
        assert np.abs(-virial / volume - reference_stress).max() < 1e-6

    def test_folded_ghost_forces_sum_to_zero(self, tmp_path):
        """Newton's third law after reverse communication.

        A nonzero net force would mean the ghost bookkeeping is wrong and an MD
        run would drift, which no energy comparison alone would reveal.
        """

        checkpoint, exported = _cell_and_model(tmp_path)
        atoms = _periodic_cell()
        all_positions, all_numbers, owners, edges, n_local = _local_and_ghosts(
            atoms, R_MAX
        )
        module = torch.jit.load(str(exported))
        coordinates = torch.tensor(
            all_positions, dtype=torch.float32, requires_grad=True
        )
        species = torch.tensor(all_numbers, dtype=torch.long)
        owned_energy = module(species, coordinates, edges)[:n_local].sum()
        gradient = torch.autograd.grad(owned_energy, coordinates)[0].numpy()
        folded = np.zeros((n_local, 3))
        np.add.at(folded, owners, -gradient)
        assert np.abs(folded.sum(axis=0)).max() < 1e-4

    def test_export_metadata_satisfies_the_pair_style_abi(self, tmp_path):
        """Every field pair_mamba.cpp::load_model reads must be present."""

        _, exported = _cell_and_model(tmp_path)
        required = {
            "mamba_ace_format_version": "1",
            "output": "atomic_energy",
            "strictly_local": "1",
            "energy_units": "eV",
            "length_units": "Angstrom",
            "elements": "H O",
            "atomic_numbers": "1 8",
            "dtype": "float32",
        }
        extra = {key: "" for key in
                 list(required) + ["architecture", "architecture_version", "r_max"]}
        torch.jit.load(str(exported), _extra_files=extra)
        for key, expected in required.items():
            value = extra[key]
            value = value.decode() if isinstance(value, bytes) else value
            assert value == expected, f"{key}: {value!r} != {expected!r}"
        r_max = extra["r_max"]
        r_max = r_max.decode() if isinstance(r_max, bytes) else r_max
        assert float(r_max) == pytest.approx(R_MAX)
