"""Thermodynamic-integration equations of the CsPbI3 free-energy workflow."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

_PATH = Path(__file__).resolve().parents[1] / "studies" / "cspbi3" / "free_energy_ti" / "ti_core.py"
_SPEC = importlib.util.spec_from_file_location("ti_core", _PATH)
ti = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ti)


def _cubic_perovskite(a=6.3):
    """Ideal 5-atom cubic ABX3 cell: one lattice point, no centring."""

    from ase import Atoms

    return Atoms(
        "CsPbI3",
        scaled_positions=[
            (0.0, 0.0, 0.0), (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
        ],
        cell=np.eye(3) * a,
        pbc=True,
    )


class TestLatticeTranslations:
    def test_repeated_primitive_cell_counts_its_repetitions(self):
        atoms = _cubic_perovskite().repeat((2, 3, 2))
        count = ti.lattice_translation_count(
            atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers
        )
        assert count == 12

    def test_centring_translations_are_counted(self):
        # Conventional fcc cell: four lattice points; repeated 2 x 2 x 2.
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
        count = ti.lattice_translation_count(
            atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers
        )
        assert count == len(atoms) == 32

    def test_a_structure_without_translational_symmetry_counts_one(self):
        rng = np.random.default_rng(0)
        atoms = _cubic_perovskite().repeat((2, 2, 2))
        atoms.positions += rng.normal(scale=0.2, size=atoms.positions.shape)
        count = ti.lattice_translation_count(
            atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers
        )
        assert count == 1

    def test_tolerance_is_in_angstrom(self):
        atoms = _cubic_perovskite().repeat((2, 1, 1))
        atoms.positions[0] += [0.003, 0.0, 0.0]
        args = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
        assert ti.lattice_translation_count(*args, tolerance_A=1e-2) == 2
        assert ti.lattice_translation_count(*args, tolerance_A=1e-3) == 1


class TestEinsteinCentreOfMass:
    def test_monatomic_limit_is_the_polson_correction(self):
        """F = F_E + k T ln[(N/V) (2 pi k T / (N k))^(3/2)] / N for n_t = N."""

        n, temperature, volume, mass, spring = 32, 450.0, 400.0, 63.546, 2.5
        value = ti.einstein_free_energy_per_atom(
            temperature, volume, [mass] * n, [spring] * n, lattice_translations=n
        )
        kt_j = ti.BOLTZMANN_J_K * temperature
        k_si = spring * ti.ELECTRON_VOLT_J / ti.ANGSTROM_M**2
        m_si = mass * ti.ATOMIC_MASS_KG
        hbar_omega = ti.PLANCK_J_S / (2 * math.pi) * math.sqrt(k_si / m_si)
        f_einstein = 3.0 * kt_j * math.log(hbar_omega / kt_j)
        v_si = volume * ti.ANGSTROM_M**3
        f_com = kt_j * math.log((n / v_si) * (2 * math.pi * kt_j / (n * k_si)) ** 1.5)
        expected = (n * f_einstein + f_com) / n / ti.ELECTRON_VOLT_J
        assert value == pytest.approx(expected, rel=1e-12)

    def test_omitting_the_count_shifts_the_free_energy_by_kt_ln_n_over_n(self):
        n, temperature = 240, 450.0
        args = (temperature, 7000.0, [132.9] * n, [1.0] * n)
        with_count = ti.einstein_free_energy_per_atom(*args, lattice_translations=12)
        without = ti.einstein_free_energy_per_atom(*args, lattice_translations=1)
        assert with_count - without == pytest.approx(
            ti.KB_EV_K * temperature * math.log(12) / n, rel=1e-12
        )

    def test_the_count_is_required_and_validated(self):
        args = (300.0, 100.0, [1.0, 2.0], [1.0, 1.0])
        with pytest.raises(TypeError):
            ti.einstein_free_energy_per_atom(*args)
        with pytest.raises(ValueError, match="lattice_translations"):
            ti.einstein_free_energy_per_atom(*args, lattice_translations=0)
        with pytest.raises(ValueError, match="lattice_translations"):
            ti.einstein_free_energy_per_atom(*args, lattice_translations=3)
