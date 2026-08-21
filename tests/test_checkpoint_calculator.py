import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import CalculatorSetupError
from ase.md.verlet import VelocityVerlet
from ase.neighborlist import NewPrimitiveNeighborList
from ase.optimize import FIRE
from ase.units import fs

from mtace.calculator import MambaACECalculator
from mtace.checkpoint import load_checkpoint, restore_model, save_checkpoint
from mtace.data import baseline_energy
from mtace.model import MambaACE, MambaACEV2


class CalculatorTests(unittest.TestCase):
    @staticmethod
    def _config():
        return {
            "r_max": 3.5,
            "l_max": 1,
            "num_radial": 3,
            "hidden_dim": 6,
            "num_layers": 1,
            "correlation_order": 3,
            "correlation_channels": 3,
            "mamba_dim": 8,
            "mamba_d_state": 4,
            "mamba_backend": "torch",
        }

    def test_checkpoint_round_trip_through_ase(self):
        config = {
            "r_max": 3.5,
            "l_max": 1,
            "num_radial": 3,
            "hidden_dim": 6,
            "num_layers": 1,
            "correlation_order": 3,
            "correlation_channels": 3,
            "mamba_dim": 8,
            "mamba_d_state": 4,
            "mamba_backend": "torch",
        }
        network = MambaACE(**config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            save_checkpoint(checkpoint, network, config, atomic_energies={1: -1.0, 8: -2.0})
            payload = load_checkpoint(checkpoint)
            self.assertEqual(payload["architecture"], "mtace_v2")
            self.assertEqual(payload["architecture_version"], 11)
            atoms = Atoms(
                "H2O",
                positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.2, 0.92, 0.0]],
            )
            atoms.calc = MambaACECalculator(str(checkpoint), device="cpu")
            self.assertIsInstance(atoms.calc.model, MambaACEV2)
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            self.assertTrue(np.isfinite(energy))
            self.assertTrue(np.isfinite(forces).all())
            np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=2.0e-5)

    def test_v2_checkpoint_migrates_to_tied_mamba1_compatibility_mode(self):
        config = {
            "r_max": 3.5,
            "l_max": 1,
            "num_radial": 3,
            "hidden_dim": 6,
            "num_layers": 1,
            "correlation_order": 3,
            "correlation_channels": 3,
            "mamba_dim": 8,
            "mamba_d_state": 3,
            "mamba_backend": "torch",
        }
        old_model = MambaACE(
            **config,
            tokenizer_type="legacy_basis",
            num_shells=config["num_radial"],
            mamba_variant="mamba1",
            mamba_bidirectional_tied=True,
            mamba_mimo_rank=1,
        )
        old_state = dict(old_model.state_dict())
        old_state["species_embedding.weight"] = old_state["species_embedding.weight"][:118]
        payload = {
            "format_version": 1,
            "architecture": "mtace_v2",
            "architecture_version": 2,
            "model_config": config,
            "training_config": {},
            "model_state_dict": old_state,
            "optimizer_state_dict": None,
            "epoch": 0,
            "atomic_energies": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "old.pt"
            torch.save(payload, checkpoint)
            calculator = MambaACECalculator(str(checkpoint), device="cpu")
        self.assertTrue(calculator.model.layers[0].mixer.bidirectional_tied)
        self.assertEqual(calculator.model.layers[0].mixer.__class__.__name__, "MambaSequenceMixer")
        self.assertEqual(calculator.model.ace.tokenizer_type, "legacy_basis")
        self.assertEqual(calculator.model.ace.sequence_length, config["num_radial"])
        self.assertEqual(calculator.model.species_embedding.num_embeddings, 119)

    def test_v4_mamba3_checkpoint_preserves_legacy_angle_equation(self):
        config = self._config()
        old_model = MambaACE(
            **config,
            tokenizer_type="legacy_basis",
            num_shells=config["num_radial"],
            mamba_angle_mode="legacy_bounded",
            mamba_mimo_rank=1,
        )
        payload = {
            "format_version": 1,
            "architecture": "mtace_v2",
            "architecture_version": 4,
            "model_config": config,
            "training_config": {},
            "model_state_dict": old_model.state_dict(),
            "optimizer_state_dict": None,
            "epoch": 0,
            "atomic_energies": {1: -1.0},
            "atomic_numbers": [1],
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v4.pt"
            torch.save(payload, checkpoint)
            calculator = MambaACECalculator(checkpoint, device="cpu")
        direction = calculator.model.layers[0].mixer.forward_direction
        self.assertEqual(direction.angle_mode, "legacy_bounded")
        self.assertEqual(direction.mimo_rank, 1)

    def test_v5_checkpoint_defaults_to_the_official_angle_equation(self):
        config = self._config()
        old_model = MambaACE(
            **config,
            tokenizer_type="legacy_basis",
            num_shells=config["num_radial"],
            mamba_angle_mode="official",
            mamba_mimo_rank=1,
            shell_coupling_mode="legacy",
        )
        payload = {
            "format_version": 1,
            "architecture": "mtace_v2",
            "architecture_version": 5,
            "model_config": config,
            "training_config": {},
            "model_state_dict": old_model.state_dict(),
            "optimizer_state_dict": None,
            "epoch": 0,
            "atomic_energies": {1: -1.0},
            "atomic_numbers": [1],
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v5.pt"
            torch.save(payload, checkpoint)
            restored, _ = restore_model(checkpoint, device="cpu")
        direction = restored.layers[0].mixer.forward_direction
        self.assertEqual(direction.angle_mode, "official")

    def test_v6_attention_checkpoint_preserves_historical_internal_dropout(self):
        config = {
            **self._config(),
            "mixer_type": "attention",
            "attention_heads": 2,
            "dropout": 0.25,
        }
        old_model = MambaACE(
            **config,
            attention_dropout=0.25,
            shell_coupling_mode="legacy",
        )
        payload = {
            "format_version": 1,
            "architecture": "mtace_v2",
            "architecture_version": 6,
            "model_config": config,
            "training_config": {},
            "model_state_dict": old_model.state_dict(),
            "optimizer_state_dict": None,
            "epoch": 0,
            "atomic_energies": {1: -1.0},
            "atomic_numbers": [1],
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v6_attention.pt"
            torch.save(payload, checkpoint)
            calculator = MambaACECalculator(checkpoint, device="cpu")
        self.assertEqual(
            calculator.model.layers[0].mixer.attention_dropout.p, 0.25
        )

    def test_v7_checkpoint_preserves_legacy_shell_coupling_exactly(self):
        config = {
            **self._config(),
            "tokenizer_type": "physical_shells",
            "num_shells": 7,
            "mamba_mimo_rank": 1,
        }
        # Architecture v9 changed four defaults.  A v7 reference must therefore be
        # built with the historical settings explicitly; the migration is required
        # to reproduce exactly this model from the stored configuration alone.
        historical = {
            "shell_coupling_mode": "legacy",
            "shell_boundary_mode": "renormalize",
            "avg_num_neighbors": 1.0,
            "continuum_mode": False,
            "mamba_rotary_layout": "pairs",
            "mamba_scan_mode": "parallel",
            "ffn_type": "swiglu",
            "invariant_pair_channels": 0,
        }
        old_model = MambaACE(**config, **historical).eval()
        payload = {
            "format_version": 1,
            "architecture": "mtace_v2",
            "architecture_version": 7,
            "model_config": config,
            "training_config": {},
            "model_state_dict": old_model.state_dict(),
            "optimizer_state_dict": None,
            "epoch": 0,
            "atomic_energies": {1: -1.0},
            "atomic_numbers": [1],
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v7.pt"
            torch.save(payload, checkpoint)
            restored, _ = restore_model(checkpoint, device="cpu")

        self.assertEqual(restored.shell_coupling_mode, "legacy")
        self.assertEqual(restored.ace.shell_coupling_mode, "legacy")
        self.assertEqual(restored.layers[0].token_reduction, "sqrt_length")
        z = torch.tensor([1, 1], dtype=torch.long)
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.2, 0.1]])
        edge_index = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
        edge_shift = torch.zeros(2, 3)
        cell = torch.eye(3) * 5.0
        expected = old_model.atomic_energies(
            z, positions, cell, edge_index, edge_shift
        )
        actual = restored.atomic_energies(z, positions, cell, edge_index, edge_shift)
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_missing_atomic_reference_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "Z=\\[6\\]"):
            baseline_energy(torch.tensor([1, 6]), {1: -1.0, 8: -2.0})

    def test_calculator_validates_runtime_mamba_backend(self):
        with self.assertRaisesRegex(ValueError, "mamba_backend"):
            MambaACECalculator("unused.pt", device="cpu", mamba_backend="fast")
        with self.assertRaisesRegex(CalculatorSetupError, "requires a CUDA device"):
            MambaACECalculator("unused.pt", device="cpu", mamba_backend="cuda")

    def test_float64_checkpoint_and_linear_neighbor_backend_are_preserved(self):
        config = self._config()
        network = MambaACE(**config).double()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            save_checkpoint(checkpoint, network, config, atomic_energies={1: -1.0})
            atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
            atoms.calc = MambaACECalculator(checkpoint, device="cpu")
            atoms.get_potential_energy()
            self.assertEqual(atoms.calc.dtype, torch.float64)
            self.assertEqual(next(atoms.calc.model.parameters()).dtype, torch.float64)
            self.assertIsInstance(atoms.calc._neighbor_list.nl, NewPrimitiveNeighborList)

    def test_species_contract_is_fail_closed_for_legacy_checkpoints(self):
        config = self._config()
        network = MambaACE(**config)
        payload = {
            "format_version": 1,
            "architecture": network.architecture,
            "architecture_version": network.architecture_version,
            "model_config": config,
            "training_config": {},
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": None,
            "epoch": 0,
            "atomic_energies": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            torch.save(payload, checkpoint)
            atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
            atoms.calc = MambaACECalculator(checkpoint, device="cpu")
            with self.assertRaisesRegex(CalculatorSetupError, "no trained-species metadata"):
                atoms.get_potential_energy()
            atoms.calc = MambaACECalculator(checkpoint, device="cpu", elements=["H"])
            self.assertTrue(np.isfinite(atoms.get_potential_energy()))

    def test_checkpoint_rejects_species_outside_training_set(self):
        config = self._config()
        network = MambaACE(**config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            save_checkpoint(
                checkpoint,
                network,
                config,
                atomic_energies={},
                atomic_numbers=[1],
            )
            atoms = Atoms("O", positions=[[0.0, 0.0, 0.0]])
            atoms.calc = MambaACECalculator(checkpoint, device="cpu")
            with self.assertRaisesRegex(CalculatorSetupError, "outside the checkpoint"):
                atoms.get_potential_energy()

    def test_periodic_cell_updates_stress_and_ase_drivers(self):
        torch.manual_seed(7)
        config = self._config()
        network = MambaACE(**config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            save_checkpoint(checkpoint, network, config, atomic_energies={1: -1.0})
            atoms = Atoms(
                "H3",
                positions=[[0.15, 0.20, 0.10], [4.80, 0.25, 0.15], [2.1, 1.0, 0.4]],
                cell=[[5.0, 0.0, 0.0], [0.3, 4.8, 0.0], [0.2, 0.1, 4.6]],
                pbc=True,
            )
            atoms.calc = MambaACECalculator(
                checkpoint, device="cpu", neighbor_skin=0.4
            )
            energy = atoms.get_potential_energy()
            atomic = atoms.get_potential_energies()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            self.assertAlmostEqual(energy, float(atomic.sum()), places=5)
            np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=3.0e-5)
            self.assertTrue(np.isfinite(stress).all())

            displacement = 5.0e-4
            displaced_energies = []
            for sign in (-1.0, 1.0):
                displaced = atoms.copy()
                displaced.positions[0, 0] += sign * displacement
                displaced.calc = MambaACECalculator(checkpoint, device="cpu")
                displaced_energies.append(displaced.get_potential_energy())
            numerical_force = -(displaced_energies[1] - displaced_energies[0]) / (
                2.0 * displacement
            )
            self.assertAlmostEqual(forces[0, 0], numerical_force, delta=3.0e-3)

            delta = 2.0e-3
            volume = atoms.get_volume()
            strain_components = [
                (0, 0, 0, 1.0),
                (1, 1, 1, 1.0),
                (2, 2, 2, 1.0),
                (1, 2, 3, 2.0),
                (0, 2, 4, 2.0),
                (0, 1, 5, 2.0),
            ]
            for row, column, voigt, shear_factor in strain_components:
                numerical = []
                for sign in (-1.0, 1.0):
                    strained = atoms.copy()
                    deformation = np.eye(3)
                    deformation[row, column] += sign * delta
                    if row != column:
                        deformation[column, row] += sign * delta
                    strained.set_cell(atoms.cell.array @ deformation, scale_atoms=True)
                    strained.calc = MambaACECalculator(checkpoint, device="cpu")
                    numerical.append(strained.get_potential_energy())
                finite_difference = (numerical[1] - numerical[0]) / (
                    2.0 * delta * volume * shear_factor
                )
                self.assertAlmostEqual(stress[voigt], finite_difference, delta=3.0e-3)

            original = energy
            atoms.set_cell(atoms.cell.array * 1.03, scale_atoms=True)
            changed = atoms.get_potential_energy()
            self.assertFalse(np.isclose(original, changed, rtol=0.0, atol=1.0e-7))

            with FIRE(atoms, logfile=None) as optimizer:
                optimizer.run(steps=1)
            atoms.set_velocities(np.zeros((len(atoms), 3)))
            with VelocityVerlet(atoms, timestep=0.1 * fs, logfile=None) as dynamics:
                dynamics.run(1)
            self.assertTrue(np.isfinite(atoms.get_positions()).all())


if __name__ == "__main__":
    unittest.main()
