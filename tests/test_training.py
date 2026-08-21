import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mtace.checkpoint import load_checkpoint, save_checkpoint
from mtace.data import (
    AtomisticDataset,
    build_neighbor_tensors,
    stress_matrix,
    stress_mse,
    stress_target,
    stress_to_mandel,
    solve_atomic_energies,
)
from mtace.model import MambaACE
from train import (
    EV_A3_TO_GPA,
    adamw_parameter_groups,
    capture_rng_state,
    conflicting_model_settings,
    finalize_metrics,
    last_checkpoint_path,
    new_metric_accumulator,
    restore_rng_state,
    update_metric_accumulator,
    validate_training_species,
)


class OptimizerConfigurationTests(unittest.TestCase):
    def test_ssm_dynamics_are_excluded_from_weight_decay(self):
        model = MambaACE(
            r_max=3.5,
            l_max=1,
            num_radial=3,
            hidden_dim=6,
            num_layers=1,
            correlation_order=3,
            correlation_channels=3,
            mamba_dim=8,
            mamba_d_state=4,
            mamba_backend="torch",
        )
        groups = adamw_parameter_groups(model, 1.0e-4)
        no_decay = {id(parameter) for group in groups if group["weight_decay"] == 0.0 for parameter in group["params"]}
        mixer = model.layers[0].mixer
        self.assertIn(id(mixer.forward_direction.dt_bias), no_decay)
        self.assertIn(id(mixer.forward_direction.D), no_decay)
        self.assertIn(id(mixer.backward_direction.dt_bias), no_decay)
        self.assertIn(id(mixer.backward_direction.D), no_decay)
        self.assertIn(id(model.readout.network[0].bias), no_decay)
        self.assertIn(id(model.layers[0].scalar_norm.weight), no_decay)


class CheckpointSavingTests(unittest.TestCase):
    def test_last_checkpoint_path_preserves_extension(self):
        self.assertEqual(last_checkpoint_path("model.pt"), Path("model_last.pt"))
        self.assertEqual(last_checkpoint_path("model"), Path("model_last"))

    def test_migrated_defaults_do_not_block_resume_but_conflicts_do(self):
        checkpoint = {
            "architecture": "mtace_v2",
            "architecture_version": 7,
            "model_config": {
                "num_radial": 4,
                "tokenizer_type": "physical_shells",
                "num_shells": 16,
                "mamba_variant": "mamba3",
                "mamba_angle_mode": "official",
            },
        }
        original_request = dict(checkpoint["model_config"])
        self.assertEqual(
            conflicting_model_settings(checkpoint, original_request), []
        )
        self.assertEqual(
            conflicting_model_settings(
                checkpoint,
                {**original_request, "shell_coupling_mode": "conservative"},
            ),
            ["shell_coupling_mode"],
        )

    def test_checkpoint_records_training_state_and_is_atomically_replaced(self):
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model_last.pt"
            save_checkpoint(
                path,
                model,
                {},
                optimizer=optimizer,
                epoch=3,
                atomic_energies={1: -0.5},
                atomic_numbers=[1],
                scheduler=scheduler,
                metrics={"loss": 0.25, "force_rmse_ev_a": 0.1},
                training_metrics={"loss": 0.3, "stress_rmse_gpa": 0.4},
                best_validation_loss=0.2,
                checkpoint_role="last",
            )
            checkpoint = load_checkpoint(path)
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["checkpoint_role"], "last")
            self.assertEqual(checkpoint["training_metrics"]["loss"], 0.3)
            self.assertEqual(checkpoint["validation_metrics"]["loss"], 0.25)
            self.assertEqual(checkpoint["best_validation_loss"], 0.2)
            self.assertEqual(checkpoint["scheduler_state_dict"], scheduler.state_dict())
            self.assertFalse((path.parent / f".{path.name}.tmp").exists())

    def test_all_training_rng_states_are_restored(self):
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        generator = torch.Generator().manual_seed(14)
        state = capture_rng_state(generator)
        expected = (
            random.random(),
            np.random.random(),
            torch.rand(()),
            torch.rand((), generator=generator),
        )
        random.seed(101)
        np.random.seed(102)
        torch.manual_seed(103)
        generator.manual_seed(104)
        restore_rng_state(state, generator)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        torch.testing.assert_close(torch.rand(()), expected[2], atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            torch.rand((), generator=generator), expected[3], atol=0.0, rtol=0.0
        )


class StressAndMetricTests(unittest.TestCase):
    def test_mandel_stress_loss_is_the_tensor_frobenius_norm(self):
        error = torch.tensor(
            [[1.0, 2.0, -3.0], [2.0, 4.0, 5.0], [-3.0, 5.0, -2.0]],
            dtype=torch.float64,
        )
        mandel = stress_to_mandel(error)
        torch.testing.assert_close(mandel.square().sum(), error.square().sum())
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        rotated = rotation @ error @ rotation.T
        zero = torch.zeros_like(error)
        torch.testing.assert_close(stress_mse(error, zero), stress_mse(rotated, zero))

    def test_virial_to_ase_stress_has_the_correct_sign_and_volume(self):
        atoms = Atoms("H", cell=[2.0, 3.0, 4.0], pbc=True)
        virial = np.array(
            [[2.0, 0.5, 0.0], [0.5, -3.0, 0.25], [0.0, 0.25, 4.0]]
        )
        atoms.info["virial"] = virial
        stress, has_stress = stress_target(atoms)
        self.assertTrue(has_stress)
        np.testing.assert_allclose(stress, -virial / 24.0)

        invalid = Atoms("H")
        invalid.info["stress"] = np.zeros(6)
        with self.assertRaisesRegex(ValueError, "full-rank cell"):
            stress_target(invalid)

    def test_asymmetric_and_inconsistent_tensor_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "symmetric"):
            stress_matrix([[1.0, 0.2, 0.0], [0.1, 2.0, 0.0], [0.0, 0.0, 3.0]])
        atoms = Atoms("H", cell=[2.0, 2.0, 2.0], pbc=True)
        atoms.info["stress"] = np.eye(3) * 0.1
        atoms.info["virial"] = -np.eye(3) * 1.6
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            stress_target(atoms)

    def test_training_neighbors_are_filtered_in_model_precision(self):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.99999999, 0.0, 0.0]])
        edge_index, edge_shift = build_neighbor_tensors(
            atoms, 1.0, dtype=torch.float32
        )
        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(edge_shift.shape, (0, 3))

    def test_dataset_rejects_nonfinite_raw_force_labels(self):
        atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=0.0,
            forces=np.array([[np.nan, 0.0, 0.0]]),
        )
        with self.assertRaisesRegex(ValueError, "forces"):
            AtomisticDataset([atoms], r_max=2.0)

    def test_metric_reductions_use_all_force_and_mandel_stress_components(self):
        totals = new_metric_accumulator(torch.device("cpu"))
        energy_error = torch.tensor(0.002)
        force_error = torch.tensor([[1.0, -1.0, 2.0]])
        stress_error = torch.tensor(
            [[0.001, 0.004, 0.003], [0.004, 0.002, 0.005], [0.003, 0.005, 0.006]]
        )
        update_metric_accumulator(
            totals,
            torch.tensor(0.5),
            energy_error,
            force_error,
            stress_error,
        )
        metrics = finalize_metrics(totals)
        expected_stress = float(stress_to_mandel(stress_error).square().mean().sqrt())
        self.assertAlmostEqual(metrics["loss"], 0.5)
        self.assertAlmostEqual(metrics["energy_rmse_mev_atom"], 2.0, places=6)
        self.assertAlmostEqual(metrics["force_rmse_ev_a"], np.sqrt(2.0), places=6)
        self.assertAlmostEqual(metrics["stress_rmse_ev_a3"], expected_stress, places=8)
        self.assertAlmostEqual(
            metrics["stress_rmse_gpa"], EV_A3_TO_GPA * expected_stress, places=6
        )
        self.assertEqual(metrics["stress_structures"], 1)


class DatasetContractTests(unittest.TestCase):
    @staticmethod
    def labelled_atoms(symbols: str, energy: float = 0.0) -> Atoms:
        atoms = Atoms(symbols, positions=np.zeros((len(symbols), 3)))
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy,
            forces=np.zeros((len(atoms), 3)),
        )
        return atoms

    def test_validation_only_species_are_rejected(self):
        training = [self.labelled_atoms("H")]
        validation = [self.labelled_atoms("O")]
        with self.assertRaisesRegex(ValueError, "absent from training"):
            validate_training_species(training, validation)

    def test_rank_deficient_atomic_reference_fit_is_rejected(self):
        frames = [
            self.labelled_atoms("HHO", energy=-1.0),
            self.labelled_atoms("HHO", energy=-1.1),
        ]
        with self.assertRaisesRegex(ValueError, "not identifiable"):
            solve_atomic_energies(frames)

    def test_full_rank_atomic_reference_fit_is_supported(self):
        frames = [
            self.labelled_atoms("H", energy=-0.5),
            self.labelled_atoms("O", energy=-2.0),
            self.labelled_atoms("HHO", energy=-3.0),
        ]
        references = solve_atomic_energies(frames)
        self.assertAlmostEqual(references[1], -0.5)
        self.assertAlmostEqual(references[8], -2.0)


if __name__ == "__main__":
    unittest.main()
