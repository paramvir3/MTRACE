import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from mtace.checkpoint import save_checkpoint
from mtace.model import MambaACEV2


def assert_nested_equal(test, first, second, path="state"):
    if torch.is_tensor(first):
        test.assertTrue(torch.equal(first, second), path)
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys(), path)
        for key in first:
            assert_nested_equal(test, first[key], second[key], f"{path}.{key}")
    elif isinstance(first, (list, tuple)):
        test.assertEqual(len(first), len(second), path)
        for index, (left, right) in enumerate(zip(first, second)):
            assert_nested_equal(test, left, right, f"{path}[{index}]")
    else:
        test.assertEqual(first, second, path)


class ExactResumeTests(unittest.TestCase):
    def test_interrupted_muon_run_is_bitwise_equal_to_uninterrupted_run(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            frames = []
            for index, distance in enumerate((0.75, 0.85, 0.95, 1.05)):
                atoms = Atoms(
                    "H2",
                    positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
                    cell=[5.0, 5.0, 5.0],
                    pbc=True,
                )
                displacement = distance - 0.9
                force = -displacement
                atoms.calc = SinglePointCalculator(
                    atoms,
                    energy=0.5 * displacement**2 + 0.01 * index,
                    forces=np.array([[-force, 0.0, 0.0], [force, 0.0, 0.0]]),
                    stress=np.zeros(6),
                )
                frames.append(atoms)
            trajectory = directory / "tiny.extxyz"
            write(trajectory, frames)

            common = {
                "train_file": str(trajectory),
                "validation_fraction": 0.25,
                "split_mode": "random",
                "seed": 91,
                "save_last_checkpoint": True,
                "architecture": "mtace_v2",
                "r_max": 2.0,
                "l_max": 1,
                "num_radial": 2,
                "hidden_dim": 4,
                "num_layers": 1,
                "correlation_order": 2,
                "correlation_channels": 2,
                "radial_mlp_hidden": 4,
                "radial_mlp_layers": 1,
                "mamba_variant": "mamba3",
                "mamba_dim": 4,
                "mamba_d_state": 4,
                "mamba_expand": 2,
                "mamba_headdim": 4,
                "mamba_angle_mode": "official",
                "mamba_backend": "torch",
                "ffn_hidden": 8,
                "dropout": 0.05,
                "readout_hidden": 4,
                "epochs": 2,
                "batch_size": 2,
                "device": "cpu",
                "dtype": "float32",
                "precompute_neighbors": True,
                "report_stress_metrics": True,
                "clip_grad_norm": 10.0,
                "early_stopping_patience": 0,
                "optimizer": "muon",
                "learning_rate": 1.0e-3,
                "minimum_learning_rate": 1.0e-5,
                "weight_decay": 1.0e-5,
                "muon_parameter_mode": "hidden",
                "solve_atomic_energies": False,
                "atomic_energies": {"H": 0.0},
                "energy_weight": 1.0,
                "forces_weight": 1.0,
                "stress_weight": 1.0,
            }

            def run(config, name, resume=None):
                path = directory / f"{name}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=True))
                command = [sys.executable, str(root / "train.py"), "--config", str(path)]
                if resume is not None:
                    command.extend(("--resume", str(resume)))
                completed = subprocess.run(
                    command,
                    cwd=root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout)

            uninterrupted = dict(
                common,
                model_save_path=str(directory / "full.pt"),
                last_model_save_path=str(directory / "full_last.pt"),
                stop_after_epoch=2,
            )
            staged = dict(
                common,
                model_save_path=str(directory / "staged.pt"),
                last_model_save_path=str(directory / "staged_last.pt"),
                stop_after_epoch=1,
            )
            run(uninterrupted, "full")
            run(staged, "staged_first")
            staged["stop_after_epoch"] = 2
            # The minimum-loss model.pt contains the complete state needed for
            # exact continuation, not only model weights.
            run(staged, "staged_second", directory / "staged.pt")

            full = torch.load(directory / "full_last.pt", weights_only=True)
            resumed = torch.load(directory / "staged_last.pt", weights_only=True)
            self.assertEqual(full["epoch"], resumed["epoch"])
            assert_nested_equal(self, full["model_state_dict"], resumed["model_state_dict"])
            assert_nested_equal(
                self, full["optimizer_state_dict"], resumed["optimizer_state_dict"]
            )
            assert_nested_equal(
                self, full["scheduler_state_dict"], resumed["scheduler_state_dict"]
            )
            self.assertEqual(full["training_metrics"], resumed["training_metrics"])
            self.assertEqual(full["validation_metrics"], resumed["validation_metrics"])

    def test_warm_restart_loads_legacy_model_weights_and_resets_training_state(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            frames = []
            for index, distance in enumerate((0.75, 0.85, 0.95, 1.05)):
                atoms = Atoms(
                    "H2",
                    positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
                    cell=[5.0, 5.0, 5.0],
                    pbc=True,
                )
                displacement = distance - 0.9
                force = -displacement
                atoms.calc = SinglePointCalculator(
                    atoms,
                    energy=0.5 * displacement**2 + 0.01 * index,
                    forces=np.array([[-force, 0.0, 0.0], [force, 0.0, 0.0]]),
                    stress=np.zeros(6),
                )
                frames.append(atoms)
            trajectory = directory / "tiny.extxyz"
            write(trajectory, frames)

            model_config = {
                "r_max": 2.0,
                "l_max": 1,
                "num_radial": 2,
                "hidden_dim": 4,
                "num_layers": 1,
                "correlation_order": 2,
                "correlation_channels": 2,
                "radial_mlp_hidden": 4,
                "radial_mlp_layers": 1,
                "mamba_variant": "mamba3",
                "mamba_dim": 4,
                "mamba_d_state": 4,
                "mamba_expand": 2,
                "mamba_headdim": 4,
                "mamba_angle_mode": "official",
                "mamba_backend": "torch",
                "ffn_hidden": 8,
                "dropout": 0.0,
                "readout_hidden": 4,
            }
            torch.manual_seed(123)
            parent_model = MambaACEV2(**model_config)
            parent = directory / "legacy_model.pt"
            save_checkpoint(
                parent,
                parent_model,
                model_config,
                epoch=7,
                atomic_energies={1: -0.25},
                atomic_numbers=[1],
            )
            legacy_payload = torch.load(parent, weights_only=True)
            legacy_payload.pop("training_objective_version")
            legacy_payload["rng_state"] = {}
            legacy_payload["run_signature"] = {}
            torch.save(legacy_payload, parent)

            output = directory / "restarted.pt"
            last_output = directory / "restarted_last.pt"
            config = {
                "train_file": str(trajectory),
                "validation_fraction": 0.25,
                "split_mode": "random",
                "seed": 19,
                "model_save_path": str(output),
                "last_model_save_path": str(last_output),
                "save_last_checkpoint": True,
                "architecture": "mtace_v2",
                "epochs": 2,
                "stop_after_epoch": 1,
                "batch_size": 2,
                "device": "cpu",
                "dtype": "float32",
                "optimizer": "adamw",
                "learning_rate": 0.0,
                "minimum_learning_rate": 0.0,
                "weight_decay": 0.0,
                "early_stopping_patience": 0,
                "solve_atomic_energies": True,
                "energy_weight": 1.0,
                "forces_weight": 1.0,
                "stress_weight": 1.0,
            }
            config_path = directory / "restart.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=True))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "train.py"),
                    "--config",
                    str(config_path),
                    "--restart",
                    str(parent),
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("restarted_from=", completed.stdout)
            self.assertIn("parent_epoch=7 next_epoch=1", completed.stdout)

            restarted = torch.load(last_output, weights_only=True)
            self.assertEqual(restarted["epoch"], 1)
            self.assertEqual(restarted["atomic_energies"], {1: -0.25})
            self.assertEqual(restarted["atomic_numbers"], [1])
            self.assertEqual(restarted["run_signature"]["restart_parent"]["epoch"], 7)
            assert_nested_equal(
                self,
                legacy_payload["model_state_dict"],
                restarted["model_state_dict"],
            )

            config["stop_after_epoch"] = 2
            config_path.write_text(yaml.safe_dump(config, sort_keys=True))
            resumed = subprocess.run(
                [
                    sys.executable,
                    str(root / "train.py"),
                    "--config",
                    str(config_path),
                    "--resume",
                    str(output),
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            continued = torch.load(last_output, weights_only=True)
            self.assertEqual(continued["epoch"], 2)
            self.assertEqual(
                continued["run_signature"]["restart_parent"],
                restarted["run_signature"]["restart_parent"],
            )
            assert_nested_equal(
                self,
                legacy_payload["model_state_dict"],
                continued["model_state_dict"],
            )


if __name__ == "__main__":
    unittest.main()
