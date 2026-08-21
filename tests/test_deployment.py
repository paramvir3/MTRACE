import tempfile
import unittest
from pathlib import Path

import torch

from mtace.checkpoint import save_checkpoint
from mtace.deployment import export_lammps_model
from mtace.model import MambaACE


class DeploymentTests(unittest.TestCase):
    def test_export_has_dynamic_shapes_metadata_and_conservative_gradient(self):
        torch.manual_seed(13)
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
            checkpoint = Path(directory) / "checkpoint.pt"
            exported = Path(directory) / "model.mtace.pt"
            save_checkpoint(
                checkpoint,
                network,
                config,
                atomic_energies={1: -1.25, 8: -4.5},
            )
            export_lammps_model(checkpoint, exported, ["H", "O"])
            extra = {
                key: ""
                for key in (
                    "mamba_ace_format_version",
                    "r_max",
                    "elements",
                    "atomic_numbers",
                    "dtype",
                    "strictly_local",
                )
            }
            model = torch.jit.load(str(exported), _extra_files=extra)
            self.assertEqual(extra["mamba_ace_format_version"], b"1")
            self.assertEqual(extra["elements"], b"H O")
            self.assertEqual(extra["atomic_numbers"], b"1 8")
            self.assertEqual(extra["strictly_local"], b"1")

            z = torch.tensor([1, 8, 1, 8, 1], dtype=torch.long)
            positions = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.8, 0.1, 0.0],
                    [0.2, 0.9, 0.1],
                    [0.1, 0.2, 1.0],
                    [0.7, 0.6, 0.5],
                ],
                requires_grad=True,
            )
            index = torch.arange(len(z))
            receiver = index.repeat_interleave(len(z))
            sender = index.repeat(len(z))
            keep = sender != receiver
            edges = torch.stack((sender[keep], receiver[keep]))
            atomic = model(z, positions, edges)
            forces = -torch.autograd.grad(atomic.sum(), positions)[0]
            self.assertEqual(tuple(atomic.shape), (5,))
            self.assertTrue(torch.isfinite(forces).all())
            torch.testing.assert_close(
                forces.sum(dim=0), torch.zeros(3), rtol=0.0, atol=3.0e-5
            )

    def test_export_rejects_missing_reference(self):
        config = {
            "r_max": 3.0,
            "l_max": 0,
            "num_radial": 2,
            "hidden_dim": 4,
            "num_layers": 1,
            "correlation_order": 2,
            "correlation_channels": 2,
            "mamba_dim": 4,
            "mamba_d_state": 4,
            "mamba_backend": "torch",
        }
        network = MambaACE(**config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            save_checkpoint(checkpoint, network, config, atomic_energies={1: -1.0})
            with self.assertRaisesRegex(ValueError, "Z=\\[8\\]"):
                export_lammps_model(
                    checkpoint, Path(directory) / "model.mtace.pt", ["H", "O"]
                )

    def test_export_rejects_species_absent_from_training_metadata(self):
        config = {
            "r_max": 3.0,
            "l_max": 0,
            "num_radial": 2,
            "hidden_dim": 4,
            "num_layers": 1,
            "correlation_order": 2,
            "correlation_channels": 2,
            "mamba_dim": 4,
            "mamba_d_state": 4,
            "mamba_backend": "torch",
        }
        network = MambaACE(**config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                checkpoint,
                network,
                config,
                atomic_energies={},
                atomic_numbers=[1],
            )
            with self.assertRaisesRegex(ValueError, "not present during training"):
                export_lammps_model(
                    checkpoint, Path(directory) / "model.mtace.pt", ["O"]
                )


if __name__ == "__main__":
    unittest.main()
