"""``split_seed`` fixes the validation set independently of the training seed."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

ROOT = Path(__file__).resolve().parents[1]


def _frames():
    frames = []
    for index in range(12):
        distance = 0.7 + 0.04 * index
        atoms = Atoms("H2", positions=[[0, 0, 0], [distance, 0, 0]], cell=[5, 5, 5], pbc=True)
        displacement = distance - 0.9
        atoms.calc = SinglePointCalculator(
            atoms, energy=0.5 * displacement**2,
            forces=np.array([[displacement, 0, 0], [-displacement, 0, 0]]),
        )
        frames.append(atoms)
    return frames


def _fingerprint(directory: Path, trajectory: Path, **overrides) -> str:
    config = {
        "train_file": str(trajectory), "validation_fraction": 0.25,
        "split_mode": "blocked", "split_block_size": 2,
        "architecture": "mtace_v2", "r_max": 2.0, "l_max": 1, "num_radial": 2,
        "hidden_dim": 4, "num_layers": 1, "correlation_order": 2,
        "correlation_channels": 2, "radial_mlp_hidden": 4, "radial_mlp_layers": 1,
        "mixer_type": "identity", "mamba_dim": 4, "readout_hidden": 4,
        "ffn_hidden": 8, "epochs": 1, "batch_size": 4, "device": "cpu",
        "dtype": "float32", "optimizer": "adamw", "learning_rate": 1e-3,
        "solve_atomic_energies": False, "atomic_energies": {"H": 0.0},
        "energy_weight": 1.0, "forces_weight": 1.0, "stress_weight": 0.0,
        "report_stress_metrics": False, "gate_diagnostic_every": 0,
        "model_save_path": str(directory / "model.pt"),
        "last_model_save_path": str(directory / "model_last.pt"),
    }
    config.update(overrides)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    completed = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--config", str(path)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert completed.returncode == 0, completed.stdout
    return re.search(r"validation_fingerprint=([0-9a-f]+)", completed.stdout).group(1)


def test_split_seed_holds_the_validation_set_fixed_across_seeds():
    with TemporaryDirectory() as name:
        directory = Path(name)
        trajectory = directory / "h2.extxyz"
        write(trajectory, _frames())
        fixed = {_fingerprint(directory, trajectory, seed=s, split_seed=7) for s in (1, 2, 3)}
        assert len(fixed) == 1
        # Control: without split_seed the split follows the training seed and
        # the replicates are scored on different frames.
        coupled = {_fingerprint(directory, trajectory, seed=s) for s in (1, 2, 3)}
        assert len(coupled) > 1
