"""An exact resume from a ``best`` checkpoint under EMA.

The best checkpoint holds the averaged weights (what deployment wants).  Before
this fix a resume from it loaded that average as the raw iterate and continued
a different trajectory; the ``last`` checkpoint, which holds the raw weights,
resumed exactly.  Both must now give the uninterrupted run, bit for bit.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

ROOT = Path(__file__).resolve().parents[1]


def _frames():
    frames = []
    for index, distance in enumerate((0.75, 0.85, 0.95, 1.05, 0.8, 1.0)):
        atoms = Atoms("H2", positions=[[0, 0, 0], [distance, 0, 0]], cell=[5, 5, 5], pbc=True)
        d = distance - 0.9
        atoms.calc = SinglePointCalculator(
            atoms, energy=0.5 * d * d + 0.01 * index,
            forces=np.array([[d, 0, 0], [-d, 0, 0]]), stress=np.zeros(6),
        )
        frames.append(atoms)
    return frames


COMMON = {
    "validation_fraction": 0.34, "split_mode": "random", "seed": 5,
    "save_last_checkpoint": True, "architecture": "mtace_v2", "r_max": 2.0,
    "l_max": 1, "num_radial": 2, "hidden_dim": 4, "num_layers": 1,
    "correlation_order": 2, "correlation_channels": 2, "radial_mlp_hidden": 4,
    "radial_mlp_layers": 1, "mixer_type": "mamba", "mamba_variant": "mamba3",
    "mamba_dim": 4, "mamba_d_state": 4, "mamba_headdim": 4, "mamba_backend": "torch",
    "ffn_hidden": 8, "readout_hidden": 4, "epochs": 3, "batch_size": 2,
    "device": "cpu", "dtype": "float32", "optimizer": "muon", "learning_rate": 1e-3,
    "weight_decay": 1e-5, "ema_decay": 0.9, "early_stopping_patience": 0,
    "solve_atomic_energies": False, "atomic_energies": {"H": 0.0},
    "energy_weight": 1.0, "forces_weight": 1.0, "stress_weight": 0.0,
    "report_stress_metrics": False, "gate_diagnostic_every": 1,
}


def _run(directory: Path, name: str, resume: Path | None = None, **overrides):
    config = dict(COMMON, **overrides)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    command = [sys.executable, str(ROOT / "train.py"), "--config", str(path)]
    if resume is not None:
        command += ["--resume", str(resume)]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    assert completed.returncode == 0, completed.stdout
    return completed.stdout


def _weights(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]


def test_resume_from_best_equals_resume_from_last_equals_uninterrupted():
    with TemporaryDirectory() as name:
        directory = Path(name)
        trajectory = directory / "h2.extxyz"
        write(trajectory, _frames())
        full = dict(train_file=str(trajectory), model_save_path=str(directory / "full.pt"),
                    last_model_save_path=str(directory / "full_last.pt"))
        _run(directory, "full", **full)
        reference = _weights(directory / "full_last.pt")

        # Stop after epoch 1: best and last are then both epoch-1 checkpoints,
        # one holding the EMA average, the other the raw iterate.
        part = dict(train_file=str(trajectory), model_save_path=str(directory / "part.pt"),
                    last_model_save_path=str(directory / "part_last.pt"), stop_after_epoch=1)
        _run(directory, "part", **part)
        best = torch.load(directory / "part.pt", map_location="cpu", weights_only=False)
        assert best["checkpoint_role"] == "best" and best["raw_model_state_dict"] is not None
        # The two are genuinely different weights.
        last = _weights(directory / "part_last.pt")
        assert any(not torch.equal(best["model_state_dict"][k], last[k]) for k in last)

        for label, source in (("best", directory / "part.pt"), ("last", directory / "part_last.pt")):
            out = dict(train_file=str(trajectory), model_save_path=str(directory / f"{label}_r.pt"),
                       last_model_save_path=str(directory / f"{label}_r_last.pt"))
            _run(directory, f"resume_{label}", resume=source, **out)
            resumed = _weights(directory / f"{label}_r_last.pt")
            for key in reference:
                assert torch.equal(resumed[key], reference[key]), (label, key)


def test_an_old_best_checkpoint_without_the_raw_iterate_is_refused():
    with TemporaryDirectory() as name:
        directory = Path(name)
        trajectory = directory / "h2.extxyz"
        write(trajectory, _frames())
        part = dict(train_file=str(trajectory), model_save_path=str(directory / "part.pt"),
                    last_model_save_path=str(directory / "part_last.pt"), stop_after_epoch=1)
        _run(directory, "part", **part)
        payload = torch.load(directory / "part.pt", map_location="cpu", weights_only=False)
        payload["raw_model_state_dict"] = None
        torch.save(payload, directory / "old_best.pt")
        config = dict(COMMON, train_file=str(trajectory),
                      model_save_path=str(directory / "r.pt"),
                      last_model_save_path=str(directory / "r_last.pt"))
        path = directory / "r.yaml"
        path.write_text(yaml.safe_dump(config))
        completed = subprocess.run(
            [sys.executable, str(ROOT / "train.py"), "--config", str(path),
             "--resume", str(directory / "old_best.pt")],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        assert completed.returncode != 0
        assert "cannot be resumed exactly" in completed.stdout
