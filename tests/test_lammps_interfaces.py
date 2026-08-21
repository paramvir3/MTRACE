"""Opt-in end-to-end tests for a patched LAMMPS executable."""

import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from mtace.calculator import MambaACECalculator
from mtace.checkpoint import save_checkpoint
from mtace.deployment import export_lammps_model
from mtace.model import MambaACE


LAMMPS = os.environ.get("MAMBA_ACE_LAMMPS")
EV_A3_TO_BAR = 1_602_176.6208


@unittest.skipUnless(LAMMPS, "set MAMBA_ACE_LAMMPS to run LAMMPS integration tests")
class LAMMPSInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        torch.manual_seed(2026)
        self.config = {
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
        network = MambaACE(**self.config)
        self.checkpoint = self.root / "checkpoint.pt"
        self.exported = self.root / "model.mtace.pt"
        save_checkpoint(
            self.checkpoint,
            network,
            self.config,
            atomic_energies={1: -1.25, 8: -4.5},
        )
        export_lammps_model(self.checkpoint, self.exported, ["H", "O"])
        self.positions = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.2, 1.0],
                [3.8, 1.0, 1.1],
                [4.2, 1.1, 1.0],
                [6.0, 1.0, 1.2],
                [7.5, 1.1, 1.0],
            ]
        )
        self.cell = np.array(
            [[8.0, 0.0, 0.0], [0.5, 8.0, 0.0], [0.2, -0.3, 8.0]]
        )
        self.atoms = Atoms(
            numbers=[1, 8, 1, 8, 1, 8],
            positions=self.positions,
            cell=self.cell,
            pbc=True,
        )
        self.atoms.calc = MambaACECalculator(self.checkpoint, device="cpu")

    def tearDown(self):
        self.temporary.cleanup()

    def _input(self, style, name, processors=None, model_device="cpu"):
        lines = [
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            "region box prism 0 8 0 8 0 8 0.5 0.2 -0.3 units box",
        ]
        if processors is not None:
            lines.append(f"processors {processors}")
        lines.append("create_box 2 box")
        for number, position in zip(self.atoms.numbers, self.positions):
            atom_type = 1 if number == 1 else 2
            lines.append(
                "create_atoms {} single {:.16g} {:.16g} {:.16g} units box".format(
                    atom_type, *position
                )
            )
        lines.extend(
            [
                "mass 1 1.008",
                "mass 2 15.999",
                "newton on",
                "neighbor 0.5 bin",
                "neigh_modify every 1 delay 0 check yes",
                f"pair_style {style} device {model_device}",
                f"pair_coeff * * {self.exported} H O",
                "compute atomic_energy all pe/atom",
                "thermo 1",
                "thermo_style custom step atoms pe pxx pyy pzz pxy pxz pyz",
                "thermo_modify format float %.16g",
                "run 0",
                f"dump result all custom 1 {name}.dump id type x y z fx fy fz c_atomic_energy",
                "dump_modify result sort id format float %.16g",
                "run 0",
            ]
        )
        path = self.root / f"{name}.in"
        path.write_text("\n".join(lines) + "\n")
        return path

    def _run(self, command, input_path, name):
        completed = subprocess.run(
            command + ["-in", str(input_path), "-log", str(self.root / f"{name}.log")],
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        return self._read_result(name)

    def _read_result(self, name):
        log = (self.root / f"{name}.log").read_text().splitlines()
        thermo = None
        for index, line in enumerate(log):
            if "PotEng" in line and "Pxx" in line:
                thermo = np.fromstring(log[index + 1], sep=" ")
        self.assertIsNotNone(thermo)
        dump = (self.root / f"{name}.dump").read_text().splitlines()
        header = "ITEM: ATOMS id type x y z fx fy fz c_atomic_energy"
        start = len(dump) - 1 - dump[::-1].index(header)
        rows = np.array(
            [[float(value) for value in line.split()] for line in dump[start + 1 : start + 7]]
        )
        return thermo, rows

    def _assert_ase_parity(self, thermo, rows):
        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        stress = self.atoms.get_stress()
        pressure = -EV_A3_TO_BAR * np.array(
            [stress[0], stress[1], stress[2], stress[5], stress[4], stress[3]]
        )
        np.testing.assert_allclose(thermo[2], energy, rtol=0.0, atol=3.0e-5)
        np.testing.assert_allclose(rows[:, 5:8], forces, rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(thermo[3:9], pressure, rtol=2.0e-5, atol=0.05)
        np.testing.assert_allclose(rows[:, 8].sum(), energy, rtol=0.0, atol=3.0e-5)

    def test_host_matches_ase(self):
        result = self._run([LAMMPS], self._input("mamba", "host"), "host")
        self._assert_ase_parity(*result)

    def _nve_input(self, name, steps):
        """Constant-energy MD, which no single-point comparison can replace.

        A single-point test confirms that E and F agree with ASE at one geometry.
        It cannot detect a force that is not exactly -dE/dx *as LAMMPS assembles
        it*: a sign error on a ghost contribution, a missing reverse
        communication term, or a neighbour list that misses edges near the cutoff
        all leave the single point intact and show up only as energy drift once
        the integrator starts moving atoms.
        """

        lines = [
            "units metal",
            "atom_style atomic",
            "boundary p p p",
            f"read_data {self.data_path.name}",
            "pair_style mamba device cpu check_finite yes",
            f"pair_coeff * * {self.exported} H O",
            "neighbor 1.0 bin",
            "neigh_modify every 1 delay 0 check yes",
            # 0.2 fs, not 0.5.  The fixture model is untrained, so its potential
            # energy surface is far stiffer than a fitted one and 0.5 fs sits past
            # the velocity-Verlet stability limit: measured spread is 2.8e-3
            # eV/atom at 0.5 fs, 8.8e-6 at 0.25 fs and 1.4e-6 at 0.125 fs.  The
            # 317x drop across the first halving is loss of stability, not a force
            # error; inside the stable regime the scaling is the expected O(dt^2).
            "timestep 0.0002",
            "velocity all create 150.0 4321 mom yes rot yes dist gaussian",
            "fix integrate all nve",
            "thermo 5",
            "thermo_style custom step temp pe ke etotal",
            "thermo_modify format float %.16g",
            f"run {steps}",
        ]
        path = self.root / f"{name}.in"
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_nve_conserves_energy(self):
        """Total energy must be stationary under the velocity-Verlet integrator."""

        from ase.io import write as ase_write

        self.data_path = self.root / "nve.data"
        cell = self.atoms.copy()
        cell.pbc = True
        ase_write(
            str(self.data_path), cell, format="lammps-data", specorder=["H", "O"],
            masses=True, atom_style="atomic", units="metal", force_skew=True,
        )
        steps = 40
        completed = subprocess.run(
            [LAMMPS, "-in", str(self._nve_input("nve", steps)),
             "-log", str(self.root / "nve.log")],
            cwd=self.root, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)

        log = (self.root / "nve.log").read_text().splitlines()
        start = None
        for index, line in enumerate(log):
            if line.split()[:5] == ["Step", "Temp", "PotEng", "KinEng", "TotEng"]:
                start = index + 1
        self.assertIsNotNone(start, "no NVE thermo block in the LAMMPS log")
        totals = []
        for line in log[start:]:
            fields = line.split()
            if len(fields) != 5:
                break
            totals.append(float(fields[4]))
        self.assertGreaterEqual(len(totals), 5, "too few thermo rows to judge drift")

        atoms = len(self.atoms)
        picoseconds = steps * 0.0002
        spread = (max(totals) - min(totals)) / atoms
        drift = abs(totals[-1] - totals[0]) / atoms / picoseconds
        # A conservative force field integrated with velocity Verlet keeps the
        # total energy stationary to the integrator's O(dt^2) bound.  A broken
        # ghost force term produces monotonic drift orders of magnitude larger.
        # Measured 8.8e-6 eV/atom at 0.25 fs on this fixture.  A ghost-force or
        # reverse-communication error gives drift orders of magnitude larger and
        # does not shrink when the timestep does.
        self.assertLess(spread, 1.0e-4, f"total-energy spread {spread:.3e} eV/atom")
        self.assertLess(drift, 1.0e-3, f"drift {drift:.3e} eV/atom/ps")

    @unittest.skipUnless(
        os.environ.get("MAMBA_ACE_TEST_CUDA"),
        "set MAMBA_ACE_TEST_CUDA=1 to test LibTorch CUDA model execution",
    )
    def test_cuda_model_matches_ase(self):
        result = self._run(
            [LAMMPS],
            self._input("mamba", "cuda_model", model_device="cuda"),
            "cuda_model",
        )
        self._assert_ase_parity(*result)

    @unittest.skipUnless(
        os.environ.get("MAMBA_ACE_MPIEXEC"),
        "set MAMBA_ACE_MPIEXEC to test two-rank domain decomposition",
    )
    def test_two_rank_mpi_matches_ase(self):
        command = shlex.split(os.environ["MAMBA_ACE_MPIEXEC"]) + ["-n", "2", LAMMPS]
        result = self._run(
            command, self._input("mamba", "mpi", processors="2 1 1"), "mpi"
        )
        self._assert_ase_parity(*result)

    @unittest.skipUnless(
        os.environ.get("MAMBA_ACE_MPIEXEC"),
        "set MAMBA_ACE_MPIEXEC to test a rank with no owned atoms",
    )
    def test_mpi_rank_without_owned_atoms_matches_ase(self):
        command = shlex.split(os.environ["MAMBA_ACE_MPIEXEC"]) + ["-n", "2", LAMMPS]
        result = self._run(
            command,
            self._input("mamba", "mpi_empty_rank", processors="1 2 1"),
            "mpi_empty_rank",
        )
        self._assert_ase_parity(*result)

    @unittest.skipUnless(
        os.environ.get("MAMBA_ACE_TEST_KOKKOS"),
        "set MAMBA_ACE_TEST_KOKKOS=1 to test mamba/kk",
    )
    def test_kokkos_host_matches_ase(self):
        result = self._run(
            [
                LAMMPS,
                "-k",
                "on",
                "-sf",
                "kk",
                "-pk",
                "kokkos",
                "neigh",
                "half",
            ],
            self._input("mamba/kk", "kokkos"),
            "kokkos",
        )
        self._assert_ase_parity(*result)

    @unittest.skipUnless(
        os.environ.get("MAMBA_ACE_TEST_KOKKOS_CUDA"),
        "set MAMBA_ACE_TEST_KOKKOS_CUDA=1 to test Kokkos-CUDA zero-copy execution",
    )
    def test_kokkos_cuda_matches_ase(self):
        result = self._run(
            [
                LAMMPS,
                "-k",
                "on",
                "g",
                "1",
                "-sf",
                "kk",
                "-pk",
                "kokkos",
                "neigh",
                "half",
            ],
            self._input("mamba/kk", "kokkos_cuda", model_device="cuda"),
            "kokkos_cuda",
        )
        self._assert_ase_parity(*result)

    @unittest.skipUnless(
        os.environ.get("MAMBA_ACE_TEST_KOKKOS")
        and os.environ.get("MAMBA_ACE_MPIEXEC"),
        "set MAMBA_ACE_TEST_KOKKOS=1 and MAMBA_ACE_MPIEXEC to test Kokkos MPI",
    )
    def test_kokkos_two_rank_mpi_matches_ase(self):
        command = shlex.split(os.environ["MAMBA_ACE_MPIEXEC"]) + [
            "-n",
            "2",
            LAMMPS,
            "-k",
            "on",
            "-sf",
            "kk",
            "-pk",
            "kokkos",
            "neigh",
            "half",
        ]
        result = self._run(
            command,
            self._input("mamba/kk", "kokkos_mpi", processors="2 1 1"),
            "kokkos_mpi",
        )
        self._assert_ase_parity(*result)


if __name__ == "__main__":
    unittest.main()
