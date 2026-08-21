# CsPbI3 phase and free-energy tests

Relative phase energies and delta-cubic free energies for a MTACE
checkpoint. Ported from the Transformers-ACE reference tests so the two
potentials can be compared on identical structures, conventions and analysis.

This directory is deliberately **not** under `tests/`, which `pyproject.toml`
registers as the pytest path. These are studies, not unit tests, and they
need a trained checkpoint.

## Prerequisite

```bash
cd training/cspbi3 && python ../../train.py --config config.yaml
```

Use `model.pt` (the EMA-averaged best-validation checkpoint). Do not mix
checkpoints between phases: a relative energy computed from two different models
is not a physical quantity.

## 1. Relative phase energies

Five stoichiometric CsPbI3 polymorphs, 20 atoms (4 formula units) each:

| phase | file | volume (A^3) |
|---|---|---|
| cubic alpha | `cubic_alpha_phase.vasp` | 995.15 |
| tetragonal beta | `tetragonal_beta_phase.vasp` | 981.57 |
| orthorhombic gamma | `orthorhombic_gamma_phase.vasp` | 947.24 |
| edge-sharing delta | `edge_sharing_delta_phase.vasp` | 887.78 |
| face-sharing delta | `face_sharing_delta_phase.vasp` | 1026.54 |

Single point:

```bash
python studies/cspbi3/evaluate_phases.py \
  --model training/cspbi3/model.pt --device cpu --reference minimum
```

Relaxed, which is what the free-energy run consumes:

```bash
python studies/cspbi3/evaluate_phases.py \
  --model training/cspbi3/model.pt --device cpu \
  --relax --relax-cell --fmax 0.01
```

Writes `results/phase_energies.csv` and, when relaxing,
`results/relaxed/<phase>.vasp`. Energies are reported per formula unit; the
constant per-atom reference in the checkpoint cancels exactly in every
difference, so it cannot influence the ordering.

A single structure can be relaxed on its own with `relax_single.py`, which also
supports `FixSymmetry` and spacegroup checking.

## 2. Delta-cubic free energies

`free_energy_ti/` runs a LAMMPS-driven Frenkel-Ladd calculation at 240 atoms and
1 atm, followed by constant-pressure Gibbs-Helmholtz integration. It consumes the
**relaxed** structures from step 1, so run the relaxation first. See
[free_energy_ti/README.md](free_energy_ti/README.md) for the thermodynamics.

```bash
python studies/cspbi3/free_energy_ti/run_ti.py prepare --profile pilot
```

`pilot` only checks deployment, LAMMPS syntax, parsing and analysis; its
trajectories are far too short to be physical. Use `screening` for diagnostics
and `production` for numbers, and note that a publication value still needs the
time-, switching-, replica- and size-convergence tests the README lists.

## Verification status of the two interfaces

Stated precisely, because one of them cannot be fully checked on a machine
without LAMMPS.

**ASE — fully verified.** `evaluate_phases.py` runs end to end on all five
structures. Underneath, `MambaACECalculator` is covered by the repository suite:
energy, forces and stress against finite differences in float64, translation,
O(3), inversion and permutation invariance, and the neighbour-list cutoff
boundary.

**LAMMPS — verified against a built binary.** `pair_mamba` and `pair_mamba/kk`
were compiled against LAMMPS `develop` with LibTorch 2.2.0, Open MPI and Kokkos
Serial, and six of the eight integration cases pass: serial host, two-rank MPI, a
rank owning no atoms, Kokkos host, Kokkos with MPI, and NVE energy conservation.
The two that remain need an NVIDIA GPU, which this machine does not have.

On the CsPbI3 cubic cell the LAMMPS single-point energy reproduced the ASE
calculator exactly (-634.8335 eV), and 200 NVE steps at 1 fs conserved the total
energy to 0.0045 meV/atom with no trend. `tests/test_lammps_ghost_equivalence.py`
additionally checks the deployment contract *without* needing LAMMPS, by
reproducing what `pair_mamba.cpp` does to a periodic cell and comparing to ASE.

Building the pair style surfaced one real portability bug, now fixed: the
per-rank GPU mapping used `MPI_Comm_split_type` and `MPI_COMM_TYPE_SHARED`, which
LAMMPS's serial MPI stubs do not provide, so the style could not compile in a
serial build at all. See `lammps/README.md` for the reproducible build recipe.

## Provenance

`evaluate_phases.py`, `relax_single.py`, `free_energy_ti/run_ti.py`,
`free_energy_ti/ti_core.py`, `free_energy_ti/config.yaml` and the five structures
are ports of the author's Transformers-ACE reference tests. `ti_core.py` is pure
thermodynamics and is unchanged. The others differ only in the calculator, the
LAMMPS pair style name (`transformers_ace` -> `mamba`), the exported archive name
and the export call signature.
