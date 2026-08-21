# MAMBA-ACE for LAMMPS

The `mamba` and `mamba/kk` pair styles evaluate the same exported strictly
local atomic-energy model. The host style stages contiguous pinned tensors for
CUDA. The Kokkos style constructs positions, atomic numbers, and directed
edges in Kokkos memory and shares those allocations with LibTorch without an
intermediate copy.

## 1. Export a checkpoint

```bash
mtace-export-lammps checkpoint.pt model.mtace.pt --elements H O
```

The exporter forces the portable differentiable Mamba backend, freezes the
TorchScript atomic-energy graph, validates a different atom/edge shape and its
coordinate gradient, and writes cutoff, precision, units, architecture, and
element metadata into the archive.
This contract is intentionally FP32/FP64 and does not embed the optional
Python-side Mamba-3 Triton/TileLang operator. `mamba/kk` can share Kokkos-CUDA
memory with LibTorch, but it evaluates this portable TorchScript graph.

## Verification status

`pair_mamba` has been built and tested end to end. Measured on macOS 15
(Apple clang 17, LibTorch 2.2.0 CPU, Open MPI 3.1, Kokkos Serial):

| test | status |
|---|---|
| `test_host_matches_ase` | pass |
| `test_two_rank_mpi_matches_ase` | pass |
| `test_mpi_rank_without_owned_atoms_matches_ase` | pass |
| `test_kokkos_host_matches_ase` | pass |
| `test_kokkos_two_rank_mpi_matches_ase` | pass |
| `test_nve_conserves_energy` | pass |
| `test_cuda_model_matches_ase` | needs an NVIDIA GPU |
| `test_kokkos_cuda_matches_ase` | needs an NVIDIA GPU |

On a 20-atom CsPbI3 cubic cell the LAMMPS single-point potential energy
reproduced the ASE calculator exactly (-634.8335 eV), and 200 NVE steps at 1 fs
conserved the total energy to 0.09 meV over the whole cell, i.e. 0.0045 meV/atom
with no visible trend.

Two build notes that cost real time if you hit them cold:

* **`pair_mamba` requires a real MPI or the `MPI_STUBS` guard.** LAMMPS's serial
  stubs implement neither `MPI_Comm_split_type` nor `MPI_COMM_TYPE_SHARED`, which
  the per-rank GPU mapping uses. The style now compiles in both configurations.
* **`pair_mamba/kk` requires LAMMPS `develop`.** It uses the `kkfloat`/`kkacc`
  mixed-precision Kokkos view typedefs, which do not exist in `stable_29Aug2024`
  or `stable_22Jul2025`. The host style builds on any of them.

Reproduce:

```bash
git clone --depth 1 --branch develop https://github.com/lammps/lammps.git
./lammps/patch_lammps.sh ./lammps
cmake -S lammps/cmake -B build -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -D BUILD_MPI=on -D PKG_KOKKOS=on -D Kokkos_ENABLE_SERIAL=on
cmake --build build --parallel

MAMBA_ACE_LAMMPS=$PWD/build/lmp \
MAMBA_ACE_MPIEXEC=$(which mpirun) \
MAMBA_ACE_TEST_KOKKOS=1 \
  python -m pytest tests/test_lammps_interfaces.py -v
```

## 2. Build LAMMPS

LAMMPS 10 September 2025 or newer and LibTorch/PyTorch 2.2 or newer are
required. Use the same LibTorch installation used for export.

```bash
./lammps/patch_lammps.sh /path/to/lammps
cmake -S /path/to/lammps/cmake -B build-lammps \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-lammps --parallel
```

For NVIDIA Kokkos, add the Kokkos options appropriate to the installed CUDA
architecture, for example `-D PKG_KOKKOS=on -D Kokkos_ENABLE_CUDA=on` and a
`Kokkos_ARCH_*` value. LAMMPS must use double precision for coordinates and
forces. Launch one MPI rank per GPU. The automatic mapping uses the rank within
each shared-memory node; scheduler GPU binding with one visible GPU per rank is
preferred. `MAMBA_ACE_DEVICE=cpu|cuda|cuda:N` overrides automatic selection.

## 3. Run

```lammps
units metal
atom_style atomic
newton on
pair_style mamba device auto check_finite yes
pair_coeff * * model.mtace.pt H O
```

The names after the model map LAMMPS types 1 through N to exported elements.
Every type must be mapped. For each 1--2, 1--3, and 1--4 level in a molecular
atom style, the LJ and Coulomb special weights must not both be zero; that
combination removes the pair before this style can see it. MAMBA-ACE masks the
LAMMPS special-neighbor bits and does not scale learned interactions by either
weight. Thus `special_bonds lj 1 1 1` is sufficient, while a nonzero Coulomb
weight also preserves the neighbor for mixed MLP/electrostatic models.

For Kokkos/CUDA:

```bash
mpiexec -n 4 lmp -k on g 4 -sf kk -pk kokkos neigh full -in in.mamba
```

Use `pair_style mamba/kk device cuda` when suffix selection is not enabled.
MAMBA-ACE requests a **full** neighbor list (`NeighConst::REQ_FULL`), so pass
`-pk kokkos neigh full`; a half list cannot supply every directed
neighbor-to-center edge that the strictly local atomic energy needs.
The Kokkos implementation supports host and CUDA execution. HIP and SYCL are
rejected until LibTorch and Kokkos memory/stream interoperability is available
for those backends.

## Ownership and tensors

For rank `r`, the scalar differentiated by LibTorch is

```text
E_r = sum(i in owned_r) E_i({r_j - r_i : |r_j-r_i| < r_max}).
```

The full neighbor list supplies every directed neighbor-to-center edge for an
owned center. LAMMPS ghost coordinates already contain the correct periodic
image, so no cell-shift reconstruction is needed. Differentiation with respect
to all owned and ghost coordinates produces partial rank forces; `newton pair
on` reverse-communicates ghost contributions exactly once. Global energy and
virial are ordinary MPI sums of rank-local contributions.

The configurational virial is evaluated independently of force-position
postprocessing:

```text
W_r = -sym[d E_r({r}(I + epsilon)) / d epsilon] at epsilon=0.
```

LAMMPS order is `(xx, yy, zz, xy, xz, yz)`. ASE reports tensile-positive
stress `sigma = -W/V` in `(xx, yy, zz, yz, xz, xy)` order. A per-atom virial
is deliberately unsupported because a many-body atomic-energy virial
partition is not unique. Both styles expose the model's per-center atomic
energies through LAMMPS per-atom energy computes.

## Scaling notes

- The potential is strictly local; there is no model communication beyond the
  normal LAMMPS halo exchange and reverse force communication.
- One model instance is loaded per MPI rank. Do not place multiple ranks on a
  GPU unless memory and throughput have been measured for that choice.
- `mamba/kk` uses zero-copy Kokkos tensors, but currently fences at the
  Kokkos/LibTorch boundary for correctness because the two runtimes do not
  expose a stable common-stream ABI. Model kernels remain asynchronous within
  LibTorch.
- The model cutoff is exact and independent of the LAMMPS neighbor skin.
- `check_finite yes` is the default and terminates collectively if an owned
  energy, any owned/ghost force contribution, or the requested virial is NaN or
  infinite. It can be disabled only after an application-specific performance
  and stability validation.
- CUDA runtime correctness must be validated on the exact LAMMPS, Kokkos,
  LibTorch, driver, and GPU stack used for production before long MD runs.

## Production parity gates

The opt-in integration suite compares total energy, every force component,
pressure/virial, and summed atomic energies against ASE. Set the executable and
run the host and MPI gates first:

```bash
MAMBA_ACE_LAMMPS=/path/to/lmp \
MAMBA_ACE_MPIEXEC="mpiexec --map-by slot:OVERSUBSCRIBE" \
python -m pytest tests/test_lammps_interfaces.py
```

On an NVIDIA build, additionally set `MAMBA_ACE_TEST_CUDA=1` for staged
LibTorch-CUDA execution, `MAMBA_ACE_TEST_KOKKOS=1` for the Kokkos host path, and
`MAMBA_ACE_TEST_KOKKOS_CUDA=1` for the Kokkos-CUDA zero-copy path. A production
release must not mark an accelerator path validated when its corresponding test
was skipped.
