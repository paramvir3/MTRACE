# Third-party provenance and attribution

MAMBA-ACE is released under the MIT License (see `LICENSE`).  It contains no
copied third-party source, but several components are **independent
reimplementations of published algorithms**, and one component reproduces the
numerics of a sibling research code.  This file states exactly what came from
where, so that reviewers, users and the manuscript's code-availability statement
have a single authoritative record.

## Reimplemented algorithms

### Mamba-3 state-space recurrence — `mtace/mamba3.py`

`Mamba3Direction`, `heavy_tail_activation`, `rotate_state_pairs`,
`rotate_state_halves`, the exponential-trapezoidal coefficients
`(alpha, beta, gamma)`, the normalized and biased key/query projections, and the
rank-R MIMO parameterization are an independent PyTorch reimplementation of the
recurrence described in

> A. Lahoti, K. Y. Li, B. Chen, C. Wang, A. Bick, J. Z. Kolter, T. Dao and
> A. Gu, *Mamba-3: Improved Sequence Modeling using State Space Principles*,
> arXiv:2603.15569 (2026),

and released in the `state-spaces/mamba` repository (Apache License 2.0).  No
Apache-licensed source is vendored here.  The optional fused kernels
(`mamba_ssm.ops.triton.mamba3.*`, `mamba_ssm.ops.tilelang.mamba3.*`) are imported
from the upstream package when it is installed and retain their own license; the
keyword names in our call sites (`Q`, `K`, `V`, `ADT`, `DT`, `Trap`, `MIMO_V`,
`MIMO_Z`, `MIMO_Out`, `Angles`, `rotary_dim_divisor`) are chosen to match that
public API and are the only place the upstream interface appears.

**Verification status.** The portable path is tested against an independent
serial recurrence in FP64 for value, first and second derivatives.  Numerical
equivalence between the portable path and the fused kernels has *not* been
demonstrated on this hardware; see the manuscript for the required GPU
parity check that must be run before any fused-inference claim.

### Muon optimizer — `mtace/optim.py`

`zeropower_via_newton_schulz5`, the quintic coefficients
`(3.4445, -4.7750, 2.0315)`, the normalized EMA/Nesterov recurrence and the
transpose convention reimplement

> K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse and J. Bernstein,
> *Muon: An optimizer for hidden layers in neural networks* (2024),
> <https://kellerjordan.github.io/posts/muon/>.

The `match_rms_adamw` scaling rule `0.2 * sqrt(max(A, B))` follows

> J. Liu *et al.*, *Muon is scalable for LLM training*, arXiv:2502.16982 (2025).

Parameter-role routing is our own and is documented in the manuscript.

### Equivariant operations — `e3nn`

Spherical harmonics, `o3.Linear`, `o3.Norm` and
`o3.FullyConnectedTensorProduct` come from the `e3nn` package (MIT), cited as
Geiger and Smidt, arXiv:2207.09453.

### Architectural ideas that are *not* ours

The manuscript cites these; they are listed here so the code record matches.

* The density trick, repeated equivariant tensor products and per-order linear
  readout of `physics.py::ACEV2Descriptor` follow the **MACE** construction
  (Batatia *et al.*, NeurIPS 2022), itself built on the **atomic cluster
  expansion** (Drautz, Phys. Rev. B **99**, 014104 (2019)).  Our contractions are
  learned `FullyConnectedTensorProduct` stacks, *not* the complete symmetric
  product (B) basis, so the completeness results of Dusson *et al.* do not apply
  to them.
* The scalar gate of `ssm.py::EquivariantMambaACEBlock`, which shares one
  coefficient over all magnetic components of an irrep, is the standard
  **equivariant gated nonlinearity** (Weiler *et al.* 2018; Thomas *et al.* 2018;
  `e3nn`'s `o3.Gate`; PaiNN).  It is not introduced by this work.
* The invariant map `J` (even scalars plus squared irrep norms) is the standard
  invariant extraction used by TFN/NequIP/PaiNN readouts and is `e3nn`'s
  `o3.Norm`.
* Strict locality with a single cutoff and no message passing is the design of
  **Allegro** (Musaelian *et al.* 2023), which is the correct baseline for our
  scaling claims.

## Sibling research code

The ACE frontend is described in the source as "TRACE-v2 compatible".  TRACE-v2
is the descriptor of the authors' own unpublished **Flash-ACE** research code.
`tests/test_v2_compatibility.py` compares against a sibling checkout when one is
present and skips otherwise.  No Flash-ACE source is included in this
repository; the frontend here was written independently against the same
equations so that MAMBA-ACE is self-contained.  Any publication using this code
must state that relationship explicitly.

## LAMMPS interface

`lammps/pair_mamba.*`, `lammps/pair_mamba_kokkos.*`, `lammps/patch_lammps.sh` and
`lammps/cmake/MAMBAACE.cmake` were written for this project.  Their structure —
a TorchScript atomic-energy model evaluated on owned plus ghost coordinates, with
forces from reverse-mode differentiation and the virial from a strain
derivative — deliberately follows the established pattern of the LAMMPS pair
styles for machine-learned potentials, in particular
`mir-group/pair_nequip_allegro` (reviewed at commit `2e19360b2639`, see
the interface review, now folded into the manuscript) and the MACE pair style.  Earlier revisions of
this repository used MACE's `.mace.pt` extension for the exported archive; that
was misleading and the extension is now `.mtace.pt`.

## Datasets

`examples/cspbi3/train.extxyz` is a 979-frame CsPbI3 subset used for smoke tests
and the pilot study.  It is not redistributed with any warranty of provenance and
must be replaced by a properly cited dataset for publication.
