# CsPbI3 training run

Trains a MTACE architecture-v10 potential on the 979-frame CsPbI3
trajectory, using the same dataset, split and descriptor capacity as the TRACE
reference run so the two are directly comparable.

## Run

```bash
cd training/cspbi3
python ../../train.py --config config.yaml
```

`model.pt` is the best-validation checkpoint and holds the **EMA-averaged**
weights; that is the file every downstream test should use. `model_last.pt`
holds the raw iterate and exists so a run can be resumed exactly:

```bash
python ../../train.py --config config.yaml --resume model_last.pt
```

## Dataset

The trajectory is referenced in place at `../../examples/cspbi3/train.extxyz`
rather than copied. It is byte-identical to the TRACE reference dataset
(SHA256 `746320f0fc427346c9b368799367fd5794ecdc870565532c4ed120aecd8e399c`).

| property | value |
|---|---|
| frames | 979 |
| composition | Cs8Pb8I24, 40 atoms, identical in every frame |
| labels | energy, forces, stress (all 979 frames) |
| energy spread | 30.6 meV/atom about the mean |
| force RMS | 0.259 eV/Angstrom, max 2.57 |
| stress RMS | 1.39e-3 eV/Angstrom^3 |
| shortest interatomic distance | 2.680 Angstrom |
| mean directed neighbours / atom | 16.62 |

## Two configurations

`config.yaml` is the baseline. `config_pairs.yaml` differs **only** in the
architecture-v10 expressivity terms, so running both is a controlled comparison:

| | baseline | pairs |
|---|---|---|
| `shell_pair_channels` / `width` | 0 | 8 / 2 |
| `coupling_mode` | `gate` | `path_weights` |
| `invariant_pair_channels` | 0 | 4 |
| parameters | 230,305 | 258,877 |

Watch `gate_shell_dependence` in the log. `residual_fraction` is the share of the
equivariant update that a shell-constant gate cannot reproduce. Because
`sum_k T_ik = A_i`, a shell-constant gate reduces the mixer exactly to the direct
ACE path, so a value near zero means the mixer is decorative **regardless of what
the error columns say**. It should grow during training. This is the falsifiable
test of whether the sequence mixer earns its cost.

## Choices that are not free parameters

**`shell_degree: 5` is required, not preference.** Cubic shell splines are C2 but
not C3, with a third-derivative jump of `6*((L-1)/(r_max-r_min))^3` at every shell
radius. Third-order interatomic force constants are exactly that derivative, and
CsPbI3 is a strongly anharmonic halide perovskite whose octahedral tilting,
ultralow lattice thermal conductivity and soft-mode transitions all live there.
The free-energy workflow integrates across that anharmonicity. See
[../../docs/DERIVATIVE_CONTRACT.md](../../docs/DERIVATIVE_CONTRACT.md).

**`shell_r_min: 2.0`.** Measured over all 979 frames, the shortest interatomic
distance is 2.680 Angstrom. Uniform shells on `[0, r_max]` leave 4 of these 16
shells below any distance that ever occurs (12 of 32 at `num_shells: 32`), so the
mixer would scan hard zeros. `r_min = 2.0` leaves a 0.680 Angstrom margin and no
dead shell. The trainer measures the margin itself and refuses to start if it is
too small.

**Reference energies are fixed, not fitted.** Every frame is Cs8Pb8I24, so the
species-count matrix has rank 1 for three species: per-species reference energies
are **not identifiable** and a least-squares fit returns an arbitrary point on a
two-parameter family. `solve_atomic_energies` would correctly refuse. The one
identifiable quantity is a single per-atom offset, set here to the training-set
mean `-31.8596670627702` eV/atom. It cannot affect forces, stress, or any
relative energy between stoichiometric CsPbI3 phases, because it cancels exactly
in every difference per formula unit. It only fixes the absolute zero.

**`normalize_loss_weights: true`.** Each squared-error term is divided by the
dataset variance of its target, so the weights are dimensionless and a term equal
to 1.0 means "no better than predicting the mean". The three raw scales here span
three orders of magnitude, so unnormalised weights make the balance opaque.

**`mamba_backend: torch`.** Force and stress training differentiates a force, so
it needs a double backward. The fused CUDA kernels expose no guaranteed double
backward and the trainer selects the portable scan regardless; setting it
explicitly documents the intent.

## Cost

Roughly 40 ms per atom per force-training step on CPU. At 979 frames of 40 atoms
that is around 26 minutes per epoch on 10 threads, so 100 epochs is a multi-day
CPU run. Set `device: cuda` on a GPU node. To check the pipeline first, add
`stop_after_epoch: 2`.
