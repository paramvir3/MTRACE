# MTACE

A machine-learning interatomic potential built on a **hybrid Mamba-attention
stack**: a bidirectional Mamba-3 state-space mixer with sparse self-attention
anchors, communicating between smooth radial shells of a TRACE-v2 Atomic Cluster
Expansion descriptor.

This repository has its own git history, starting at the initial commit. The
descriptor, mixer and LAMMPS machinery derive from MAMBA-ACE, which still lives
at `/Users/paramvir/Documents/Codex/2026-07-12/this/work/MAMBA-ACE-architecture-v8`
and is frozen from MTACE's point of view; that tree is where the pre-MTACE
history is kept. **No git remote is configured** — this is deliberate, do not
add one without asking.

## Read this first

[`docs/MTACE_METHODS.tex`](docs/MTACE_METHODS.tex) is the manuscript, and it is
now the single source of truth for this project: the architecture, every
equation, the derivative-order contract, the diagnostics, and an explicit
separation of what is proved, what is measured, what is inherited from the
language-model literature, and what is still conjecture. It replaces the earlier
roadmap, methods drafts, audits and study notes, all of which were removed from
`docs/` in favour of it; they remain recoverable from git history.

Three ideas transferred from the Nemotron 3 / LatentMoE papers are implemented
at checkpoint `architecture_version` 11:

1. **Per-layer mixer schedule** — `mtace/schedule.py`, `mixer_schedule=[...]`.
   The central mechanism: mostly Mamba with attention as sparse anchors.
2. **Equivariant `r_eff` diagnostic** — `mtace/diagnostics.py`,
   `experiments/effective_rank.py`.
3. **Smooth compact-support expert routing** — `mtace/routing.py`,
   `num_experts=N`.

Both new model settings are inert at their defaults, so a v10 checkpoint
restores bit-for-bit with no migration.

## Standing constraints

These come from the author and hold for every change:

- **Never hallucinate.** If a number, equation or API is not verified, say so.
- **Physics with precise equations.** Every mechanism gets its equation written
  down before it gets code.
- **Test everything** — that the equations are right, the algorithms are right,
  the physics is right. Symmetry and conservation are checked in FP64 against
  finite differences, not assumed.
- **Focus on expressive and smooth behaviour.** Expressivity claims are backed
  by a degree/body-order argument; smoothness claims are backed by the
  derivative-order contract, stated in the manuscript — cubic shells make the
  third derivative jump by `6[(L-1)/(r_c - r_min)]^3` at every shell radius, and
  that jump *grows* as the mesh is refined, so anything touching a third
  derivative needs `shell_degree: 5`.
- Report results honestly, including negative ones. This project has withdrawn
  two of its own claims after measurement, and the central claim that
  state-space mixing helps is **not yet established**: in the pilot no mixer is
  distinguishable from an identity control. The manuscript's "What is
  established, and what is not" and "Limitations" sections carry both, and must
  keep carrying them.

## Running things

Full suite — 324 tests, about two and a half minutes, all symmetry and
conservation checks in FP64:

```bash
python3 -m pytest -q
```

The package is not pip-installed; `python3 -m pytest` from the repository root
puts the tree on `sys.path`. 12 tests skip without a LAMMPS binary
(`MAMBA_ACE_LAMMPS`) or CUDA — that is expected on this machine.

Scripts under `experiments/` are not on `sys.path` when run directly, so they
take the same prefix the rest of the tree does:

```bash
PYTHONPATH=. python3 experiments/effective_rank.py --frames examples/cspbi3/train.extxyz
```

Train on the bundled CsPbI3 set:

```bash
cd training/cspbi3 && python3 ../../train.py --config config.yaml
```

## Naming, and one trap

The Python package is `mtace` and exported models use `.mtace.pt`. The **LAMMPS
layer was deliberately left as MAMBA-ACE**: pair styles `mamba` and `mamba/kk`,
files `lammps/pair_mamba.*`, environment variables `MAMBA_ACE_*`, and the
TorchScript metadata key `mamba_ace_format_version`. That layer is the only part
verified against a compiled binary, and renaming it invalidates that
verification. The metadata key is ABI in particular — `pair_mamba.cpp` reads it
and rejects models that do not match, so renaming it in `deployment.py` alone
makes new exports unloadable by an existing binary. Do not "tidy" any of this
without a rebuild.
