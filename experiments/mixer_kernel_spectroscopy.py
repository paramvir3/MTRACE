#!/usr/bin/env python3
"""Measure the radial correlation kernel each mixer actually realises.

Every mixer in this architecture is, at bottom, a prior on the shell-to-shell
correlation kernel

    y_k = sum_k' K(k, k') u_k' .

Unrolling the state-space recurrence gives

    K(k, k') = <Q_k, P_k'> exp[ -int_{xi_k'}^{xi_k} a(xi) dxi ] ,

so the SSM is a *low-rank* kernel (rank <= R d_s) times a *learned exponential
band*.  Attention is full rank with no decay prior; DeepSets is diagonal plus
rank one; the dense mixer is an arbitrary L x L matrix; identity is the identity.

That turns the mixer ablation into a physics question with a yes/no answer:

    Is the true radial correlation kernel low-rank and exponentially banded?

If it is, the state-space prior is correct and should win on sample efficiency.
If it is full rank, attention is the right choice and the SSM is a handicap.

This script answers it by measuring K directly as the mixer Jacobian
``dy_k / du_k'``, which works uniformly for every mixer, trained or untrained.
Two numbers summarise each kernel:

* **effective rank** -- ``exp(H)`` for the Shannon entropy H of the normalised
  singular-value spectrum.  A rank-one kernel gives 1, a full white kernel gives
  L.  This is the participation ratio of the spectrum and is smoother than any
  thresholded rank.
* **band decay** -- the ratio of mean ``|K|`` on the first off-diagonal to the
  diagonal, and the exponential rate fitted to ``log mean|K_{k,k+d}|`` over d.

Usage:

    python experiments/mixer_kernel_spectroscopy.py                # untrained priors
    python experiments/mixer_kernel_spectroscopy.py --checkpoint model.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from mtace.model import MambaACEV2

DOUBLE = torch.float64


def mixer_kernel(layer, atoms: int, length: int, width: int, seed: int = 0):
    """Jacobian ``dy_k / du_k'`` of the mixer, averaged over channels and atoms.

    The mixer is generally nonlinear, so the Jacobian is evaluated at a
    representative operating point rather than being exact globally.  That is the
    right object anyway: it is the linear response of the shell axis, which is
    what "correlation kernel" means for a nonlinear map.
    """

    torch.manual_seed(seed)
    controls = torch.randn(atoms, length, width, dtype=DOUBLE, requires_grad=True)
    # Every mixer returns ``hidden + delta``.  Differentiating the raw output
    # therefore measures ``I + dDelta/du``, whose spectrum is dominated by the
    # identity and reports full rank for all mixers including the identity
    # control -- an artefact, not a result.  Subtracting the skip connection
    # isolates the mixing operator, which is the object the kernel taxonomy is
    # about.  For the identity mixer this correctly yields exactly zero.
    outputs = layer._mixer_states(controls, False) - controls
    kernel = np.zeros((length, length))
    for k in range(length):
        gradient = torch.autograd.grad(
            outputs[:, k].pow(2).sum(), controls, retain_graph=True
        )[0]
        # Row k of |K| up to a positive channel-wise factor.
        kernel[k] = gradient.norm(dim=-1).mean(dim=0).detach().numpy()
    return kernel


def effective_rank(kernel: np.ndarray) -> float:
    """exp(Shannon entropy of the normalised singular spectrum)."""

    singular = np.linalg.svd(kernel, compute_uv=False)
    total = singular.sum()
    if total <= 0.0:
        return 0.0
    probabilities = singular / total
    probabilities = probabilities[probabilities > 1e-15]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def band_profile(kernel: np.ndarray, max_offset: int = 6):
    """Mean |K| as a function of shell separation, and a fitted decay rate."""

    length = kernel.shape[0]
    profile = []
    for offset in range(min(max_offset, length)):
        values = [
            abs(kernel[k, k + offset]) for k in range(length - offset)
        ] + [abs(kernel[k + offset, k]) for k in range(length - offset)]
        profile.append(float(np.mean(values)))
    positive = [(d, v) for d, v in enumerate(profile) if v > 0.0]
    rate = float("nan")
    if len(positive) >= 3:
        d = np.array([p[0] for p in positive], dtype=float)
        y = np.log(np.array([p[1] for p in positive], dtype=float))
        # log|K| ~ -rate * d  =>  a positive rate is an exponential band.
        rate = float(-np.polyfit(d, y, 1)[0])
    return profile, rate


def build(mixer: str, length: int, **overrides):
    torch.manual_seed(0)
    settings = dict(
        r_max=6.0, l_max=2, num_radial=8, hidden_dim=32, num_layers=1,
        correlation_order=3, correlation_channels=8, num_shells=length,
        shell_r_min=1.5, shell_degree=5, avg_num_neighbors=12.0,
        invariant_norm="homogeneous", mixer_type=mixer, mamba_dim=32,
        mamba_d_state=8, mamba_headdim=8, readout_hidden=32,
        mamba_backend="torch",
    )
    settings.update(overrides)
    return MambaACEV2(**settings).double().eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="analyse a trained model instead of the priors")
    parser.add_argument("--shells", type=int, default=16)
    parser.add_argument("--atoms", type=int, default=8)
    parser.add_argument("--mixers", nargs="+",
                        default=["identity", "mlp", "dense", "attention", "mamba"])
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()

    records = {}
    if arguments.checkpoint:
        from mtace.checkpoint import restore_model

        model, _ = restore_model(arguments.checkpoint, "cpu")
        model = model.double().eval()
        pairs = [(f"checkpoint:{model.layers[0].mixer_type}", model)]
    else:
        pairs = [(m, build(m, arguments.shells)) for m in arguments.mixers]

    print(f"Radial MIXING kernel K(k,k') = d(y_k - u_k)/du_k'   "
          f"(L = {arguments.shells}, {arguments.atoms} atoms)\n")
    print(f"{'mixer':>22} {'eff. rank':>10} {'rank/L':>8} "
          f"{'K(k,k+1)/K(k,k)':>17} {'decay rate':>11}")
    for name, model in pairs:
        layer = model.layers[0]
        length = int(model.ace.sequence_length)
        width = layer.node_context.out_features
        kernel = mixer_kernel(layer, arguments.atoms, length, width)
        rank = effective_rank(kernel)
        profile, rate = band_profile(kernel)
        ratio = profile[1] / profile[0] if profile[0] > 0 else float("nan")
        records[name] = {
            "effective_rank": rank,
            "relative_rank": rank / length,
            "band_profile": profile,
            "decay_rate": rate,
            "first_offdiagonal_ratio": ratio,
        }
        print(f"{name:>22} {rank:10.3f} {rank/length:8.3f} "
              f"{ratio:17.4f} {rate:11.4f}")

    print("\nHow to read this:")
    print("  effective rank ~ 1   : the mixer applies essentially one radial mode")
    print("  effective rank ~ L   : full-rank mixing, no low-rank structure")
    print("  decay rate > 0       : an exponentially banded kernel")
    print("  decay rate ~ 0       : long-range, unbanded shell coupling")
    print("\nThe state-space prior is low-rank AND banded.  If a trained dense or")
    print("attention mixer turns out full-rank and unbanded, that prior is wrong")
    print("for this data and the SSM is paying a cost for nothing -- which is a")
    print("publishable negative result, and far cheaper to obtain than a")
    print("converged accuracy comparison.")

    if arguments.json:
        destination = Path(arguments.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(records, indent=2))
        print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
