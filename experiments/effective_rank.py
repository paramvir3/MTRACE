#!/usr/bin/env python3
"""Measure the equivariant effective rank of the ACE shell tokens.

``coupling_channels`` and ``shell_pair_channels`` compress the token features to
a reduced equivariant space before the expensive tensor product and lift back.
Those widths were chosen against a memory wall -- the per-sample weights of the
full token-to-node map cost 328 MiB per 1000 atoms at L = 32, against 23 MiB at
eight coupling channels -- and not against anything measured.  At
``coupling_channels = 8`` versus ``correlation_channels = 16`` the compression
ratio is ``alpha = 2``.

Barron's theorem says the approximation budget is set by the nonlinear width and
not by the input width, so compressing the input is free provided the nonlinear
budget is held fixed.  What bounds the compression from below is the effective
rank of the features themselves.  This script measures it, per irrep block:

    r_eff = exp( -sum_i p_i log p_i ) ,   p_i = sigma_i / sum_j sigma_j .

Read the output as follows.  ``r_eff_ch`` is the number of *channel* directions
actually carrying signal in that irrep block, and it is the quantity that bounds
a channel width; compress to just above it, never below.  ``util`` is
``r_eff_ch / channels``: a block at 0.35 is using about a third of the
multiplicity it was given.  ``Schur`` is the isotropic prediction
``(2l+1) * r_eff_ch`` for ``r_eff_full``; a large gap between the two means the
sampled features are anisotropic, which is a statement about the dataset's
orientational coverage rather than about the model.

The estimator is exactly invariant under rotation, translation, inversion and
atom relabeling, so a single unrotated pass over the data suffices -- see the
module docstring of ``mtace/diagnostics.py`` for the orthogonality argument.

Usage:

    python experiments/effective_rank.py --frames examples/cspbi3/train.extxyz
    python experiments/effective_rank.py --frames data.extxyz --checkpoint model.pt

Without ``--checkpoint`` this measures the *untrained* tokenizer, which is still
informative -- the ACE descriptor is largely fixed by the basis rather than by
training -- but the number that should decide a production width is the one
measured on a trained checkpoint.  The script says which it reported.
"""

from __future__ import annotations

import argparse
import json

import torch
from ase.io import read

from mtace.checkpoint import restore_model
from mtace.data import build_neighbor_tensors
from mtace.diagnostics import format_effective_rank, token_effective_rank
from mtace.model import MambaACEV2


def frame_batches(frames, r_max, dtype=torch.float64):
    """Yield the mapping ``token_effective_rank`` consumes, one frame at a time."""

    for atoms in frames:
        edge_index, edge_shift = build_neighbor_tensors(atoms, r_max, dtype=dtype)
        if edge_index.shape[1] == 0:
            continue
        yield {
            "z": torch.as_tensor(atoms.numbers, dtype=torch.long),
            "pos": torch.as_tensor(atoms.positions, dtype=dtype),
            "cell": torch.as_tensor(atoms.cell.array, dtype=dtype),
            "edge_index": edge_index,
            "edge_shift": edge_shift,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, help="extxyz trajectory")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--count", type=int, default=50, help="frames to accumulate")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--r-max", type=float, default=6.0)
    parser.add_argument("--num-shells", type=int, default=32)
    parser.add_argument("--correlation-channels", type=int, default=16)
    parser.add_argument("--json", default=None, help="also write the report here")
    arguments = parser.parse_args()

    if arguments.checkpoint is not None:
        model, _ = restore_model(arguments.checkpoint, mamba_backend="torch")
        model = model.double().eval()
        r_max = float(model.r_max)
        provenance = f"trained checkpoint {arguments.checkpoint}"
    else:
        torch.manual_seed(0)
        r_max = float(arguments.r_max)
        model = MambaACEV2(
            r_max=r_max,
            num_shells=arguments.num_shells,
            correlation_channels=arguments.correlation_channels,
            mamba_backend="torch",
        ).double().eval()
        provenance = "untrained tokenizer (no --checkpoint given)"

    frames = read(arguments.frames, index=f"0:{arguments.count}:{arguments.stride}")
    records = token_effective_rank(model, frame_batches(frames, r_max))

    print(f"source          : {provenance}")
    print(f"frames          : {len(frames)}  ({records[0]['rows']} atom-shell rows)")
    print(f"token irreps    : {model.ace.irreps_correlation}")
    layer = model.layers[0]
    if layer.coupling_mode == "path_weights":
        print(f"compressed width: coupling_channels = {layer.coupling_channels}")
    else:
        print(
            "compressed width: coupling_mode='gate' does not compress; the "
            "bound below applies to shell_pair_channels "
            f"({layer.shell_pair_channels}) and to coupling_channels if "
            "path_weights is enabled"
        )
    print()
    print(format_effective_rank(records))
    print()
    worst = max(records, key=lambda record: record["channel_utilisation"])
    print(
        "The binding block is "
        f"{worst['irrep']} at r_eff_ch = {worst['r_eff_channel']:.2f} of "
        f"{worst['channels']} channels.  A compressed width below that is "
        "measured to be lossy; above it, the Barron argument says the cost is "
        "carried by the nonlinear budget instead."
    )

    if arguments.json is not None:
        with open(arguments.json, "w") as stream:
            json.dump({"provenance": provenance, "blocks": records}, stream, indent=2)
        print(f"wrote {arguments.json}")


if __name__ == "__main__":
    main()
