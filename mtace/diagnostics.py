"""Equivariant effective rank of the ACE token features.

Latent-space routing compresses the input to width ``l`` before the expensive
map and lifts back, at compression ratio ``alpha = d / l``.  Two results make
that admissible.  Barron's theorem (1993) -- a one-hidden-layer network with
``u`` nonlinear units achieves mean squared error ``O(1/u)`` independent of the
input dimension ``d`` -- says the approximation budget is set by the nonlinear
width, not the input width, so compressing the input is free provided the
nonlinear budget is held fixed.  Against that sits a task-specific effective
rank ``r_eff`` below which quality collapses, which lower-bounds the admissible
latent width.

MTACE already has the structural half of this: ``coupling_channels`` and
``shell_pair_channels`` down-project to a reduced *equivariant* space, run the
tensor product there and lift back.  That was derived from a memory wall -- the
per-sample weights of the full token-to-node map cost 328 MiB per 1000 atoms at
L = 32, against 23 MiB at eight coupling channels -- and the width was chosen by
hand.  ``coupling_channels = 8`` against ``correlation_channels = 16`` is
``alpha = 2``.

This module measures the bound instead of guessing it.  For each irrep order
``l``, stack the token features over a dataset into ``X^(l)`` of shape
``(N_atoms L) x (c (2l+1))``, take the singular values
``sigma_1 >= sigma_2 >= ...`` and report the participation ratio

    r_eff = exp( -sum_i p_i log p_i ) ,   p_i = sigma_i / sum_j sigma_j .

This is the estimator ``experiments/mixer_kernel_spectroscopy.py`` already
applies to the mixer kernel, applied to features instead.

**Why this is a legitimate equivariant diagnostic.**  Under a global rotation
``R`` the block transforms as ``X -> X (I_c kron D^(l)(R))^T``.  That matrix is
orthogonal, so the singular values -- and therefore ``r_eff`` -- are *exactly*
invariant.  It is not approximately invariant and does not need orientation
averaging.

Two ranks are reported per block, and they measure different things:

``r_eff_full``      the roadmap definition above, over all ``c (2l+1)``
                    columns.
``r_eff_channel``   the same estimator applied to the ``m``-contracted Gram
                    ``G_{c c'} = sum_{rows} sum_m x_{c l m} x_{c' l m}``, a
                    ``c x c`` invariant.  This is the one that bounds a channel
                    width such as ``coupling_channels``, because that parameter
                    truncates the multiplicity, never the ``2l+1`` magnetic
                    components -- an equivariant projection cannot do the
                    latter.

They are not independent.  If the feature distribution is O(3) invariant then by
Schur's lemma ``E[x_{clm} x_{c'lm'}] = delta_{mm'} M_{c c'}``, so the full Gram
is ``M kron I_{2l+1}``, its spectrum is that of ``M`` with multiplicity
``2l+1``, and the entropies differ by exactly ``log(2l+1)``:

    r_eff_full = (2l + 1) * r_eff_channel .

``tests/test_effective_rank.py`` checks that identity on rotation-averaged
features.  A measured departure from it is itself informative: it means the
sampled features are anisotropic, i.e. the dataset does not cover orientations.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from e3nn import o3


def participation_ratio(values: torch.Tensor) -> float:
    """``exp(-sum_i p_i log p_i)`` for ``p_i = v_i / sum_j v_j``, ``v >= 0``.

    Returns 0.0 for an all-zero spectrum.  Equals ``n`` for ``n`` equal values
    and 1.0 when one value carries everything, so it reads as "how many
    directions are actually in use".
    """

    values = values.to(torch.float64).clamp(min=0.0)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probabilities = values / total
    positive = probabilities[probabilities > 0.0]
    entropy = float(-(positive * positive.log()).sum())
    return math.exp(entropy)


def _singular_values_from_gram(gram: torch.Tensor) -> torch.Tensor:
    """``sigma_i(X)`` from ``G = X^T X``, i.e. ``sqrt`` of its eigenvalues.

    Forming the Gram squares the condition number, which is acceptable here
    because a participation ratio is dominated by the large singular values and
    because the Gram is what accumulates additively over a dataset.  Everything
    is done in float64 and round-off negatives are clamped away.
    """

    eigenvalues = torch.linalg.eigvalsh(gram.to(torch.float64))
    return eigenvalues.clamp(min=0.0).sqrt().flip(0)


class IrrepGramAccumulator:
    """Accumulate per-irrep Gram matrices of equivariant features over batches.

    Features are supplied with shape ``(..., irreps.dim)``; every leading axis is
    flattened into the row index, so shell tokens of shape
    ``(atoms, shells, dim)`` contribute ``atoms * shells`` rows, which is the
    ``(N_atoms L)`` of the definition.
    """

    def __init__(self, irreps):
        self.irreps = o3.Irreps(irreps)
        self.rows = 0
        self._full: list[torch.Tensor] = []
        self._channel: list[torch.Tensor] = []
        for multiplicity, irrep in self.irreps:
            multiplicity = int(multiplicity)
            dimension = int(irrep.dim)
            self._full.append(
                torch.zeros(
                    (multiplicity * dimension, multiplicity * dimension),
                    dtype=torch.float64,
                )
            )
            self._channel.append(
                torch.zeros((multiplicity, multiplicity), dtype=torch.float64)
            )

    @torch.no_grad()
    def update(self, features: torch.Tensor) -> "IrrepGramAccumulator":
        if features.shape[-1] != self.irreps.dim:
            raise ValueError(
                f"features must have last dimension {self.irreps.dim}, "
                f"got {features.shape[-1]}"
            )
        flat = features.reshape(-1, self.irreps.dim).to(torch.float64)
        self.rows += int(flat.shape[0])
        offset = 0
        for index, (multiplicity, irrep) in enumerate(self.irreps):
            multiplicity = int(multiplicity)
            dimension = int(irrep.dim)
            width = multiplicity * dimension
            block = flat[:, offset : offset + width]
            # The contraction runs on whatever device the features live on, and
            # only the small (D x D) result is moved.  Accumulating on the
            # features' device instead would tie the accumulator to it, and
            # accumulating the features on the host would copy every row.
            store = self._full[index].device
            self._full[index] += (block.T @ block).to(store)
            # Contracting over m is what makes the c x c Gram an O(3) invariant:
            # sum_m x_{c m} x_{c' m} is a scalar product of two irrep vectors.
            resolved = block.reshape(-1, multiplicity, dimension)
            self._channel[index] += torch.einsum(
                "ncm,ndm->cd", resolved, resolved
            ).to(store)
            offset += width
        return self

    def report(self) -> list[dict[str, float | str | int]]:
        """One record per irrep block, in the order of ``irreps``."""

        records: list[dict[str, float | str | int]] = []
        for index, (multiplicity, irrep) in enumerate(self.irreps):
            multiplicity = int(multiplicity)
            full_sigma = _singular_values_from_gram(self._full[index])
            channel_sigma = _singular_values_from_gram(self._channel[index])
            full = participation_ratio(full_sigma)
            channel = participation_ratio(channel_sigma)
            records.append(
                {
                    "irrep": str(irrep),
                    "l": int(irrep.l),
                    "channels": multiplicity,
                    "columns": multiplicity * int(irrep.dim),
                    "rows": self.rows,
                    "r_eff_full": full,
                    "r_eff_channel": channel,
                    # The Schur prediction, for comparison against r_eff_full.
                    "r_eff_full_predicted": channel * float(irrep.dim),
                    # What fraction of the available channel directions carry
                    # signal.  Compress to just above r_eff_channel, not below.
                    "channel_utilisation": channel / multiplicity,
                }
            )
        return records


@torch.no_grad()
def token_effective_rank(
    model,
    batches: Iterable[dict],
    device: str | torch.device | None = None,
) -> list[dict[str, float | str | int]]:
    """Measure ``r_eff`` of the ACE shell tokens over a dataset.

    ``batches`` yields mappings with the keys ``z``, ``pos``, ``cell``,
    ``edge_index`` and ``edge_shift`` -- the signature the model's own
    diagnostics use.  The tokens are the tensor that ``coupling_down`` and
    ``shell_pair_down`` compress, so their per-irrep channel rank is the
    quantity that bounds ``coupling_channels`` and ``shell_pair_channels``.

    ``device`` defaults to whichever device the model already sits on.  Passing
    one that the model is not on moves the inputs but not the weights, which
    only fails later and further away.
    """

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    accumulator = IrrepGramAccumulator(model.ace.irreps_correlation)
    for batch in batches:
        z = batch["z"].to(device)
        pos = batch["pos"].to(device)
        cell = batch["cell"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_shift = batch["edge_shift"].to(device)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        if edge_shift.numel() > 0:
            edge_vec = edge_vec + edge_shift.to(pos) @ cell
        edge_len = torch.linalg.vector_norm(edge_vec, dim=-1)
        _, tokens, _, _ = model.ace(
            model.species_embedding(z), edge_index, edge_vec, edge_len
        )
        accumulator.update(tokens)
    return accumulator.report()


def format_effective_rank(records: Iterable[dict]) -> str:
    """Render :meth:`IrrepGramAccumulator.report` as a fixed-width table."""

    header = (
        f"{'irrep':>8} {'chan':>5} {'r_eff_ch':>9} {'util':>6} "
        f"{'r_eff_full':>11} {'Schur':>9}"
    )
    lines = [header, "-" * len(header)]
    for record in records:
        lines.append(
            f"{record['irrep']:>8} {record['channels']:>5d} "
            f"{record['r_eff_channel']:>9.3f} {record['channel_utilisation']:>6.3f} "
            f"{record['r_eff_full']:>11.3f} {record['r_eff_full_predicted']:>9.3f}"
        )
    return "\n".join(lines)
