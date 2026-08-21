"""Per-layer mixer schedules: a Mamba stack with sparse attention anchors.

Unrolling the state-space recurrence over the shell axis gives the radial
correlation kernel

    y_k = sum_k' K(k, k') u_k' ,
    K(k, k') = <Q_k, P_k'> exp[ -int_{xi_k'}^{xi_k} a(xi) dxi ] ,

so a state-space mixer is a *low-rank* kernel -- rank at most ``R d_s`` -- times
a *learned exponential band*.  Attention carries neither constraint.  They are
different priors on the same object, and ``experiments/mixer_kernel_spectroscopy.py``
measures which one each mixer realises at initialisation:

    mixer      eff. rank / L    band decay
    identity         0.000            --
    mamba            0.783         0.255
    attention        0.366         0.087

Mamba is banded and comparatively high rank; attention is low rank and
essentially unbanded.  Neither dominates, which is the situation in which a
mixed stack can beat either pure stack.

The cost argument runs the same way.  At the shell counts used here, L = 16..32,
attention costs O(L^2 d) ~ 1.6e4 MAC per atom, and measured against the identity
control attention runs at 1.5x while Mamba runs at 6-10x.  Replacing a Mamba
layer with an attention layer makes the model *cheaper*.

Physics risk is nil: every mixer in this module consumes only O(3)-invariant
controls and every one is smooth, so symmetry, energy conservation and the
derivative-order contract do not depend on the choice.  What the schedule
changes is the inductive bias, and only that.

Honest scope: the converged multi-seed mixer study on the full dataset is still
open, and in the three-seed pilot (``docs/PILOT_RESULTS.md``) no mixer is
distinguishable from the identity control.  The schedule is cheap and safe; it
is not yet known to be necessary.
"""

from __future__ import annotations

from typing import Sequence

# Kept in sync with the dispatch in ``EquivariantMambaACEBlock.__init__``.
MIXER_NAMES = ("mamba", "attention", "dense", "mlp", "deepsets", "identity")


def resolve_mixer_schedule(
    mixer_type: str,
    mixer_schedule: Sequence[str] | str | None,
    num_layers: int,
) -> tuple[str, ...]:
    """Return one mixer name per layer.

    ``mixer_schedule=None`` broadcasts the scalar ``mixer_type`` to every layer,
    which is the pre-schedule behaviour exactly.  A shorter schedule is cycled,
    so ``["mamba", "mamba", "attention"]`` over six layers is a period-three
    pattern; a schedule of the full length is used as written.
    """

    if int(num_layers) < 1:
        raise ValueError("num_layers must be positive")
    if mixer_schedule is None:
        entries = [str(mixer_type)]
    elif isinstance(mixer_schedule, str):
        entries = [mixer_schedule]
    else:
        entries = [str(entry) for entry in mixer_schedule]
    if not entries:
        raise ValueError("mixer_schedule must contain at least one mixer")
    entries = [entry.strip().lower() for entry in entries]
    unknown = sorted({entry for entry in entries if entry not in MIXER_NAMES})
    if unknown:
        raise ValueError(
            f"unknown mixer(s) {unknown} in mixer_schedule; "
            f"valid names are {list(MIXER_NAMES)}"
        )
    if len(entries) > int(num_layers):
        raise ValueError(
            f"mixer_schedule has {len(entries)} entries but the model has "
            f"{int(num_layers)} layers; a schedule may be shorter than the "
            "stack and be cycled, but never longer"
        )
    return tuple(entries[index % len(entries)] for index in range(int(num_layers)))


def anchored_schedule(
    num_layers: int,
    num_anchors: int,
    anchor: str = "attention",
    base: str = "mamba",
) -> tuple[str, ...]:
    """Place ``num_anchors`` attention layers evenly through a Mamba stack.

    Anchors go at ``floor((i + 1/2) * num_layers / num_anchors)`` for
    ``i = 0 .. num_anchors - 1``, which spaces them evenly and keeps them off
    both ends of the stack.  Nemotron 3 Super describes its own placement as a
    "periodic interleaving pattern" with attention "strategically inserted as
    global anchors", which is what this reproduces.

    Unlike :func:`resolve_mixer_schedule` this returns an explicit full-length
    schedule, so the anchor positions do not depend on the stack depth dividing
    the pattern length.

    See :func:`anchor_count_for` for how many anchors to ask for; the naive
    reading of the published ratio is off by a factor of about two.
    """

    num_layers = int(num_layers)
    num_anchors = int(num_anchors)
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if not 0 <= num_anchors <= num_layers:
        raise ValueError("num_anchors must lie in [0, num_layers]")
    for name in (anchor, base):
        if str(name).lower() not in MIXER_NAMES:
            raise ValueError(f"unknown mixer {name!r}; valid names are {list(MIXER_NAMES)}")
    schedule = [str(base).lower()] * num_layers
    for index in range(num_anchors):
        position = ((2 * index + 1) * num_layers) // (2 * num_anchors)
        schedule[min(position, num_layers - 1)] = str(anchor).lower()
    return tuple(schedule)


# One attention anchor per this many MTACE layers.  See ``anchor_count_for``.
MIXERS_PER_ANCHOR = 7


def anchor_count_for(num_layers: int, mixers_per_anchor: int = MIXERS_PER_ANCHOR) -> int:
    """How many attention anchors an ``num_layers``-deep MTACE stack should get.

    **The published ratio is easy to misread, and this project misread it.**
    The LatentMoE hybrid is tabulated as "52 (24 Mamba/MoE, 4 Attn)", which
    reads at a glance as 4 attention layers in 52, i.e. 7.7%.  But the 52 counts
    Mamba blocks, MoE blocks and attention blocks separately:

        24 Mamba + 24 MoE + 4 attention = 52 .

    The MoE blocks are feed-forward layers, not sequence mixers.  The number of
    *mixers* is therefore 24 + 4 = 28, and attention is 4/28 = 14.3% of them --
    roughly one in seven, not one in thirteen.

    That distinction is what matters here, because one MTACE layer contains both
    a mixer and a scalar residual block, so an MTACE layer corresponds to a
    (Mamba, MoE) pair rather than to a single tabulated layer.  The correct
    transfer is one anchor per seven MTACE layers.

    At least one anchor is returned for any stack of at least one layer: a
    hybrid with no attention is not a hybrid.
    """

    num_layers = int(num_layers)
    mixers_per_anchor = int(mixers_per_anchor)
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if mixers_per_anchor < 1:
        raise ValueError("mixers_per_anchor must be positive")
    return max(1, round(num_layers / mixers_per_anchor))


def nemotron_style_schedule(num_layers: int, mixers_per_anchor: int = MIXERS_PER_ANCHOR):
    """A Mamba stack with evenly spaced attention anchors at the published ratio."""

    return anchored_schedule(
        num_layers, anchor_count_for(num_layers, mixers_per_anchor)
    )
