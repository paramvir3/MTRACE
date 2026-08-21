"""Smooth compact-support expert routing.

Mixture-of-experts layers select experts with a top-k rule.  Top-k is a
discontinuous function of its inputs, and in an interatomic potential the
routing scores are functions of the atomic positions, so at a score crossing

    E(R) jumps   =>   F = -grad E   acquires a delta function.

The energy is then not continuous, let alone twice differentiable, and molecular
dynamics across that boundary is ill-posed rather than merely inaccurate.  A
language model tolerates this because its output need not be a smooth function
of a continuous input; ours must be.  Softmax over all experts is smooth but
dense, which discards the point of the construction.

This module replaces the rank cut by a compactly supported smooth switch.  With
``h_i`` the O(3)-invariant per-atom context, ``s_e(h_i)`` a learned score,
``theta_e`` a learned threshold, ``b_e`` a load-balancing bias carried outside
the gradient (see :class:`CompactSupportRouter`; it is zero unless balancing is
switched on) and ``tau > 0`` a width,

    u_e = (theta_e + b_e - s_e(h_i)) / tau ,
    w_e = f( clamp(u_e, 0, 1) ) ,
    y_i = Shared(h_i) + W_up sum_e w_e Expert_e(W_down h_i) ,

where ``W_down`` and ``W_up`` are the identity unless latent expert compression
is enabled (see :class:`RoutedScalarFFN`), and ``f`` is a switching polynomial
with ``f(0) = 1`` and

    f'(1) = ... = f^(k)(1) = 0 ,   f'(0) = ... = f^(k)(0) = 0

to the derivative-order contract ``k``.  Both joins of ``f . clamp`` are then
``C^k``, so an expert switching off is exactly as smooth as an atom leaving the
neighbour list, and for the identical reason.

Three properties make this usable:

1. **True sparsity.**  ``w_e = 0`` *exactly* -- not small -- when
   ``s_e <= theta_e + b_e - tau``.  The factored evaluation below returns a
   floating-point zero at ``x = 1``, so the expert can be skipped and the
   arithmetic really is saved.  :class:`RoutedScalarFFN` does skip it.
2. **No singularity.**  The weights are deliberately left *unnormalised*, so
   there is no denominator that can vanish.  The always-on ``Shared`` branch
   carries the baseline, so the routed branch may go smoothly to zero everywhere
   without the energy collapsing.  The shared branch is load-bearing here, not
   an optimisation detail.
3. **No max operator.**  A normalised variant ``w_e ~ f((s_max - s_e) / tau)``
   reintroduces ``s_max``, which is not differentiable at a tie, so the kink
   returns through the back door.  The unnormalised form avoids the question.

The price of all this, stated plainly
-------------------------------------

The conditions ``f'(0) = f'(1) = 0`` that buy the ``C^k`` contract also make the
routing gradient **exactly zero outside the band** ``0 < u_e < 1``.  Measured,
not argued: with every atom saturated on, or every atom switched off, the
gradient reaching ``score_projection`` and ``theta`` is ``0.0``.

Three consequences follow, and they are properties of compact support rather
than defects to be fixed -- any switch with a genuine zero has them:

* An expert that is off for *every* atom is **permanently dead** under gradient
  descent.  Its own weights get no gradient because ``w_e = 0`` multiplies
  them, and its score gets none either because ``f'(1) = 0``.  Nothing in the
  loss can bring it back.
* An expert that is on for every atom teaches the router nothing: the routing
  is saturated and ``f'(0) = 0``.
* Only (atom, expert) pairs with ``0 < w_e < 1`` carry routing gradient at all.
  :meth:`RoutedScalarFFN.routing_statistics` reports that as
  ``transition_fraction``, and it is the number to watch -- a run where it sits
  at zero has a router that cannot learn, whatever the loss curve does.

This is why the load-balancing bias is **load-bearing rather than an
optimisation**.  It is updated outside the gradient, so it is the only
mechanism here that can move a saturated or dead expert back into the band.
The default threshold is chosen to start every atom at the band centre for the
same reason; see :class:`CompactSupportRouter`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Switching polynomials, in the factored form that makes the multiplicity of the
# root at x = 1 manifest and that evaluates to an exact floating-point zero
# there.  Both are the minimal-degree polynomial meeting their contract.
#
#   contract | f(x)                                             | root at x = 1
#   ---------|--------------------------------------------------|--------------
#   C^2      | (1-x)^3 (1 + 3x + 6x^2)                          | multiplicity 3
#   C^4      | (1-x)^5 (1 + 5x + 15x^2 + 35x^3 + 70x^4)         | multiplicity 5
#
# Expanded, these are 1 - 10x^3 + 15x^4 - 6x^5 and
# 1 - 126x^5 + 420x^6 - 540x^7 + 315x^8 - 70x^9.  ``tests/test_routing.py``
# checks both expansions with exact integer arithmetic rather than trusting the
# transcription.
#
# The C^2 row is the same polynomial as ``physics.SmoothPolynomialCutoff``; the
# radial envelope and the router switch off for the same reason and to the same
# order.  The C^4 row is monotone on [0, 1] because f'(x) = -630 x^4 (x-1)^4,
# which is nonpositive there.
_SWITCH_ORDERS = {"c2": 2, "c4": 4}


def switch_polynomial(x: torch.Tensor, contract: str = "c2") -> torch.Tensor:
    """Evaluate the switching polynomial ``f`` on an already-clamped ``x``.

    ``x`` must lie in ``[0, 1]``; :class:`CompactSupportRouter` clamps before
    calling.  ``f(0) = 1`` (expert fully on) and ``f(1) = 0`` (expert fully off,
    exactly).
    """

    if contract == "c2":
        return (1.0 - x).pow(3) * (1.0 + 3.0 * x + 6.0 * x.square())
    if contract == "c4":
        return (1.0 - x).pow(5) * (
            1.0
            + 5.0 * x
            + 15.0 * x.square()
            + 35.0 * x.pow(3)
            + 70.0 * x.pow(4)
        )
    raise ValueError("contract must be 'c2' or 'c4'")


def switch_contract_order(contract: str) -> int:
    """Number of derivatives of the switch that vanish at both joins."""

    try:
        return _SWITCH_ORDERS[str(contract).lower()]
    except KeyError as exception:
        raise ValueError("contract must be 'c2' or 'c4'") from exception


def resolve_switch_contract(contract: str, shell_degree: int) -> str:
    """Pick the switch that does not downgrade the model's smoothness order.

    A quintic shell tokenizer (``shell_degree = 5``) delivers a ``C^4`` energy.
    Routing it with the ``C^2`` switch would silently make the whole model
    ``C^2`` again, which is the difference between having and not having third
    order force constants.  ``'auto'`` therefore follows the tokenizer.
    """

    contract = str(contract).lower()
    if contract == "auto":
        return "c4" if int(shell_degree) >= 5 else "c2"
    if contract not in _SWITCH_ORDERS:
        raise ValueError("router_switch must be 'auto', 'c2', or 'c4'")
    return contract


class CompactSupportRouter(nn.Module):
    """Smooth, unnormalised, compactly supported expert weights.

    Emits ``w`` of shape ``(atoms, num_experts)`` with ``w in [0, 1]``, equal to
    one where ``s_e >= theta_e + b_e`` and exactly zero where
    ``s_e <= theta_e + b_e - tau``, for the load-balancing bias ``b_e`` defined
    below (zero unless ``balance_rate > 0``).
    """

    def __init__(
        self,
        context_dim: int,
        num_experts: int,
        tau: float = 1.0,
        contract: str = "c2",
        threshold_init: float | None = None,
        balance_rate: float = 0.0,
        balance_target: float | None = None,
    ):
        super().__init__()
        if int(context_dim) < 1:
            raise ValueError("context_dim must be positive")
        if int(num_experts) < 1:
            raise ValueError("num_experts must be positive")
        if float(tau) <= 0.0:
            raise ValueError("router tau must be positive")
        if float(balance_rate) < 0.0:
            raise ValueError("balance_rate must be nonnegative")
        if balance_target is not None and not 0.0 <= float(balance_target) <= 1.0:
            raise ValueError("balance_target must lie in [0, 1]")
        self.context_dim = int(context_dim)
        self.num_experts = int(num_experts)
        self.tau = float(tau)
        # Deliberately strict: 'auto' has to be resolved against the tokenizer's
        # shell degree, which the router cannot see.  Silently defaulting it here
        # would be the exact failure the resolution rule exists to prevent.
        contract = str(contract).lower()
        if contract not in _SWITCH_ORDERS:
            raise ValueError(
                "CompactSupportRouter needs an explicit 'c2' or 'c4' contract; "
                "call resolve_switch_contract() to turn 'auto' into one"
            )
        self.contract = contract
        # ---- where to start the threshold ------------------------------------
        # The routing gradient is nonzero only for 0 < u < 1, and both switches
        # put the maximum of |f'| at the band centre u = 1/2:
        #
        #     C^2 : f'(u) = -30 u^2 (1-u)^2      C^4 : f'(u) = -630 u^4 (u-1)^4
        #
        # both symmetric about 1/2.  Both also satisfy f(1/2) = 1/2 exactly.
        # Scores start at mean zero -- the bias is zeroed below, the weight is
        # zero-mean, and the context is layer-normalised -- so
        #
        #     u = (theta - s)/tau = 1/2   at   theta = tau/2 ,
        #
        # which centres the band on the score distribution.  Every expert then
        # starts half on, with the largest routing gradient available, and can
        # move in either direction.
        #
        # The obvious-looking ``threshold_init=0.0`` is a trap: it puts every
        # atom with a positive score at u <= 0, where f'(0) = 0 gives the router
        # no gradient at all, and leaves any expert whose scores all fall below
        # -tau permanently dead.  Pass an explicit float to override.
        if threshold_init is None:
            threshold_init = 0.5 * self.tau
        self.score_projection = nn.Linear(self.context_dim, self.num_experts)
        self.threshold = nn.Parameter(
            torch.full((self.num_experts,), float(threshold_init))
        )
        # ``theta`` is a location parameter -- it says *where* the switch turns
        # over, in the units of the score -- so it is the same kind of object as
        # the bias of ``score_projection``, which the optimizer already exempts.
        # Decaying it does not regularise anything; it drags every threshold
        # toward zero, which is itself a routing preference.
        self.threshold._no_weight_decay = True
        # Zeroed so the score distribution starts centred, which is what makes
        # theta = tau/2 the band centre above.
        nn.init.zeros_(self.score_projection.bias)

        # ---- auxiliary-loss-free load balancing ------------------------------
        # LatentMoE trains with a load-balancing loss (coefficient 1e-4) *plus*
        # DeepSeek's auxiliary-loss-free strategy: a per-expert bias added to the
        # routing score for selection only, updated by a sign rule outside the
        # gradient.
        #
        # For an interatomic potential the loss-free half is not merely
        # equivalent, it is *strictly preferable*, and for a reason that has no
        # analogue in a language model.  There, an auxiliary term trades off
        # against perplexity, which is itself only a proxy.  Here the training
        # loss defines the potential energy surface: adding lambda * L_balance
        # means the fitted forces no longer minimise the force error alone, so a
        # non-physical term has biased a physical observable.  A bias updated
        # outside the gradient leaves the objective exactly as it was.
        #
        # It is also free of smoothness cost, which is the property that matters
        # most here.  ``balance_bias`` is a buffer, not a parameter: it takes no
        # gradient, it is constant within any forward pass, and it is frozen in
        # eval.  It therefore shifts ``theta`` and nothing else, and
        #
        #     u_e = (theta_e + b_e - s_e) / tau
        #
        # is exactly as smooth in the atomic positions as it was before -- the
        # C^k contract of the switch is untouched.  Load balancing here costs
        # nothing in smoothness, which is not true of any scheme that changes the
        # routing *function* to spread load.
        self.balance_rate = float(balance_rate)
        self.balance_target = None if balance_target is None else float(balance_target)
        self.register_buffer("balance_bias", torch.zeros(self.num_experts))

    def scores(self, context: torch.Tensor) -> torch.Tensor:
        """``s_e(h_i)``, shape ``(atoms, num_experts)``."""

        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(f"context must have shape (atoms, {self.context_dim})")
        return self.score_projection(context)

    def weights_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """``w_e = f(clamp((theta_e + b_e - s_e) / tau, 0, 1))``."""

        u = (self.threshold + self.balance_bias - scores) / self.tau
        return switch_polynomial(u.clamp(min=0.0, max=1.0), self.contract)

    @torch.no_grad()
    def update_balance(self, weights: torch.Tensor) -> None:
        """Sign-rule update of the load-balancing bias. Training only.

        With ``rho_e`` the fraction of atoms for which expert ``e`` is active,

            b_e <- b_e + gamma * sign(rho_e - rho*) ,

        so an overloaded expert has its threshold raised and sheds atoms.  The
        compact support makes ``rho_e`` an *exact* count rather than a soft
        proxy: ``w_e > 0`` is a genuine indicator, not a threshold on small
        values.

        ``rho*`` defaults to the mean occupancy over experts, which balances load
        without imposing a target sparsity; set ``balance_target`` to drive the
        occupancy towards a chosen level instead.
        """

        if self.balance_rate <= 0.0:
            return
        occupancy = (weights > 0.0).to(weights.dtype).mean(dim=0)
        target = (
            occupancy.mean() if self.balance_target is None else self.balance_target
        )
        self.balance_bias += self.balance_rate * torch.sign(occupancy - target)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        weights = self.weights_from_scores(self.scores(context))
        if self.training:
            # Applied after the weights this pass uses, so the bias is constant
            # throughout the forward and the returned forces remain the exact
            # gradient of the returned energy.
            self.update_balance(weights.detach())
        return weights


def routing_capacity(num_experts: int, context_dim: int) -> int:
    """Greatest number of distinct active expert sets this router can realise.

    The active set of atom ``i`` is determined by the signs of the ``N`` affine
    functions ``s_e(h_i) - (theta_e + b_e - tau)``, so the reachable sets are the
    cells of an arrangement of ``N`` hyperplanes in ``R^d``.  Distinct cells carry
    distinct sign vectors, so this is an equality in general position and an
    upper bound otherwise -- degenerate weights realise fewer.  By Schlafli's
    formula an arrangement in general position has

        C(N, d) = sum_{j=0}^{min(d, N)} binom(N, j)

    cells, which is ``2^N`` when ``N <= d`` and grows only polynomially, like
    ``O(N^d)``, once ``N > d``.

    **This is why the router must not be compressed.**  LatentMoE's Design
    Principle V argues for raising the expert count ``N``, and its Principle IV
    argues for shrinking the width ``d``; done to the *router* those two pull
    against each other, because the capacity above saturates as soon as
    ``N > d``.  Nemotron 3 Super states the resolution as a design choice --
    "all non-routed computations, including the routing gate (gating network),
    shared expert computation, and non-expert layers, remain in the full hidden
    dimension d" -- without deriving it.  The formula above is the derivation:
    compress the experts, never the router.  :class:`RoutedScalarFFN` enforces
    it structurally, by giving the router the uncompressed context.
    """

    num_experts = int(num_experts)
    context_dim = int(context_dim)
    if num_experts < 1 or context_dim < 1:
        raise ValueError("num_experts and context_dim must be positive")
    return sum(math.comb(num_experts, j) for j in range(min(context_dim, num_experts) + 1))


class _ScalarExpert(nn.Module):
    """One expert: the same shape of map as the block's own scalar FFN."""

    def __init__(self, context_dim: int, hidden: int, out_dim: int, swiglu: bool):
        super().__init__()
        self.swiglu = bool(swiglu)
        if self.swiglu:
            self.input_projection = nn.Linear(context_dim, 2 * hidden)
        else:
            self.input_projection = nn.Linear(context_dim, hidden)
        self.output_projection = nn.Linear(hidden, out_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if self.swiglu:
            gate, value = self.input_projection(context).chunk(2, dim=-1)
            return self.output_projection(F.silu(gate) * value)
        return self.output_projection(F.silu(self.input_projection(context)))


class RoutedScalarFFN(nn.Module):
    """``sum_e w_e Expert_e(h_i)``: the routed branch only.

    The shared branch stays where it already is, in the owning block, so that
    ``num_experts = 0`` reproduces the unrouted arithmetic exactly rather than
    approximately.  This module contributes a residual that is initialised small
    -- by ``expert_scale``, following the ``layer_scale`` convention used
    everywhere else in the block -- and is identically zero once every expert has
    switched off.

    ``backend`` selects how the sum is evaluated:

    ``"dense"``   evaluate every expert on every atom and weight the results.
                  Scriptable, and the reference for correctness.
    ``"sparse"``  evaluate ``Expert_e`` only on the atoms with ``w_e > 0``.
                  This is where the compact support is actually cashed in.

    The outputs agree to floating-point round-off, and so do the gradients
    wherever both exist -- measured at 2e-19 in FP64.  They are not bitwise
    identical because the sparse path contracts over a different set of rows.

    **One difference is not round-off, and it is a training difference.**  For
    an expert that is inactive on *every* atom in the batch, the dense path
    still evaluates it and produces an exact zero gradient, while the sparse
    path skips it and leaves ``grad`` as ``None``.  Torch optimizers skip
    ``p.grad is None`` entirely, decoupled weight decay included, so a fully
    dead expert is decayed toward zero under ``dense`` and frozen under
    ``sparse``.  Neither is wrong -- ``sparse`` arguably has the better claim,
    since a skipped expert should not move -- but the two are not
    interchangeable mid-run, and an expert whose weights have been decayed away
    contributes nothing even if the router later revives it.
    """

    def __init__(
        self,
        context_dim: int,
        out_dim: int,
        num_experts: int,
        expert_hidden: int,
        swiglu: bool = True,
        tau: float = 1.0,
        contract: str = "c2",
        threshold_init: float | None = None,
        expert_scale_init: float | None = 1.0e-2,
        backend: str = "dense",
        latent_dim: int | None = None,
        balance_rate: float = 0.0,
        balance_target: float | None = None,
    ):
        super().__init__()
        if int(num_experts) < 1:
            raise ValueError("num_experts must be positive")
        if int(expert_hidden) < 1:
            raise ValueError("expert_hidden must be positive")
        backend = str(backend).lower()
        if backend not in {"dense", "sparse"}:
            raise ValueError("routing_backend must be 'dense' or 'sparse'")
        self.backend = backend
        self.num_experts = int(num_experts)
        self.context_dim = int(context_dim)
        self.out_dim = int(out_dim)
        # ---- latent expert compression ---------------------------------------
        # LatentMoE runs the routed experts in a compressed space and lifts back,
        #
        #     l-MoE(x) = W_up ( sum_e p'_e E_e(W_down x ; l) ) + Shared(x ; d) ,
        #     alpha = d / l ,
        #
        # keeping the expert intermediate width ``m`` fixed so the nonlinear
        # budget ``U_eff ~ K m`` of Design Principle III is preserved.  Measured
        # there: quality holds to ``alpha <= 4`` and degrades at ``alpha = 8``.
        #
        # Both projections are linear, hence C-infinity, so this cannot touch the
        # derivative-order contract; and the context is already O(3) invariant,
        # so it cannot touch equivariance either.  It is a pure capacity/cost
        # trade with no physics risk.
        #
        # **The router is deliberately not compressed.**  See
        # :func:`routing_capacity`: the number of realisable active sets is the
        # cell count of an arrangement of N hyperplanes in the router's input
        # dimension, which saturates once N exceeds that dimension.  Compressing
        # the router would cap exactly the combinatorial diversity that Design
        # Principle V is trying to buy.
        if latent_dim is not None:
            latent_dim = int(latent_dim)
            if latent_dim < 1:
                raise ValueError("latent_dim must be positive")
            if latent_dim > self.context_dim:
                raise ValueError(
                    f"latent_dim={latent_dim} exceeds context_dim="
                    f"{self.context_dim}; alpha = context_dim / latent_dim below "
                    "one is an expansion, not a compression.  The context width "
                    "is the block's invariant dimension, which is derived from "
                    "hidden_dim and the irreps rather than set directly"
                )
        self.latent_dim = latent_dim
        expert_in = self.context_dim if latent_dim is None else latent_dim
        expert_out = self.out_dim if latent_dim is None else latent_dim
        self.latent_down = (
            None if latent_dim is None
            else nn.Linear(self.context_dim, latent_dim, bias=False)
        )
        self.latent_up = (
            None if latent_dim is None
            else nn.Linear(latent_dim, self.out_dim, bias=False)
        )
        self.router = CompactSupportRouter(
            context_dim,
            num_experts,
            tau=tau,
            contract=contract,
            threshold_init=threshold_init,
            balance_rate=balance_rate,
            balance_target=balance_target,
        )
        self.experts = nn.ModuleList(
            [
                _ScalarExpert(expert_in, int(expert_hidden), expert_out, swiglu)
                for _ in range(self.num_experts)
            ]
        )
        self.expert_scale = (
            nn.Parameter(torch.full((int(out_dim),), float(expert_scale_init)))
            if expert_scale_init is not None
            else None
        )
        if self.expert_scale is not None:
            # The exact analogue of ``layer_scale`` and ``ffn_scale``, both of
            # which the optimizer exempts by name.  It starts at 1e-2, so leaving
            # it decayed would have weight decay steadily switching the routed
            # branch off -- suppressing the mechanism rather than regularising it.
            self.expert_scale._no_weight_decay = True

    @property
    def compression_ratio(self) -> float:
        """``alpha = d / l``; 1.0 when the experts are uncompressed."""

        return 1.0 if self.latent_dim is None else self.context_dim / self.latent_dim

    def routing_capacity(self) -> int:
        """Distinct active expert sets this layer's router can realise."""

        return routing_capacity(self.num_experts, self.router.context_dim)

    def _dense(
        self, context: torch.Tensor, weights: torch.Tensor, width: int
    ) -> torch.Tensor:
        total = context.new_zeros((context.shape[0], width))
        for index, expert in enumerate(self.experts):
            total = total + weights[:, index : index + 1] * expert(context)
        return total

    def _sparse(
        self, context: torch.Tensor, weights: torch.Tensor, width: int
    ) -> torch.Tensor:
        total = context.new_zeros((context.shape[0], width))
        for index, expert in enumerate(self.experts):
            column = weights[:, index]
            # Exactly zero, not merely small: the factored switch returns 0.0 for
            # every score at or below theta - tau, so this mask is the true
            # support of the expert and dropping those rows changes nothing.
            active = torch.nonzero(column, as_tuple=False).squeeze(-1)
            if active.numel() == 0:
                continue
            selected = context.index_select(0, active)
            update = column.index_select(0, active).unsqueeze(-1) * expert(selected)
            total = total.index_add(0, active, update)
        return total

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # The router reads the *uncompressed* context; only the experts see the
        # latent.  routing_capacity() explains why that asymmetry is required.
        weights = self.router(context)
        if self.latent_down is None:
            expert_input = context
            width = self.out_dim
        else:
            expert_input = self.latent_down(context)
            width = int(self.latent_dim)
        # The sparse path must be disabled for BOTH scripting and tracing.
        # ``torch.jit.is_scripting()`` is False under ``torch.jit.trace``, so
        # guarding on it alone let the tracer record ``torch.nonzero`` output as
        # a *constant* index set: the exported model would then reuse the tracing
        # example's expert-activity pattern for every future input.  That failure
        # is invisible whenever the tracing example happens to activate every
        # expert, which is exactly the case at default thresholds.
        # The two backends agree to 1e-13, so falling back is free.
        if self.backend == "sparse" and not (
            torch.jit.is_scripting() or torch.jit.is_tracing()
        ):
            update = self._sparse(expert_input, weights, width)
        else:
            update = self._dense(expert_input, weights, width)
        if self.latent_up is not None:
            update = self.latent_up(update)
        if self.expert_scale is not None:
            update = update * self.expert_scale
        return update

    @torch.no_grad()
    def routing_statistics(self, context: torch.Tensor) -> dict[str, float]:
        """Occupancy of the routed branch, for monitoring during training.

        ``active_fraction`` is the fraction of (atom, expert) pairs with
        ``w_e > 0``; it is the quantity the sparse backend actually saves
        against.  ``saturated_fraction`` counts the pairs at ``w_e = 1``, where
        the expert is fully on.  A run in which ``active_fraction`` never falls
        below one is dense in all but name.

        ``transition_fraction`` is the one to watch.  It counts the pairs with
        ``0 < w_e < 1``, which are **exactly** the pairs carrying routing
        gradient: ``f'`` is strictly negative inside the band and identically
        zero at and beyond both joins.  If it reaches zero the router has
        stopped learning, no matter what the loss curve is doing, and only the
        load-balancing bias can move it again.

        ``dead_experts`` counts experts inactive on every atom.  Under gradient
        descent alone those are unrecoverable -- see the module docstring.
        """

        # Deliberately not self.router(...): that would fire the load-balancing
        # update, so merely observing the model would change it.
        weights = self.router.weights_from_scores(self.router.scores(context))
        total = float(weights.numel())
        occupancy = (weights > 0.0).to(weights.dtype).mean(dim=0)
        return {
            "active_fraction": float((weights > 0.0).sum()) / total,
            "saturated_fraction": float((weights >= 1.0).sum()) / total,
            "transition_fraction": float(
                ((weights > 0.0) & (weights < 1.0)).sum()
            ) / total,
            "dead_experts": float((occupancy == 0.0).sum()),
            "weight_mean": float(weights.mean()),
            "experts_per_atom": float((weights > 0.0).sum(dim=-1).to(weights).mean()),
            # Spread of per-expert occupancy: 0 is perfectly balanced load, and
            # a large value means a few experts are carrying the model.
            "occupancy_spread": float(occupancy.max() - occupancy.min()),
            "balance_bias_spread": float(
                self.router.balance_bias.max() - self.router.balance_bias.min()
            ),
        }
