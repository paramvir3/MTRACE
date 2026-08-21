"""Smooth compact-support expert routing: the switch, the sparsity, the physics."""

import unittest

import torch

from mtace.model import MambaACEV2
from mtace.routing import (
    CompactSupportRouter,
    RoutedScalarFFN,
    resolve_switch_contract,
    switch_contract_order,
    switch_polynomial,
)


def _convolve(left, right):
    """Exact integer polynomial product, ascending coefficient order."""

    product = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            product[i + j] += a * b
    return product


def data(positions, species=None):
    count = positions.shape[0]
    senders, receivers = [], []
    for receiver in range(count):
        for sender in range(count):
            if sender != receiver:
                senders.append(sender)
                receivers.append(receiver)
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    return {
        "z": torch.tensor(species or [8, 1, 1, 8][:count], dtype=torch.long),
        "pos": positions,
        "cell": torch.eye(3, dtype=positions.dtype) * 9.0,
        "edge_index": edge_index,
        "edge_shift": torch.zeros((edge_index.shape[1], 3), dtype=positions.dtype),
        "volume": torch.tensor(729.0, dtype=positions.dtype),
    }


POSITIONS = torch.tensor(
    [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]],
    dtype=torch.float64,
)


def routed_model(seed=17, num_experts=1, tau=1.0, backend="dense", **overrides):
    torch.manual_seed(seed)
    settings = dict(
        r_max=4.5,
        l_max=2,
        num_radial=4,
        hidden_dim=8,
        num_layers=1,
        correlation_order=4,
        correlation_channels=4,
        mamba_dim=12,
        mamba_d_state=4,
        mamba_backend="torch",
        readout_hidden=8,
        num_experts=num_experts,
        expert_hidden=8,
        router_tau=tau,
        routing_backend=backend,
    )
    settings.update(overrides)
    return MambaACEV2(**settings).double().eval()


def router_scores(network, structure):
    """``s_e(h_i)`` for layer 0, through the exact path ``forward`` uses."""

    layer = network.layers[0]
    edge_vec = structure["pos"][structure["edge_index"][0]] - structure["pos"][
        structure["edge_index"][1]
    ]
    edge_len = torch.linalg.vector_norm(edge_vec, dim=-1)
    features, tokens, kind, coordinate = network.ace(
        network.species_embedding(structure["z"]), structure["edge_index"], edge_vec, edge_len
    )
    mixed = layer._mixed_features(features, tokens, kind, coordinate)
    invariants = layer.scalar_norm(layer._node_invariants(mixed))
    return layer.routed_ffn.router.scores(invariants)


class SwitchPolynomialTests(unittest.TestCase):
    """The two admissible switches, checked rather than transcribed."""

    def test_factored_forms_expand_to_the_stated_polynomials(self):
        # C^2: (1-x)^3 (1 + 3x + 6x^2) = 1 - 10x^3 + 15x^4 - 6x^5
        self.assertEqual(
            _convolve([1, -3, 3, -1], [1, 3, 6]),
            [1, 0, 0, -10, 15, -6],
        )
        # C^4: (1-x)^5 (1 + 5x + 15x^2 + 35x^3 + 70x^4)
        #      = 1 - 126x^5 + 420x^6 - 540x^7 + 315x^8 - 70x^9
        self.assertEqual(
            _convolve([1, -5, 10, -10, 5, -1], [1, 5, 15, 35, 70]),
            [1, 0, 0, 0, 0, -126, 420, -540, 315, -70],
        )

    def test_boundary_values_and_vanishing_derivatives(self):
        for contract in ("c2", "c4"):
            order = switch_contract_order(contract)
            for point, expected in ((0.0, 1.0), (1.0, 0.0)):
                x = torch.tensor(point, dtype=torch.float64, requires_grad=True)
                value = switch_polynomial(x, contract)
                self.assertAlmostEqual(float(value), expected, places=14)
                derivative = value
                for _ in range(order):
                    derivative = torch.autograd.grad(
                        derivative, x, create_graph=True
                    )[0]
                    self.assertLess(
                        abs(float(derivative)),
                        1.0e-12,
                        f"{contract} derivative fails to vanish at x={point}",
                    )

    def test_c4_switch_is_strictly_smoother_than_the_c2_switch(self):
        # f'''(1) = -60 for the C^2 switch: it is C^2 at the join and no better.
        x = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        third = switch_polynomial(x, "c2")
        for _ in range(3):
            third = torch.autograd.grad(third, x, create_graph=True)[0]
        self.assertAlmostEqual(float(third), -60.0, places=10)

    def test_c4_switch_is_monotone_because_its_derivative_factors(self):
        x = torch.linspace(0.0, 1.0, 401, dtype=torch.float64, requires_grad=True)
        value = switch_polynomial(x, "c4")
        gradient = torch.autograd.grad(value.sum(), x)[0]
        # f'(x) = -630 x^4 (x - 1)^4, nonpositive on [0, 1].
        expected = -630.0 * x.detach().pow(4) * (x.detach() - 1.0).pow(4)
        torch.testing.assert_close(gradient, expected, atol=1.0e-11, rtol=1.0e-11)
        self.assertLessEqual(float(gradient.max()), 0.0)

    def test_contract_resolution_follows_the_shell_degree(self):
        self.assertEqual(resolve_switch_contract("auto", 3), "c2")
        self.assertEqual(resolve_switch_contract("auto", 5), "c4")
        self.assertEqual(resolve_switch_contract("c2", 5), "c2")
        with self.assertRaises(ValueError):
            resolve_switch_contract("c3", 3)
        # A quintic tokenizer must not be silently downgraded to C^2.
        network = MambaACEV2(
            hidden_dim=8, num_layers=1, num_shells=6, num_radial=4,
            correlation_channels=4, mamba_dim=12, readout_hidden=8,
            shell_degree=5, num_experts=2, mamba_backend="torch",
        )
        self.assertEqual(network.layers[0].routed_ffn.router.contract, "c4")

    def test_router_refuses_an_unresolved_contract(self):
        with self.assertRaisesRegex(ValueError, "explicit 'c2' or 'c4'"):
            CompactSupportRouter(4, 2, contract="auto")


class CompactSupportTests(unittest.TestCase):
    def test_support_is_exactly_compact(self):
        router = CompactSupportRouter(3, 2, tau=0.5, contract="c2").double()
        with torch.no_grad():
            router.threshold.fill_(0.0)
        # w = 1 for s >= theta, exactly 0 for s <= theta - tau.
        scores = torch.tensor(
            [[0.7, 0.0], [-0.5, -0.25], [-1.0, -0.5000001]], dtype=torch.float64
        )
        weights = router.weights_from_scores(scores)
        self.assertEqual(float(weights[0, 0]), 1.0)
        self.assertEqual(float(weights[0, 1]), 1.0)
        # s = -tau exactly: the far join, and a true floating-point zero.
        self.assertEqual(float(weights[1, 0]), 0.0)
        self.assertEqual(float(weights[2, 0]), 0.0)
        self.assertEqual(float(weights[2, 1]), 0.0)
        # Strictly inside the transition, so strictly between the two joins.
        self.assertGreater(float(weights[1, 1]), 0.0)
        self.assertLess(float(weights[1, 1]), 1.0)

    def test_weights_are_unnormalised_so_no_denominator_can_vanish(self):
        router = CompactSupportRouter(3, 4, tau=0.5, contract="c2").double()
        with torch.no_grad():
            router.threshold.fill_(0.0)
        # Every expert off at once is a legal state; a normalised router would
        # divide by zero here.
        weights = router.weights_from_scores(
            torch.full((5, 4), -10.0, dtype=torch.float64)
        )
        self.assertEqual(float(weights.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(weights).all())

    def test_dense_and_sparse_backends_agree(self):
        torch.manual_seed(3)
        dense = RoutedScalarFFN(
            context_dim=6, out_dim=5, num_experts=4, expert_hidden=7,
            tau=0.5, contract="c2", backend="dense",
        ).double()
        sparse = RoutedScalarFFN(
            context_dim=6, out_dim=5, num_experts=4, expert_hidden=7,
            tau=0.5, contract="c2", backend="sparse",
        ).double()
        sparse.load_state_dict(dense.state_dict())
        with torch.no_grad():
            # Spread the thresholds so some experts are genuinely switched off.
            dense.router.threshold.copy_(
                torch.tensor([-2.0, 0.0, 0.5, 2.0], dtype=torch.float64)
            )
            sparse.router.threshold.copy_(dense.router.threshold)
        context = torch.randn(32, 6, dtype=torch.float64)
        weights = dense.router(context)
        self.assertGreater(float((weights == 0.0).sum()), 0.0, "no expert switched off")
        self.assertGreater(float((weights > 0.0).sum()), 0.0, "every expert switched off")
        torch.testing.assert_close(
            sparse(context), dense(context), atol=1.0e-13, rtol=1.0e-13
        )

    def test_a_fully_dead_expert_gets_no_gradient_under_the_sparse_backend(self):
        """The one dense/sparse difference that is not round-off.

        An expert inactive on *every* atom is still evaluated by the dense path,
        which produces an exact zero gradient, but is skipped entirely by the
        sparse path, which leaves ``grad`` as ``None``.  Torch optimizers skip
        ``p.grad is None``, decoupled weight decay included, so such an expert
        is decayed toward zero under ``dense`` and frozen under ``sparse``.
        Pinned here because it makes the backends non-interchangeable mid-run.
        """

        torch.manual_seed(5)
        settings = dict(
            context_dim=16, out_dim=6, num_experts=4, expert_hidden=8, tau=0.4
        )
        dense = RoutedScalarFFN(backend="dense", **settings).double()
        sparse = RoutedScalarFFN(backend="sparse", **settings).double()
        sparse.load_state_dict(dense.state_dict())
        with torch.no_grad():
            # Expert 3's threshold is far above any score: dead on every atom.
            thresholds = torch.tensor([-2.0, 0.0, 0.5, 40.0], dtype=torch.float64)
            dense.router.threshold.copy_(thresholds)
            sparse.router.threshold.copy_(thresholds)
        context = torch.randn(40, 16, dtype=torch.float64)
        weights = dense.router.weights_from_scores(dense.router.scores(context))
        self.assertEqual(int((weights[:, 3] > 0).sum()), 0, "expert 3 is not dead")
        self.assertGreater(int((weights[:, 0] > 0).sum()), 0)

        # The outputs are still identical: a dead expert contributes nothing.
        torch.testing.assert_close(
            sparse(context), dense(context), atol=1.0e-13, rtol=1.0e-13
        )
        for module in (dense, sparse):
            module.zero_grad()
            module(context).square().sum().backward()
        self.assertIsNotNone(dense.experts[3].output_projection.weight.grad)
        self.assertEqual(
            float(dense.experts[3].output_projection.weight.grad.abs().max()), 0.0
        )
        self.assertIsNone(sparse.experts[3].output_projection.weight.grad)
        # Every live expert still agrees to round-off.
        for index in (0, 1, 2):
            for left, right in zip(
                dense.experts[index].parameters(), sparse.experts[index].parameters()
            ):
                torch.testing.assert_close(
                    right.grad, left.grad, atol=1.0e-13, rtol=1.0e-13
                )

    def test_sparse_backend_gradients_match_the_dense_backend(self):
        """Agreement when every expert is live on at least one atom.

        That precondition is asserted rather than assumed: without it the
        comparison would hit the ``None``-versus-zero case pinned above.
        """

        torch.manual_seed(5)
        dense = RoutedScalarFFN(
            context_dim=6, out_dim=4, num_experts=3, expert_hidden=5,
            tau=0.4, contract="c2", backend="dense",
        ).double()
        sparse = RoutedScalarFFN(
            context_dim=6, out_dim=4, num_experts=3, expert_hidden=5,
            tau=0.4, contract="c2", backend="sparse",
        ).double()
        sparse.load_state_dict(dense.state_dict())
        with torch.no_grad():
            dense.router.threshold.copy_(
                torch.tensor([-1.0, 0.2, 1.0], dtype=torch.float64)
            )
            sparse.router.threshold.copy_(dense.router.threshold)
        context = torch.randn(24, 6, dtype=torch.float64)
        weights = dense.router.weights_from_scores(dense.router.scores(context))
        for index in range(3):
            self.assertGreater(
                int((weights[:, index] > 0).sum()), 0,
                f"expert {index} is dead, so this test would compare None to zero",
            )
        for module in (dense, sparse):
            module.zero_grad()
            module(context).square().sum().backward()
        for (name, left), (_, right) in zip(
            dense.named_parameters(), sparse.named_parameters()
        ):
            torch.testing.assert_close(
                right.grad, left.grad, atol=1.0e-12, rtol=1.0e-12,
                msg=f"gradient mismatch for {name}",
            )

    def test_routing_is_inert_when_no_experts_are_requested(self):
        baseline = routed_model(num_experts=0)
        self.assertIsNone(baseline.layers[0].routed_ffn)
        routed = routed_model(num_experts=2)
        # The routed model carries strictly more parameters, and every parameter
        # the baseline has survives unchanged in name.
        baseline_names = set(dict(baseline.named_parameters()))
        routed_names = set(dict(routed.named_parameters()))
        self.assertTrue(baseline_names < routed_names)
        self.assertEqual(baseline.routing_occupancy(**_arguments(POSITIONS)), [])


def _arguments(positions):
    structure = data(positions)
    return {
        "z": structure["z"],
        "pos": structure["pos"],
        "cell": structure["cell"],
        "edge_index": structure["edge_index"],
        "edge_shift": structure["edge_shift"],
    }


class RoutedModelPhysicsTests(unittest.TestCase):
    def test_symmetry_survives_routing(self):
        network = routed_model(num_experts=3, tau=0.5)
        energy, forces, _, _ = network(data(POSITIONS), compute_stress=False)
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
        rotated = network(data(POSITIONS @ rotation.T), compute_stress=False)
        torch.testing.assert_close(rotated[0], energy, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(
            rotated[1], forces @ rotation.T, atol=1.0e-10, rtol=1.0e-10
        )
        inverted = network(data(-POSITIONS), compute_stress=False)
        torch.testing.assert_close(inverted[0], energy, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(inverted[1], -forces, atol=1.0e-10, rtol=1.0e-10)
        translated = network(
            data(POSITIONS + torch.tensor([2.3, -1.1, 0.4], dtype=torch.float64)),
            compute_stress=False,
        )
        torch.testing.assert_close(translated[0], energy, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(translated[1], forces, atol=1.0e-10, rtol=1.0e-10)

    def test_forces_are_the_energy_gradient(self):
        network = routed_model(num_experts=3, tau=0.5)
        _, forces, _, _ = network(data(POSITIONS), compute_stress=False)
        step = 1.0e-6
        for atom in range(POSITIONS.shape[0]):
            for axis in range(3):
                plus = POSITIONS.clone()
                minus = POSITIONS.clone()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                derivative = (
                    network(data(plus), compute_stress=False)[0]
                    - network(data(minus), compute_stress=False)[0]
                ) / (2.0 * step)
                self.assertAlmostEqual(
                    float(-derivative), float(forces[atom, axis]), places=6
                )

    def test_routing_occupancy_reports_real_sparsity(self):
        network = routed_model(num_experts=4, tau=0.5)
        with torch.no_grad():
            network.layers[0].routed_ffn.router.threshold.copy_(
                torch.tensor([-5.0, 0.0, 0.0, 5.0], dtype=torch.float64)
            )
        report = network.routing_occupancy(**_arguments(POSITIONS))
        self.assertEqual(len(report), 1)
        # Expert 3 has a threshold far above any score, so it is off everywhere;
        # expert 0 is on everywhere.  Occupancy must sit strictly inside (0, 1).
        self.assertGreater(report[0]["active_fraction"], 0.0)
        self.assertLess(report[0]["active_fraction"], 1.0)


class RoutingBoundarySmoothnessTests(unittest.TestCase):
    """The falsification test: is the energy C^2 across a routing boundary?

    A trajectory is constructed that *provably* crosses ``s_e = theta``, by
    setting the threshold to the score the model itself produces at the midpoint.
    Both joins of the switch are exercised: ``u = 0``, where the expert leaves
    saturation, and ``u = 1``, where it switches off entirely.
    """

    DIRECTION = torch.tensor([0.37, -0.51, 0.62], dtype=torch.float64)

    def _energy_and_slope(self, network, t):
        positions = POSITIONS.clone()
        positions[0] = POSITIONS[0] + t * self.DIRECTION
        energy, forces, _, _ = network(data(positions), compute_stress=False)
        # g(t) = E(R0 + t d);  g'(t) = -F_0 . d, from the model's own autograd.
        return float(energy), float(-(forces[0] * self.DIRECTION).sum())

    def _pin_threshold_at_crossing(self, network, offset):
        """Put atom 0 exactly on a join of expert 0 at ``t = 0``."""

        scores = router_scores(network, data(POSITIONS))
        with torch.no_grad():
            network.layers[0].routed_ffn.router.threshold[0] = scores[0, 0] + offset
        return float(scores[0, 0])

    def _assert_crossing_is_transversal(self, network):
        span = 4.0e-3
        low = router_scores(
            network,
            data(
                torch.cat(
                    ((POSITIONS[0] - span * self.DIRECTION)[None], POSITIONS[1:])
                )
            ),
        )[0, 0]
        high = router_scores(
            network,
            data(
                torch.cat(
                    ((POSITIONS[0] + span * self.DIRECTION)[None], POSITIONS[1:])
                )
            ),
        )[0, 0]
        self.assertGreater(
            abs(float(high - low)),
            1.0e-7,
            "the score is stationary along the path, so nothing is crossed",
        )

    def _second_derivative_jump(self, network, h):
        """``|g''(0+) - g''(0-)|`` from one-sided differences of the force."""

        _, slope_centre = self._energy_and_slope(network, 0.0)
        _, slope_left = self._energy_and_slope(network, -h)
        _, slope_right = self._energy_and_slope(network, h)
        left = (slope_centre - slope_left) / h
        right = (slope_right - slope_centre) / h
        return abs(right - left)

    def test_energy_is_c2_across_both_joins_of_the_switch(self):
        for offset, join in ((0.0, "u=0"), (1.0, "u=1")):
            network = routed_model(num_experts=1, tau=1.0)
            self._pin_threshold_at_crossing(network, offset)
            self._assert_crossing_is_transversal(network)
            # A C^2 energy has no jump in g'', so the one-sided estimates must
            # converge together as the step is refined.  A kink would leave a
            # fixed, step-independent gap.
            jumps = [
                self._second_derivative_jump(network, h)
                for h in (8.0e-4, 4.0e-4, 2.0e-4)
            ]
            self.assertLess(
                jumps[-1], jumps[0],
                f"second-derivative gap does not shrink at join {join}: {jumps}",
            )
            self.assertLess(
                jumps[-1], 0.35 * jumps[0],
                f"gap shrinks too slowly for a C^2 join at {join}: {jumps}",
            )

    def test_a_hard_threshold_router_fails_the_same_test(self):
        """The control: the test has teeth only if top-k style routing fails it.

        Replacing the smooth switch by an indicator makes the energy jump at the
        crossing.  The central difference of the energy then estimates
        ``jump / 2h``, which *grows* without bound as the step is refined, while
        the model's own force stays finite -- the delta function in
        ``F = -grad E`` made visible.
        """

        network = routed_model(num_experts=1, tau=1.0)
        # Threshold exactly on the score at t = 0, so the indicator flips there.
        self._pin_threshold_at_crossing(network, 0.0)
        self._assert_crossing_is_transversal(network)
        router = network.layers[0].routed_ffn.router
        # Give the routed branch enough weight for the jump to be measurable.
        with torch.no_grad():
            network.layers[0].routed_ffn.expert_scale.fill_(1.0)
        router.weights_from_scores = lambda scores: (
            scores >= router.threshold
        ).to(scores.dtype)

        errors = []
        for h in (8.0e-4, 4.0e-4, 2.0e-4):
            _, slope = self._energy_and_slope(network, 0.0)
            forward, _ = self._energy_and_slope(network, h)
            backward, _ = self._energy_and_slope(network, -h)
            errors.append(abs((forward - backward) / (2.0 * h) - slope))
        self.assertGreater(
            errors[-1], 4.0 * errors[0],
            f"hard routing was expected to diverge as 1/h, saw {errors}",
        )


class RoutingGradientSupportTests(unittest.TestCase):
    """The gradient is exactly zero outside the band. This is the price of
    compact support, and it has consequences worth pinning.

    The same ``f'(0) = f'(1) = 0`` that buys the C^k contract also removes the
    learning signal at both joins, so only pairs strictly inside ``0 < u < 1``
    teach the router anything.
    """

    @staticmethod
    def _router(threshold, tau=1.0, contract="c2", seed=0):
        torch.manual_seed(seed)
        router = CompactSupportRouter(4, 3, tau=tau, contract=contract).double()
        with torch.no_grad():
            router.threshold.fill_(threshold)
        return router

    def _grads(self, router, context):
        weights = router.weights_from_scores(router.scores(context))
        router.zero_grad()
        weights.sum().backward()
        return weights, router

    def test_saturated_on_gives_the_router_no_gradient(self):
        torch.manual_seed(0)
        context = torch.randn(16, 4, dtype=torch.float64)
        router = self._router(-50.0)
        weights, router = self._grads(router, context)
        self.assertTrue(bool((weights == 1.0).all()))
        self.assertEqual(float(router.score_projection.weight.grad.abs().max()), 0.0)
        self.assertEqual(float(router.threshold.grad.abs().max()), 0.0)

    def test_fully_off_gives_the_router_no_gradient(self):
        torch.manual_seed(0)
        context = torch.randn(16, 4, dtype=torch.float64)
        router = self._router(50.0)
        weights, router = self._grads(router, context)
        self.assertTrue(bool((weights == 0.0).all()))
        self.assertEqual(float(router.score_projection.weight.grad.abs().max()), 0.0)
        self.assertEqual(float(router.threshold.grad.abs().max()), 0.0)

    def test_inside_the_band_the_router_does_get_gradient(self):
        torch.manual_seed(0)
        context = torch.randn(64, 4, dtype=torch.float64)
        router = self._router(0.5)  # band centre for tau = 1
        weights, router = self._grads(router, context)
        self.assertGreater(float(((weights > 0.0) & (weights < 1.0)).sum()), 0)
        self.assertGreater(float(router.score_projection.weight.grad.abs().max()), 0.0)

    def test_a_dead_expert_cannot_revive_under_gradient_descent(self):
        """Documented consequence, not a defect: nothing in the loss reaches it."""

        torch.manual_seed(1)
        module = RoutedScalarFFN(
            context_dim=6, out_dim=4, num_experts=3, expert_hidden=5,
            tau=0.5, contract="c2", balance_rate=0.0,
        ).double().train()
        with torch.no_grad():
            module.router.threshold.copy_(
                torch.tensor([-1.0, 0.0, 30.0], dtype=torch.float64)
            )
        context = torch.randn(32, 6, dtype=torch.float64)

        def occupancy():
            weights = module.router.weights_from_scores(module.router.scores(context))
            return (weights > 0.0).to(weights.dtype).mean(dim=0)

        self.assertEqual(float(occupancy()[2]), 0.0)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.05)
        for _ in range(200):
            optimizer.zero_grad()
            (module(context) - 1.0).square().mean().backward()
            optimizer.step()
        self.assertEqual(
            float(occupancy()[2]), 0.0,
            "a dead expert revived without the balancing bias, so the "
            "zero-gradient analysis in routing.py is wrong",
        )

    def test_the_balancing_bias_is_the_only_thing_that_moves_a_dead_expert(self):
        torch.manual_seed(1)
        module = RoutedScalarFFN(
            context_dim=6, out_dim=4, num_experts=3, expert_hidden=5,
            tau=0.5, contract="c2", balance_rate=0.05,
        ).double().train()
        with torch.no_grad():
            module.router.threshold.copy_(
                torch.tensor([-1.0, 0.0, 30.0], dtype=torch.float64)
            )
        context = torch.randn(32, 6, dtype=torch.float64)
        for _ in range(50):
            module(context)
        # The bias walks the dead expert's effective threshold down at rate
        # gamma per step; it is outside the gradient, so it is unaffected by
        # f'(1) = 0.
        self.assertLess(float(module.router.balance_bias[2]), -1.0)

    def test_transition_fraction_counts_exactly_the_pairs_with_gradient(self):
        torch.manual_seed(2)
        module = RoutedScalarFFN(
            context_dim=6, out_dim=3, num_experts=4, expert_hidden=5, tau=1.0
        ).double()
        context = torch.randn(48, 6, dtype=torch.float64)
        scores = module.router.scores(context).detach().requires_grad_(True)
        weights = module.router.weights_from_scores(scores)
        weights.sum().backward()
        # dw/ds is nonzero exactly where 0 < w < 1.
        has_gradient = scores.grad != 0.0
        in_band = (weights > 0.0) & (weights < 1.0)
        self.assertTrue(bool((has_gradient == in_band).all()))
        report = module.routing_statistics(context)
        self.assertAlmostEqual(
            report["transition_fraction"],
            float(in_band.sum()) / float(weights.numel()),
            places=12,
        )


class SwitchBandCentreTests(unittest.TestCase):
    """Both switches are symmetric about u = 1/2, which sets the default init."""

    def test_both_switches_are_one_half_at_the_band_centre(self):
        for contract in ("c2", "c4"):
            value = switch_polynomial(
                torch.tensor(0.5, dtype=torch.float64), contract
            )
            self.assertEqual(float(value), 0.5)

    def test_the_gradient_magnitude_peaks_at_the_band_centre(self):
        for contract in ("c2", "c4"):
            x = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64, requires_grad=True)
            gradient = torch.autograd.grad(
                switch_polynomial(x, contract).sum(), x
            )[0]
            self.assertEqual(int(gradient.abs().argmax()), 500)

    def test_the_default_threshold_centres_the_band(self):
        for tau in (0.5, 1.0, 2.0):
            router = CompactSupportRouter(8, 3, tau=tau)
            self.assertAlmostEqual(float(router.threshold[0]), 0.5 * tau, places=12)
        # An explicit value still wins.
        router = CompactSupportRouter(8, 3, tau=1.0, threshold_init=0.0)
        self.assertEqual(float(router.threshold[0]), 0.0)

    def test_a_freshly_built_router_actually_has_gradient(self):
        """The default must not start the model in the no-gradient regime."""

        torch.manual_seed(0)
        router = CompactSupportRouter(8, 4, tau=1.0).double()
        context = torch.randn(200, 8, dtype=torch.float64)
        weights = router.weights_from_scores(router.scores(context))
        router.zero_grad()
        weights.sum().backward()
        self.assertGreater(float(((weights > 0.0) & (weights < 1.0)).sum()), 0)
        self.assertGreater(float(router.score_projection.weight.grad.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
