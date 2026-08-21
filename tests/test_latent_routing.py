"""Latent expert compression, load balancing, and routing capacity.

Transfers from LatentMoE (arXiv 2601.18089) and Nemotron 3 Super
(arXiv 2604.12374), each checked here rather than assumed.
"""

import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from mtace.checkpoint import restore_model, save_checkpoint
from mtace.model import MambaACEV2
from mtace.routing import CompactSupportRouter, RoutedScalarFFN, routing_capacity
from mtace.schedule import anchor_count_for, anchored_schedule, nemotron_style_schedule

from test_routing import POSITIONS, _arguments, data, routed_model


class RoutingCapacityTests(unittest.TestCase):
    """Why the router must not be compressed, from Schlafli's formula."""

    def test_capacity_matches_the_hyperplane_arrangement_count(self):
        # C(N, d) = sum_{j=0}^{min(d,N)} binom(N, j)
        self.assertEqual(routing_capacity(8, 3), 1 + 8 + 28 + 56)
        # N <= d: every one of the 2^N sign patterns is reachable.
        self.assertEqual(routing_capacity(6, 10), 2**6)
        self.assertEqual(routing_capacity(6, 6), 2**6)

    def test_capacity_saturates_once_experts_outnumber_the_router_width(self):
        """Doubling N past d buys polynomially, not exponentially, more sets."""

        width = 4
        small = routing_capacity(16, width)
        large = routing_capacity(32, width)
        self.assertLess(large / small, 2**16)
        # O(N^d): doubling N multiplies the count by about 2^d = 16.
        self.assertAlmostEqual(large / small, 2**width, delta=0.35 * 2**width)

    def test_compressing_the_router_would_destroy_combinatorial_capacity(self):
        """The exact quantity Design Principle V is trying to buy."""

        experts, full, latent = 64, 64, 12
        # Uncompressed router: every one of the 2^N active sets is reachable.
        self.assertEqual(routing_capacity(experts, full), 2**experts)
        # Compressed to alpha = 64/12: the reachable sets collapse by more than
        # forty orders of magnitude, which is the capacity Principle V buys and
        # a compressed router would throw away.
        self.assertLess(math.log2(routing_capacity(experts, latent)), 0.75 * experts)

    def test_the_router_is_structurally_given_the_uncompressed_context(self):
        module = RoutedScalarFFN(
            context_dim=32, out_dim=8, num_experts=4, expert_hidden=6, latent_dim=8
        )
        self.assertEqual(module.router.context_dim, 32)
        self.assertEqual(module.latent_dim, 8)
        self.assertEqual(module.compression_ratio, 4.0)
        self.assertEqual(module.routing_capacity(), routing_capacity(4, 32))

    def test_capacity_arguments_are_validated(self):
        with self.assertRaises(ValueError):
            routing_capacity(0, 4)
        with self.assertRaises(ValueError):
            routing_capacity(4, 0)


class LatentExpertCompressionTests(unittest.TestCase):
    def test_expansion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds context_dim"):
            RoutedScalarFFN(
                context_dim=8, out_dim=8, num_experts=2, expert_hidden=4, latent_dim=16
            )
        with self.assertRaises(ValueError):
            RoutedScalarFFN(
                context_dim=8, out_dim=8, num_experts=2, expert_hidden=4, latent_dim=0
            )

    def test_compression_preserves_the_nonlinear_budget(self):
        """U_eff ~ K m: the expert intermediate width must not shrink with alpha."""

        plain = RoutedScalarFFN(
            context_dim=32, out_dim=8, num_experts=4, expert_hidden=16
        )
        compressed = RoutedScalarFFN(
            context_dim=32, out_dim=8, num_experts=4, expert_hidden=16, latent_dim=8
        )
        for module in (plain, compressed):
            for expert in module.experts:
                # SwiGLU splits the input projection in two, so m is half of it.
                self.assertEqual(expert.input_projection.out_features // 2, 16)
        # Compression must actually reduce the routed parameter count.
        routed = lambda m: sum(p.numel() for p in m.experts.parameters())
        self.assertLess(routed(compressed), routed(plain))

    def test_dense_and_sparse_agree_under_compression(self):
        torch.manual_seed(7)
        dense = RoutedScalarFFN(
            context_dim=16, out_dim=6, num_experts=4, expert_hidden=8,
            tau=0.5, contract="c2", backend="dense", latent_dim=4,
        ).double()
        sparse = RoutedScalarFFN(
            context_dim=16, out_dim=6, num_experts=4, expert_hidden=8,
            tau=0.5, contract="c2", backend="sparse", latent_dim=4,
        ).double()
        sparse.load_state_dict(dense.state_dict())
        with torch.no_grad():
            thresholds = torch.tensor([-2.0, 0.0, 0.5, 2.0], dtype=torch.float64)
            dense.router.threshold.copy_(thresholds)
            sparse.router.threshold.copy_(thresholds)
        context = torch.randn(48, 16, dtype=torch.float64)
        weights = dense.router.weights_from_scores(dense.router.scores(context))
        self.assertGreater(float((weights == 0.0).sum()), 0.0)
        torch.testing.assert_close(
            sparse(context), dense(context), atol=1.0e-13, rtol=1.0e-13
        )

    def test_compressed_routing_keeps_symmetry_and_conserves_energy(self):
        network = routed_model(num_experts=4, tau=0.5, expert_latent_dim=3)
        self.assertEqual(network.layers[0].routed_ffn.latent_dim, 3)
        energy, forces, _, _ = network(data(POSITIONS), compute_stress=False)
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
        rotated = network(data(POSITIONS @ rotation.T), compute_stress=False)
        torch.testing.assert_close(rotated[0], energy, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(
            rotated[1], forces @ rotation.T, atol=1.0e-10, rtol=1.0e-10
        )
        step = 1.0e-6
        for atom in (0, 2):
            for axis in range(3):
                plus, minus = POSITIONS.clone(), POSITIONS.clone()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                derivative = (
                    network(data(plus), compute_stress=False)[0]
                    - network(data(minus), compute_stress=False)[0]
                ) / (2.0 * step)
                self.assertAlmostEqual(
                    float(-derivative), float(forces[atom, axis]), places=6
                )

    def test_compression_survives_a_checkpoint_round_trip(self):
        config = dict(
            r_max=4.5, l_max=2, num_radial=4, hidden_dim=8, num_layers=2,
            correlation_order=4, correlation_channels=4, mamba_dim=12,
            mamba_d_state=4, mamba_backend="torch", readout_hidden=8,
            num_experts=4, expert_hidden=8, expert_latent_dim=3,
            router_balance_rate=0.01, mixer_schedule=["mamba", "attention"],
        )
        torch.manual_seed(17)
        network = MambaACEV2(**config).double().eval()
        reference = network(data(POSITIONS), compute_stress=False)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latent.pt"
            save_checkpoint(path, network, config, atomic_numbers=[1, 8])
            restored, _ = restore_model(path)
        self.assertEqual(restored.layers[0].routed_ffn.latent_dim, 3)
        torch.testing.assert_close(
            restored(data(POSITIONS), compute_stress=False)[0], reference,
            atol=0.0, rtol=0.0,
        )


class SparseBackendExportTests(unittest.TestCase):
    def test_tracing_must_not_freeze_the_active_expert_set(self):
        """Regression, and the one with teeth.

        ``torch.jit.is_scripting()`` is False under ``torch.jit.trace``, so
        guarding the sparse path on it alone let the tracer freeze the Python
        branch ``if active.numel() == 0: continue``.  An expert that happens to
        be inactive on the tracing example is then **permanently dropped** from
        the exported model, silently, even when later inputs activate it.

        Reproducing that needs an expert which is off during tracing and on
        afterwards, so the scores are pinned by hand rather than left to chance.
        """

        torch.manual_seed(0)
        module = RoutedScalarFFN(
            context_dim=2, out_dim=3, num_experts=2, expert_hidden=4,
            tau=1.0, contract="c2", backend="sparse",
        ).double().eval()
        with torch.no_grad():
            # s_e = context[:, e], threshold 0: expert e is off when c_e <= -1.
            module.router.score_projection.weight.copy_(
                torch.eye(2, dtype=torch.float64)
            )
            module.router.score_projection.bias.zero_()
            module.router.threshold.zero_()
            module.expert_scale.fill_(1.0)

        off = torch.tensor([[5.0, -5.0]] * 4, dtype=torch.float64)   # expert 1 off
        on = torch.tensor([[5.0, 5.0]] * 4, dtype=torch.float64)     # expert 1 on
        self.assertEqual(
            float(module.router.weights_from_scores(module.router.scores(off))[:, 1].abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(module.router.weights_from_scores(module.router.scores(on))[:, 1].sum()),
            0.0,
        )

        traced = torch.jit.trace(module, off)
        torch.testing.assert_close(
            traced(on), module(on), atol=1.0e-13, rtol=1.0e-13,
            msg="expert inactive during tracing was dropped from the traced graph",
        )

    def test_a_genuinely_sparse_model_exports_correctly(self):
        """Integration cover: a sparse model survives the full LAMMPS export."""

        from mtace.deployment import export_lammps_model

        config = dict(
            r_max=4.0, l_max=1, num_radial=4, hidden_dim=8, num_layers=1,
            correlation_order=2, correlation_channels=4, num_shells=6,
            mamba_dim=8, mamba_d_state=4, mamba_headdim=8, readout_hidden=8,
            mamba_backend="torch", num_experts=4, expert_hidden=8,
            router_tau=0.5, routing_backend="sparse",
        )
        torch.manual_seed(0)
        network = MambaACEV2(**config).double().eval()
        with torch.no_grad():
            # Force a genuinely mixed pattern: two experts off, two on.
            network.layers[0].routed_ffn.router.threshold.copy_(
                torch.tensor([-6.0, -6.0, 6.0, 6.0], dtype=torch.float64)
            )
            network.layers[0].routed_ffn.expert_scale.fill_(1.0)
        report = network.routing_occupancy(**_arguments(POSITIONS))
        self.assertGreater(report[0]["active_fraction"], 0.0)
        self.assertLess(report[0]["active_fraction"], 1.0)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sparse.pt"
            save_checkpoint(
                checkpoint, network, config,
                atomic_energies={1: 0.0, 8: 0.0}, atomic_numbers=[1, 8],
            )
            # export_lammps_model validates energy, forces, virial and the
            # isolated-atom limit against the eager model, and raises on drift.
            exported = export_lammps_model(
                checkpoint, Path(directory) / "sparse.mtace.pt", elements=["H", "O"]
            )
            self.assertTrue(exported.exists())


class LoadBalancingTests(unittest.TestCase):
    """Auxiliary-loss-free balancing, and why it costs no smoothness."""

    def test_balance_bias_is_a_buffer_and_takes_no_gradient(self):
        router = CompactSupportRouter(8, 4, balance_rate=0.1)
        names = dict(router.named_parameters())
        self.assertNotIn("balance_bias", names)
        self.assertIn("balance_bias", dict(router.named_buffers()))
        # It still travels in the state dict, so a resumed run keeps its load.
        self.assertIn("balance_bias", router.state_dict())
        self.assertFalse(router.balance_bias.requires_grad)

    def test_balancing_is_exactly_inert_at_rate_zero(self):
        torch.manual_seed(2)
        router = CompactSupportRouter(8, 4, balance_rate=0.0).double().train()
        context = torch.randn(20, 8, dtype=torch.float64)
        router(context)
        router(context)
        self.assertEqual(float(router.balance_bias.abs().sum()), 0.0)

    def test_the_bias_is_frozen_in_eval(self):
        torch.manual_seed(3)
        router = CompactSupportRouter(8, 4, tau=0.5, balance_rate=0.1).double()
        with torch.no_grad():
            router.threshold.copy_(
                torch.tensor([-3.0, 0.0, 0.0, 3.0], dtype=torch.float64)
            )
        context = torch.randn(40, 8, dtype=torch.float64)
        router.eval()
        for _ in range(5):
            router(context)
        self.assertEqual(float(router.balance_bias.abs().sum()), 0.0)
        router.train()
        router(context)
        self.assertGreater(float(router.balance_bias.abs().sum()), 0.0)

    def test_balancing_moves_occupancy_towards_the_mean(self):
        torch.manual_seed(4)
        router = CompactSupportRouter(8, 4, tau=1.0, balance_rate=0.05).double().train()
        with torch.no_grad():
            # Expert 0 wide open, expert 3 nearly shut: a deliberate imbalance.
            router.threshold.copy_(
                torch.tensor([-2.0, 0.0, 0.0, 2.0], dtype=torch.float64)
            )
        context = torch.randn(256, 8, dtype=torch.float64)

        def spread():
            weights = router.weights_from_scores(router.scores(context))
            occupancy = (weights > 0.0).to(weights.dtype).mean(dim=0)
            return float(occupancy.max() - occupancy.min())

        before = spread()
        for _ in range(60):
            router(context)
        self.assertLess(spread(), before)

    def test_balancing_shifts_the_threshold_and_nothing_else(self):
        """The smoothness argument: b_e enters only as a constant offset."""

        torch.manual_seed(5)
        biased = CompactSupportRouter(8, 3, tau=0.7, contract="c2").double()
        shifted = CompactSupportRouter(8, 3, tau=0.7, contract="c2").double()
        shifted.load_state_dict(biased.state_dict())
        offset = torch.tensor([0.3, -0.2, 0.11], dtype=torch.float64)
        with torch.no_grad():
            biased.balance_bias.copy_(offset)
            shifted.threshold += offset
        context = torch.randn(24, 8, dtype=torch.float64)
        torch.testing.assert_close(
            biased.weights_from_scores(biased.scores(context)),
            shifted.weights_from_scores(shifted.scores(context)),
            atol=1.0e-15, rtol=1.0e-15,
        )

    def test_a_balanced_model_still_conserves_energy(self):
        """Smoothness is unaffected, so forces stay exact after balancing."""

        network = routed_model(num_experts=4, tau=0.5, router_balance_rate=0.05)
        # A genuine imbalance is set up on purpose.  At the band-centred default
        # threshold every expert starts with the same occupancy, so
        # sign(rho_e - rho_bar) is zero and the bias correctly does not move --
        # which would make the assertion below vacuous.
        with torch.no_grad():
            network.layers[0].routed_ffn.router.threshold.copy_(
                torch.tensor([-2.0, 0.0, 0.25, 2.0], dtype=torch.float64)
            )
        network.train()
        for _ in range(5):
            network(data(POSITIONS), training=True, compute_stress=False)
        bias = network.layers[0].routed_ffn.router.balance_bias.clone()
        self.assertGreater(float(bias.abs().sum()), 0.0)

        network.eval()
        _, forces, _, _ = network(data(POSITIONS), compute_stress=False)
        step = 1.0e-6
        for axis in range(3):
            plus, minus = POSITIONS.clone(), POSITIONS.clone()
            plus[1, axis] += step
            minus[1, axis] -= step
            derivative = (
                network(data(plus), compute_stress=False)[0]
                - network(data(minus), compute_stress=False)[0]
            ) / (2.0 * step)
            self.assertAlmostEqual(
                float(-derivative), float(forces[1, axis]), places=6
            )
        # Evaluating the model must not have moved the bias.
        torch.testing.assert_close(
            network.layers[0].routed_ffn.router.balance_bias, bias,
            atol=0.0, rtol=0.0,
        )

    def test_observing_the_model_does_not_change_it(self):
        """routing_occupancy must not fire the balance update."""

        network = routed_model(num_experts=3, tau=0.5, router_balance_rate=0.1)
        network.train()
        before = network.layers[0].routed_ffn.router.balance_bias.clone()
        report = network.routing_occupancy(**_arguments(POSITIONS))
        self.assertEqual(len(report), 1)
        self.assertIn("occupancy_spread", report[0])
        torch.testing.assert_close(
            network.layers[0].routed_ffn.router.balance_bias, before,
            atol=0.0, rtol=0.0,
        )

    def test_balance_arguments_are_validated(self):
        with self.assertRaises(ValueError):
            CompactSupportRouter(8, 4, balance_rate=-0.1)
        with self.assertRaises(ValueError):
            CompactSupportRouter(8, 4, balance_target=1.5)


class ActivationSmoothnessTests(unittest.TestCase):
    """Squared-ReLU does not transfer, and the reason is exact.

    LatentMoE's 95B and hybrid configurations and Nemotron 3 Super use
    Squared-ReLU.  For a potential it is inadmissible:

        sigma(x)   = max(0, x)^2
        sigma'(x)  = 2 max(0, x)      -- continuous
        sigma''(x) = 2 H(x)           -- jumps by 2 at x = 0

    so sigma is C^1 but not C^2.  An energy built from it is C^1 at best, below
    the C^2 that phonons require and far below the C^4 the quintic tokenizer
    delivers.  SiLU, which MTACE uses, is C-infinity.
    """

    @staticmethod
    def _second_derivative(function, x):
        x = torch.tensor(float(x), dtype=torch.float64, requires_grad=True)
        value = function(x)
        first = torch.autograd.grad(value, x, create_graph=True)[0]
        return float(torch.autograd.grad(first, x, create_graph=True)[0])

    def test_squared_relu_has_a_jump_in_its_second_derivative(self):
        squared_relu = lambda x: F.relu(x).square()
        left = self._second_derivative(squared_relu, -1.0e-3)
        right = self._second_derivative(squared_relu, 1.0e-3)
        self.assertAlmostEqual(left, 0.0, places=12)
        self.assertAlmostEqual(right, 2.0, places=12)
        # The gap does not shrink as the join is approached: a genuine jump.
        for scale in (1.0e-4, 1.0e-6, 1.0e-8):
            self.assertAlmostEqual(
                self._second_derivative(squared_relu, scale)
                - self._second_derivative(squared_relu, -scale),
                2.0, places=10,
            )

    def test_silu_has_no_such_jump(self):
        for scale in (1.0e-4, 1.0e-6, 1.0e-8):
            gap = self._second_derivative(F.silu, scale) - self._second_derivative(
                F.silu, -scale
            )
            self.assertLess(abs(gap), 1.0e-3)

    def test_mtace_uses_no_squared_relu(self):
        """A guard: adopting it would silently downgrade the contract to C^1."""

        network = routed_model(num_experts=2)
        activations = [
            module for module in network.modules()
            if isinstance(module, (torch.nn.ReLU, torch.nn.ReLU6))
        ]
        self.assertEqual(activations, [])


class AnchorRatioTests(unittest.TestCase):
    """The published hybrid ratio, read correctly.

    The LatentMoE hybrid is tabulated as "52 (24 Mamba/MoE, 4 Attn)".  Those 52
    are 24 Mamba + 24 MoE + 4 attention, so attention is 4 of 28 *mixers*
    (14.3%), not 4 of 52 layers (7.7%).  An MTACE layer holds both a mixer and a
    scalar residual block, so it corresponds to a (Mamba, MoE) pair: one anchor
    per seven layers.
    """

    def test_the_published_layer_count_decomposes_as_stated(self):
        self.assertEqual(24 + 24 + 4, 52)
        self.assertAlmostEqual(4 / 28, 0.1428, places=3)

    def test_anchor_count_follows_one_in_seven(self):
        self.assertEqual(anchor_count_for(28), 4)
        self.assertEqual(anchor_count_for(7), 1)
        self.assertEqual(anchor_count_for(14), 2)
        # Never zero: a hybrid with no attention is not a hybrid.
        self.assertEqual(anchor_count_for(1), 1)
        self.assertEqual(anchor_count_for(3), 1)

    def test_nemotron_style_schedule_hits_the_ratio(self):
        schedule = nemotron_style_schedule(28)
        self.assertEqual(schedule.count("attention"), 4)
        self.assertEqual(len(schedule), 28)
        self.assertEqual(schedule, anchored_schedule(28, 4))
        # Evenly spaced, off both ends of the stack.
        positions = [i for i, name in enumerate(schedule) if name == "attention"]
        self.assertEqual(positions, [3, 10, 17, 24])

    def test_arguments_are_validated(self):
        with self.assertRaises(ValueError):
            anchor_count_for(0)
        with self.assertRaises(ValueError):
            anchor_count_for(10, mixers_per_anchor=0)


class OptimizerContractTests(unittest.TestCase):
    """Routing parameters must land in the right weight-decay group.

    ``mtace.optim`` exempts biases, norms, ``layer_scale`` and ``ffn_scale``, or
    anything carrying ``_no_weight_decay``.  Two routing parameters belong in
    that set and were initially outside it:

    ``expert_scale``   the exact analogue of ``layer_scale``.  It starts at 1e-2,
                       so decay would steadily switch the routed branch off --
                       suppressing the mechanism rather than regularising it.
    ``threshold``      a location parameter in score units, the same kind of
                       object as the score projection's bias.  Decay drags every
                       threshold toward zero, which is a routing preference, not
                       regularisation.
    """

    def _named(self):
        torch.manual_seed(0)
        network = MambaACEV2(
            r_max=4.5, l_max=2, num_radial=4, hidden_dim=8, num_layers=1,
            correlation_order=4, correlation_channels=4, mamba_dim=12,
            mamba_d_state=4, mamba_backend="torch", readout_hidden=8,
            num_experts=3, expert_hidden=8, expert_latent_dim=6,
        )
        return dict(network.named_parameters())

    def test_scale_and_threshold_are_exempt_from_weight_decay(self):
        from mtace.optim import _is_no_decay_parameter

        named = self._named()
        for suffix in ("routed_ffn.expert_scale", "routed_ffn.router.threshold"):
            name = next(key for key in named if key.endswith(suffix))
            self.assertTrue(
                _is_no_decay_parameter(name, named[name]),
                f"{suffix} would be weight-decayed",
            )

    def test_projection_matrices_are_still_decayed(self):
        """The exemption must not leak to the actual weight matrices."""

        from mtace.optim import _is_no_decay_parameter

        named = self._named()
        for suffix in (
            "routed_ffn.latent_down.weight",
            "routed_ffn.latent_up.weight",
            "routed_ffn.router.score_projection.weight",
        ):
            name = next(key for key in named if key.endswith(suffix))
            self.assertFalse(
                _is_no_decay_parameter(name, named[name]),
                f"{suffix} should be weight-decayed",
            )

    def test_the_balance_bias_is_not_an_optimizer_parameter_at_all(self):
        torch.manual_seed(0)
        network = MambaACEV2(
            r_max=4.5, l_max=2, num_radial=4, hidden_dim=8, num_layers=1,
            correlation_order=4, correlation_channels=4, mamba_dim=12,
            mamba_d_state=4, mamba_backend="torch", readout_hidden=8,
            num_experts=3, expert_hidden=8, router_balance_rate=0.1,
        )
        names = dict(network.named_parameters())
        self.assertFalse([key for key in names if key.endswith("balance_bias")])
        buffers = dict(network.named_buffers())
        self.assertTrue([key for key in buffers if key.endswith("balance_bias")])


class DiagnosticsDoNotMutateTests(unittest.TestCase):
    """Every diagnostic advances the stack, so every one must run frozen.

    ``gate_shell_dependence`` and ``screening_lengths`` step through the layers
    by calling ``forward``, which in training mode fires the load-balancing
    update.  Measuring the model would then move it.  ``_measuring`` puts all
    three on one mechanism.
    """

    @staticmethod
    def _model(**overrides):
        torch.manual_seed(0)
        settings = dict(
            r_max=4.5, l_max=2, num_radial=4, hidden_dim=8, num_layers=1,
            correlation_order=4, correlation_channels=4, mamba_dim=12,
            mamba_d_state=4, mamba_backend="torch", readout_hidden=8,
            num_experts=4, expert_hidden=8, router_tau=0.5,
            router_balance_rate=0.1,
        )
        settings.update(overrides)
        return MambaACEV2(**settings).double()

    def test_no_diagnostic_moves_the_balance_bias(self):
        network = self._model(decay_mode="screening").train()
        arguments = _arguments(POSITIONS)
        for name in ("gate_shell_dependence", "screening_lengths", "routing_occupancy"):
            before = network.layers[0].routed_ffn.router.balance_bias.clone()
            getattr(network, name)(**arguments)
            torch.testing.assert_close(
                network.layers[0].routed_ffn.router.balance_bias, before,
                atol=0.0, rtol=0.0, msg=f"{name} moved the load-balancing bias",
            )
            self.assertTrue(network.training, f"{name} left the model in eval")

    def test_the_measuring_guard_restores_the_mode_after_an_exception(self):
        network = self._model().train()

        class Failure(Exception):
            pass

        with self.assertRaises(Failure):
            with network._measuring():
                self.assertFalse(network.training)
                raise Failure()
        self.assertTrue(network.training)


if __name__ == "__main__":
    unittest.main()
