"""Architecture-v12: the derivative-order contract for the whole model.

Each test pins one claim, and where possible it ships with its control: the v11
setting that *fails* the same measurement.  A smoothness test that cannot fail
demonstrates nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mtace.checkpoint import migrated_model_config
from mtace.mamba3 import (
    Mamba3Direction,
    Mamba3SequenceMixer,
    heavy_tail_activation,
    smooth_heavy_tail_activation,
)
from mtace.model import MambaACEV2
from mtace.optim import get_muon_param_groups
from mtace.physics import SmoothPolynomialCutoff
from mtace.routing import switch_polynomial

DOUBLE = torch.float64
R_MAX = 6.0


def _model(**overrides):
    torch.manual_seed(9)
    settings = dict(
        r_max=R_MAX, l_max=2, num_radial=8, hidden_dim=16, num_layers=1,
        correlation_order=3, correlation_channels=8, num_shells=12,
        shell_r_min=1.5, shell_degree=5, avg_num_neighbors=12.0,
        invariant_norm="homogeneous", mamba_dim=16, mamba_d_state=8,
        mamba_headdim=8, readout_hidden=16, mamba_backend="torch",
    )
    settings.update(overrides)
    return MambaACEV2(**settings).double().eval()


def _cluster(seed=2, count=6):
    torch.manual_seed(seed)
    positions = torch.randn(count, 3, dtype=DOUBLE) * 1.7
    z = torch.tensor([1, 6, 8, 1, 6, 8][:count], dtype=torch.long)
    cell = torch.eye(3, dtype=DOUBLE) * 100.0
    senders, receivers = [], []
    for i in range(count):
        for j in range(count):
            if i != j and torch.linalg.norm(positions[j] - positions[i]) < R_MAX:
                senders.append(j)
                receivers.append(i)
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    return positions, z, cell, edge_index, torch.zeros(edge_index.shape[1], 3, dtype=DOUBLE)


def _derivatives(function, point, order):
    """Autograd derivatives ``f^(1) .. f^(order)`` of a scalar function at ``point``."""

    t = torch.tensor(point, dtype=DOUBLE, requires_grad=True)
    value = function(t)
    out = []
    for _ in range(order):
        (value,) = torch.autograd.grad(value, t, create_graph=True)
        out.append(float(value))
    return out


# ---------------------------------------------------------------------------
# Screened decay
# ---------------------------------------------------------------------------


class TestScreeningIsComplete:
    def test_both_scan_directions_are_screened(self):
        """Under screening the free-decay projection must reach neither scan.

        v11 passed ``screening`` to the forward scan only, so the reverse scan
        kept a learned per-head rate: perturbing its decay rows moved the output.
        """

        torch.manual_seed(0)
        mixer = Mamba3SequenceMixer(
            d_model=16, d_state=8, expand=2, headdim=8, backend="torch"
        ).double()
        hidden = torch.randn(3, 10, 16, dtype=DOUBLE)
        screening = torch.rand(3, 1, 1, dtype=DOUBLE) + 0.1
        with torch.no_grad():
            reference = mixer(hidden, screening=screening)
            for direction in (mixer.forward_direction, mixer.backward_direction):
                start = (
                    2 * direction.d_inner
                    + 2 * direction.d_state * direction.mimo_rank
                    + direction.nheads
                )
                rows = slice(start, start + direction.nheads)
                direction.in_proj.weight[rows] += 5.0
                torch.testing.assert_close(
                    mixer(hidden, screening=screening), reference, rtol=0.0, atol=0.0
                )
                # Control: without screening the same rows do matter.
                assert not torch.allclose(mixer(hidden), mixer(hidden) * 0.0 + reference)

    def test_regridding_updates_the_physical_spacing(self):
        model = _model(decay_mode="screening", num_shells=11, shell_r_min=1.0)
        assert model.layers[0].shell_spacing_angstrom == pytest.approx(0.5)
        model.set_num_shells(21)
        assert model.layers[0].shell_spacing_angstrom == pytest.approx(0.25)

    def test_multiscale_screening_is_rejected(self):
        with pytest.raises(ValueError, match="shell_scales"):
            _model(decay_mode="screening", shell_scales=2)

    @pytest.mark.parametrize(
        "overrides",
        [dict(mixer_type="attention"), dict(mixer_type="identity"),
         dict(mamba_variant="mamba1", mamba_d_conv=3)],
    )
    def test_screening_needs_a_mamba3_layer(self, overrides):
        with pytest.raises(ValueError, match="mamba3"):
            _model(decay_mode="screening", **overrides)

    def test_a_hybrid_screens_only_its_mamba3_layers(self):
        model = _model(
            decay_mode="screening", num_layers=3,
            mixer_schedule=["mamba", "attention", "mamba"],
        )
        assert [layer.decay_mode for layer in model.layers] == [
            "screening", "free", "screening"
        ]
        assert model.layers[1].screening_projection is None
        positions, z, cell, edge_index, edge_shift = _cluster()
        lengths = model.screening_lengths(z, positions, cell, edge_index, edge_shift)
        assert len(lengths) == 2


# ---------------------------------------------------------------------------
# Smooth decay activation
# ---------------------------------------------------------------------------


class TestSmoothDecayActivation:
    def test_keeps_every_property_of_the_heavy_tail(self):
        x = torch.linspace(-50.0, 50.0, 2001, dtype=DOUBLE)
        f = smooth_heavy_tail_activation(x)
        assert bool((f > 0).all())
        assert float(smooth_heavy_tail_activation(torch.zeros((), dtype=DOUBLE))) == 1.0
        torch.testing.assert_close(f * smooth_heavy_tail_activation(-x), torch.ones_like(x))
        large = torch.tensor([1.0e4], dtype=DOUBLE)
        assert float(smooth_heavy_tail_activation(large) / large) == pytest.approx(1.0, rel=1e-7)
        assert float(smooth_heavy_tail_activation(-large) * large) == pytest.approx(1.0, rel=1e-7)
        # Same asymptotes as the piecewise form, on both sides.
        assert float(heavy_tail_activation(large) / large) == pytest.approx(1.0, rel=1e-3)
        assert float(heavy_tail_activation(-large) * large) == pytest.approx(1.0, rel=1e-3)

    def test_second_derivative_is_continuous_and_bounded(self):
        left = _derivatives(smooth_heavy_tail_activation, -1e-9, 2)
        right = _derivatives(smooth_heavy_tail_activation, 1e-9, 2)
        assert right[1] == pytest.approx(left[1], abs=1e-8)
        assert right[1] == pytest.approx(0.25, abs=1e-8)  # 2 (x^2 + 4)^(-3/2) at 0
        # Control: the piecewise form jumps 2 -> 0.
        assert _derivatives(heavy_tail_activation, -1e-9, 2)[1] == pytest.approx(2.0, abs=1e-6)
        assert _derivatives(heavy_tail_activation, 1e-9, 2)[1] == 0.0

    def test_unknown_activation_is_rejected(self):
        with pytest.raises(ValueError, match="decay_activation"):
            Mamba3Direction(d_model=8, d_state=4, headdim=4, decay_activation="relu")
        with pytest.raises(ValueError, match="mamba_decay_activation"):
            _model(mixer_type="attention", mamba_decay_activation="relu")

    def test_scan_hessian_is_continuous_where_the_decay_input_changes_sign(self):
        """A path on which one decay pre-activation crosses zero at t = 0.

        The output's second derivative along the path is compared just left and
        just right of the crossing.  The smooth activation must close the gap;
        the piecewise one must leave a step there.
        """

        torch.manual_seed(4)
        smooth = Mamba3Direction(
            d_model=8, d_state=4, expand=2, headdim=4, backend="torch",
            decay_activation="smooth",
        ).double()
        piecewise = Mamba3Direction(
            d_model=8, d_state=4, expand=2, headdim=4, backend="torch",
            decay_activation="heavy_tail",
        ).double()
        piecewise.load_state_dict(smooth.state_dict())
        start = 2 * smooth.d_inner + 2 * smooth.d_state + smooth.nheads
        row = smooth.in_proj.weight[start].detach()
        hidden = torch.randn(1, 4, 8, dtype=DOUBLE)
        hidden[0, 2] -= (hidden[0, 2] @ row) / (row @ row) * row
        direction = torch.zeros_like(hidden)
        direction[0, 2] = row / row.norm()
        probe = torch.randn(1, 4, 8, dtype=DOUBLE)

        def gap(module, delta=1e-6):
            path = lambda t: (module(hidden + t * direction) * probe).sum()
            return abs(_derivatives(path, delta, 2)[1] - _derivatives(path, -delta, 2)[1])

        jump = gap(piecewise)
        assert jump > 1e-6
        assert gap(smooth) < 1e-3 * jump


# ---------------------------------------------------------------------------
# Cutoff envelope
# ---------------------------------------------------------------------------


class TestCutoffEnvelope:
    def test_order_four_envelope_is_the_c4_switch(self):
        r = torch.linspace(0.0, 6.0, 101, dtype=DOUBLE)
        expected = switch_polynomial((r / 6.0).clamp(0.0, 1.0), "c4")
        torch.testing.assert_close(SmoothPolynomialCutoff(6.0, order=4)(r), expected)

    def test_vanishing_order_at_the_cutoff(self):
        c2, c4 = SmoothPolynomialCutoff(R_MAX), SmoothPolynomialCutoff(R_MAX, order=4)
        f2 = lambda t: c2(t.reshape(1)).sum()
        f4 = lambda t: c4(t.reshape(1)).sum()
        # C^2: the third derivative tends to -60 / r_max^3 inside, zero outside.
        assert _derivatives(f2, R_MAX - 1e-7, 3)[2] == pytest.approx(
            -60.0 / R_MAX**3, rel=1e-5
        )
        # C^4: near x = 1 the envelope is 126 (1-x)^5, so the fourth derivative
        # vanishes *linearly*, with the exact coefficient 126 * 5! = 15120.
        for delta in (1e-5, 1e-6):
            fourth = _derivatives(f4, R_MAX - delta, 4)[3]
            assert fourth == pytest.approx(15120.0 * (delta / R_MAX) / R_MAX**4, rel=1e-3)
        assert float(c4(torch.zeros(1, dtype=DOUBLE))) == 1.0

    def test_resolution_follows_the_shell_degree(self):
        assert _model(shell_degree=3).cutoff_switch == "c2"
        assert _model(shell_degree=5).cutoff_switch == "c4"
        assert _model(shell_degree=5).ace.radial_basis.cutoff.order == 4
        assert _model(shell_degree=5, cutoff_switch="c2").ace.radial_basis.cutoff.order == 2
        with pytest.raises(ValueError, match="cutoff_switch"):
            _model(cutoff_switch="c3")

    @pytest.mark.parametrize(
        "basis, order",
        [
            # Gaussian channels are nonzero at r_c, so the edge feature vanishes
            # to the envelope order: the third derivative is the binding one.
            ("gaussian", 3),
            # Fixed Bessel channels add their own simple zero at r_c, which
            # moves the first jump under the C^2 envelope to the fourth.
            ("bessel", 4),
        ],
    )
    def test_energy_vanishing_order_at_the_cutoff(self, basis, order):
        """The whole-model claim, measured on a dimer approaching r_max.

        The k-th energy derivative is evaluated at r_c - delta for two deltas a
        decade apart.  A ratio of one means the derivative tends to a nonzero
        constant, i.e. jumps to zero across the cutoff; a ratio of 10^p means it
        vanishes as delta^p.  The C^2 envelope must jump (control); the C^4
        envelope must vanish as delta^2.
        """

        z = torch.tensor([6, 8], dtype=torch.long)
        cell = torch.eye(3, dtype=DOUBLE) * 100.0
        edge_index = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
        shift = torch.zeros(2, 3, dtype=DOUBLE)

        def derivative(model, delta):
            def energy(d):
                pos = torch.stack(
                    (torch.zeros(3, dtype=DOUBLE), torch.stack((d, d * 0.0, d * 0.0)))
                )
                return model.atomic_energies(z, pos, cell, edge_index, shift).sum()

            return abs(_derivatives(energy, R_MAX - delta, order)[order - 1])

        quintic = _model(mixer_type="identity", layer_scale_init=1.0, radial_basis_type=basis)
        historical = _model(
            mixer_type="identity", layer_scale_init=1.0, radial_basis_type=basis,
            cutoff_switch="c2",
        )
        historical.load_state_dict(quintic.state_dict())
        for model, decades in ((historical, 0.0), (quintic, 2.0)):
            ratio = derivative(model, 1e-3) / derivative(model, 1e-4)
            assert math.log10(ratio) == pytest.approx(decades, abs=0.05)


# ---------------------------------------------------------------------------
# gate_norm scope and weight decay
# ---------------------------------------------------------------------------


class TestGateNormScope:
    def test_all_scope_normalises_every_saturating_head(self):
        states = torch.randn(3, 12, 16, dtype=DOUBLE)
        everywhere = _model(gate_norm="rms").layers[0]
        assert everywhere.gate_norm_scope == "all"
        # RMSNorm is scale invariant up to its eps = 1e-6, i.e. to a relative
        # 1e-6 / (2 mean-square) for unit-variance states.
        torch.testing.assert_close(
            everywhere._head_input(1.0e3 * states), everywhere._head_input(states),
            rtol=1e-5, atol=1e-8,
        )
        gate_only = _model(gate_norm="rms", gate_norm_scope="gate").layers[0]
        torch.testing.assert_close(gate_only._head_input(states), states)

    def test_rejects_unknown_scope(self):
        with pytest.raises(ValueError, match="gate_norm_scope"):
            _model(gate_norm="rms", gate_norm_scope="some")

    def test_normalised_shell_pairs_keep_the_physics(self):
        model = _model(
            gate_norm="rms", coupling_mode="path_weights", coupling_channels=4,
            shell_pair_channels=4, shell_pair_mode="exponential", layer_scale_init=1.0,
        )
        positions, z, cell, edge_index, edge_shift = _cluster()
        energy = lambda p: float(model.atomic_energies(z, p, cell, edge_index, edge_shift).sum())
        rotation = torch.linalg.qr(torch.randn(3, 3, dtype=DOUBLE))[0]
        assert abs(energy(positions @ rotation.T) - energy(positions)) < 1e-11
        _, forces, _, _ = model(
            {"z": z, "pos": positions, "cell": cell,
             "edge_index": edge_index, "edge_shift": edge_shift},
            training=False, compute_stress=False,
        )
        shifted = positions.clone()
        shifted[0, 1] += 1e-6
        minus = positions.clone()
        minus[0, 1] -= 1e-6
        numerical = -(energy(shifted) - energy(minus)) / 2e-6
        assert float(forces[0, 1]) == pytest.approx(numerical, abs=1e-7)

    def test_shell_pair_scale_is_exempt_from_weight_decay(self):
        model = _model(shell_pair_channels=4)
        groups = get_muon_param_groups(model, learning_rate=1e-3, weight_decay=0.1)
        no_decay = {id(p) for g in groups if g["weight_decay"] == 0.0 for p in g["params"]}
        assert id(model.layers[0].shell_pair_scale) in no_decay
        assert id(model.layers[0].layer_scale) in no_decay


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


class TestVersion12Migration:
    def test_older_checkpoints_pin_all_three_settings(self):
        for version in (9, 10, 11):
            config = migrated_model_config(
                {"architecture": "mtace_v2", "architecture_version": version,
                 "model_config": {"num_radial": 8}}
            )
            assert config["mamba_decay_activation"] == "heavy_tail"
            assert config["cutoff_switch"] == "c2"
            assert config["gate_norm_scope"] == "gate"
        current = migrated_model_config(
            {"architecture": "mtace_v2", "architecture_version": 12,
             "model_config": {"num_radial": 8}}
        )
        assert "mamba_decay_activation" not in current

    def test_the_migration_is_load_bearing(self):
        """The v12 defaults genuinely change a quintic Mamba-3 model's energy."""

        new = _model()
        old = _model(
            mamba_decay_activation="heavy_tail", cutoff_switch="c2", gate_norm_scope="gate"
        )
        old.load_state_dict(new.state_dict())
        positions, z, cell, edge_index, edge_shift = _cluster()
        e_new = new.atomic_energies(z, positions, cell, edge_index, edge_shift).sum()
        e_old = old.atomic_energies(z, positions, cell, edge_index, edge_shift).sum()
        assert abs(float(e_new - e_old)) > 1e-8


# ---------------------------------------------------------------------------
# Cross-shell mixing diagnostic
# ---------------------------------------------------------------------------


class TestMixingFraction:
    def _report(self, **overrides):
        model = _model(layer_scale_init=1.0, **overrides)
        positions, z, cell, edge_index, edge_shift = _cluster()
        return model.gate_shell_dependence(z, positions, cell, edge_index, edge_shift)[0]

    def test_identity_has_shell_dependent_gates_but_no_mixing(self):
        """Why the residual fraction alone cannot test the mixer."""

        report = self._report(mixer_type="identity")
        assert report["mixing_fraction"] == 0.0
        assert report["residual_fraction"] > 0.01

    @pytest.mark.parametrize("mixer", ["mamba", "attention", "mlp"])
    def test_genuine_mixers_mix(self, mixer):
        assert self._report(mixer_type=mixer)["mixing_fraction"] > 1e-4

    def test_dense_has_no_length_one_form(self):
        assert math.isnan(self._report(mixer_type="dense")["mixing_fraction"])

    def test_screened_and_path_weight_variants_are_supported(self):
        assert self._report(decay_mode="screening")["mixing_fraction"] > 1e-4
        report = self._report(coupling_mode="path_weights", coupling_channels=4)
        assert report["mixing_fraction"] > 1e-4

    def test_isolated_mamba3_token_is_the_diagonal_of_the_scan(self):
        """For a length-one sequence the mixer is purely per-token."""

        layer = _model().layers[0]
        controls = torch.randn(2, 1, 16, dtype=DOUBLE)
        invariants = torch.randn(2, layer.node_invariant_dim, dtype=DOUBLE)
        torch.testing.assert_close(
            layer._isolated_mixer_states(controls, invariants),
            layer._mixer_states(controls, False, invariants),
        )


# ---------------------------------------------------------------------------
# Kernel spectroscopy measures the Jacobian
# ---------------------------------------------------------------------------


def test_spectroscopy_kernel_is_the_frobenius_norm_of_the_jacobian():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "mixer_kernel_spectroscopy.py"
    spec = importlib.util.spec_from_file_location("spectroscopy", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    layer = tool.build("attention", 6).layers[0]
    width = layer.node_context.out_features
    kernel = tool.mixer_kernel(layer, 3, 6, width, seed=5)

    torch.manual_seed(5)
    controls = torch.randn(3, 6, width, dtype=DOUBLE)
    expected = np.zeros((6, 6))
    for atom in range(3):
        jacobian = torch.autograd.functional.jacobian(
            lambda u: layer._mixer_states(u[None], False)[0] - u, controls[atom]
        )
        expected += jacobian.square().sum(dim=(1, 3)).sqrt().numpy()
    np.testing.assert_allclose(kernel, expected / 3, rtol=1e-10, atol=1e-14)
    assert not np.any(tool.mixer_kernel(tool.build("identity", 6).layers[0], 3, 6, width))
