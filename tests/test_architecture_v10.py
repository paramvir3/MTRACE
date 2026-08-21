"""Architecture-v10 tests: derivative-order contract and homogeneous invariants.

Every assertion here encodes a physical or mathematical statement that the
manuscript makes, so a regression in the code shows up as a failing claim.
"""

from __future__ import annotations

import math

import pytest
import torch

from mtace.physics import CompactRadialShellBasis, cardinal_bspline
from mtace.model import MambaACEV2


DOUBLE = torch.float64


def _dimer_energy(model, distance, z, cell, edge_index, edge_shift):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=DOUBLE
    )
    return model.atomic_energies(z, positions, cell, edge_index, edge_shift).sum().item()


def _one_sided_third_derivatives(function, point, step):
    left = (
        function(point)
        - 3.0 * function(point - step)
        + 3.0 * function(point - 2.0 * step)
        - function(point - 3.0 * step)
    ) / step**3
    right = (
        function(point + 3.0 * step)
        - 3.0 * function(point + 2.0 * step)
        + 3.0 * function(point + step)
        - function(point)
    ) / step**3
    return left, right


class TestCardinalBSpline:
    """The spline is the object every smoothness claim rests on."""

    @pytest.mark.parametrize(
        "degree,knots,expected",
        [
            # Exact rational values of the centered cardinal B-spline.
            (3, [0.0, 1.0, 2.0], [2.0 / 3.0, 1.0 / 6.0, 0.0]),
            (5, [0.0, 1.0, 2.0, 3.0], [11.0 / 20.0, 13.0 / 60.0, 1.0 / 120.0, 0.0]),
        ],
    )
    def test_exact_values_at_integer_arguments(self, degree, knots, expected):
        values = cardinal_bspline(torch.tensor(knots, dtype=DOUBLE), degree)
        torch.testing.assert_close(
            values, torch.tensor(expected, dtype=DOUBLE), rtol=0.0, atol=1e-15
        )

    @pytest.mark.parametrize("degree", [3, 5])
    def test_partition_of_unity_over_integer_shifts(self, degree):
        argument = torch.linspace(-1.0, 1.0, 401, dtype=DOUBLE)
        shifts = torch.arange(-6, 7, dtype=DOUBLE)
        total = cardinal_bspline(argument[:, None] - shifts[None, :], degree).sum(dim=-1)
        torch.testing.assert_close(
            total, torch.ones_like(total), rtol=0.0, atol=1e-14
        )

    @pytest.mark.parametrize("degree", [3, 5])
    def test_is_even(self, degree):
        argument = torch.linspace(0.0, 4.0, 201, dtype=DOUBLE)
        torch.testing.assert_close(
            cardinal_bspline(argument, degree),
            cardinal_bspline(-argument, degree),
            rtol=0.0,
            atol=1e-15,
        )

    def test_rejects_even_degree(self):
        with pytest.raises(ValueError, match="degree must be 3 or 5"):
            cardinal_bspline(torch.zeros(3, dtype=DOUBLE), 4)


class TestShellPartitionOfUnity:
    @pytest.mark.parametrize("degree", [3, 5])
    def test_folded_weights_sum_to_one_everywhere(self, degree):
        basis = CompactRadialShellBasis(6.0, 24, r_min=1.5, degree=degree)
        # Deliberately probe outside [r_min, r_max]: the folded partition is exact
        # for every real coordinate, which is what makes r_min > 0 safe.
        distances = torch.linspace(-2.0, 9.0, 4001, dtype=DOUBLE)
        weights = basis.dense(distances)
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones(distances.shape[0], dtype=DOUBLE),
            rtol=0.0, atol=1e-13,
        )

    @pytest.mark.parametrize("degree,support", [(3, 4), (5, 6)])
    def test_support_size(self, degree, support):
        basis = CompactRadialShellBasis(6.0, 24, r_min=1.5, degree=degree)
        assert basis.support_size == support
        indices, _ = basis(torch.tensor([3.0], dtype=DOUBLE))
        assert indices.shape[-1] == support

    def test_num_shells_must_cover_the_support(self):
        with pytest.raises(ValueError, match="at least 6"):
            CompactRadialShellBasis(6.0, 5, degree=5)


class TestDerivativeOrderContract:
    """The central architecture-v10 claim.

    A cubic B-spline is C2 but not C3, and its third-derivative jump at a knot is
    exactly 6 in the reduced coordinate, hence

        [d3 W / dr3] = 6 ((L - 1) / (r_max - r_min))^3

    in physical units.  Quintic shells remove that jump.  The distinction is not
    academic: third-order interatomic force constants -- and therefore three-phonon
    scattering and lattice thermal conductivity -- are exactly this derivative.
    """

    @pytest.mark.parametrize("num_shells,span", [(16, 4.0), (24, 4.5), (32, 6.0)])
    def test_cubic_third_derivative_jump_matches_the_analytic_value(
        self, num_shells, span
    ):
        basis = CompactRadialShellBasis(
            6.0, num_shells, r_min=6.0 - span, degree=3
        )
        spacing = span / (num_shells - 1)
        knot = (6.0 - span) + 7 * spacing
        step = spacing / 60.0
        weights = lambda x: basis.dense(torch.tensor([x], dtype=DOUBLE))[0]
        left, right = _one_sided_third_derivatives(weights, knot, step)
        measured = float((left - right).abs().max())
        predicted = 6.0 * ((num_shells - 1) / span) ** 3
        assert measured == pytest.approx(predicted, rel=1e-3)

    def test_cubic_jump_is_step_independent_and_quintic_vanishes(self):
        span, num_shells = 4.5, 24
        spacing = span / (num_shells - 1)
        knot = (6.0 - span) + 7 * spacing
        measurements = {}
        for degree in (3, 5):
            basis = CompactRadialShellBasis(
                6.0, num_shells, r_min=6.0 - span, degree=degree
            )
            weights = lambda x: basis.dense(torch.tensor([x], dtype=DOUBLE))[0]
            series = []
            for divisor in (60, 120, 240, 480):
                step = spacing / divisor
                left, right = _one_sided_third_derivatives(weights, knot, step)
                series.append(float((left - right).abs().max()))
            measurements[degree] = series

        # Cubic: a genuine discontinuity does not shrink when the stencil does.
        cubic = measurements[3]
        assert max(cubic) / min(cubic) < 1.01

        # Quintic: what is left is one-sided truncation error, which is O(step),
        # so successive refinements must halve it.
        quintic = measurements[5]
        for coarse, fine in zip(quintic, quintic[1:]):
            assert coarse / fine == pytest.approx(2.0, rel=0.05)
        assert quintic[-1] < 0.02 * cubic[-1]

    def test_total_energy_third_derivative_is_continuous_only_for_quintic(self):
        r_max, r_min, num_shells = 6.0, 1.5, 20
        spacing = (r_max - r_min) / (num_shells - 1)
        z = torch.tensor([6, 8], dtype=torch.long)
        cell = torch.eye(3, dtype=DOUBLE) * 100.0
        edge_index = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
        edge_shift = torch.zeros((2, 3), dtype=DOUBLE)

        jumps = {}
        for degree in (3, 5):
            torch.manual_seed(11)
            model = MambaACEV2(
                r_max=r_max, l_max=2, num_radial=8, hidden_dim=16, num_layers=1,
                correlation_order=3, correlation_channels=8, num_shells=num_shells,
                shell_r_min=r_min, shell_degree=degree, avg_num_neighbors=12.0,
                mamba_dim=16, mamba_d_state=8, mamba_headdim=8, readout_hidden=16,
                # The shell branch must be at full strength or the discontinuity
                # sits below the finite-difference noise floor.
                layer_scale_init=1.0, mamba_backend="torch",
            ).double().eval()
            energy = lambda d: _dimer_energy(
                model, d, z, cell, edge_index, edge_shift
            )
            series = []
            # Stop at 2.5e-4: a third difference amplifies round-off as eps/step^3,
            # so finer stencils measure float64 noise rather than the derivative.
            for step in (2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4):
                total = 0.0
                for index in (5, 7, 9, 11):
                    knot = r_min + index * spacing
                    left, right = _one_sided_third_derivatives(energy, knot, step)
                    total += abs(left - right)
                series.append(total / 4.0)
            jumps[degree] = series

        cubic, quintic = jumps[3], jumps[5]

        # Quintic: the mismatch is pure one-sided truncation error, O(step), so
        # every refinement halves it.
        for coarse, fine in zip(quintic, quintic[1:]):
            assert coarse / fine == pytest.approx(2.0, rel=0.05)

        # Cubic: truncation error decays on top of a genuine, step-independent
        # jump, so the sequence flattens out instead of halving.
        assert cubic[-2] / cubic[-1] < 1.3

        # And the floor it flattens onto is what quintic removes.
        assert cubic[-1] > 3.0 * quintic[-1]

    def test_quintic_keeps_conservative_forces_and_rotation_invariance(self):
        torch.manual_seed(3)
        model = MambaACEV2(
            r_max=6.0, l_max=2, num_radial=8, hidden_dim=16, num_layers=1,
            correlation_order=3, correlation_channels=8, num_shells=16,
            shell_r_min=1.5, shell_degree=5, avg_num_neighbors=12.0,
            mamba_dim=16, mamba_d_state=8, mamba_headdim=8, readout_hidden=16,
            mamba_backend="torch",
        ).double().eval()
        positions = torch.randn(5, 3, dtype=DOUBLE) * 1.6
        z = torch.tensor([1, 6, 8, 6, 1], dtype=torch.long)
        cell = torch.eye(3, dtype=DOUBLE) * 100.0
        senders, receivers, shifts = [], [], []
        for i in range(5):
            for j in range(5):
                if i != j and torch.linalg.norm(positions[j] - positions[i]) < 6.0:
                    senders.append(j)
                    receivers.append(i)
                    shifts.append([0.0, 0.0, 0.0])
        edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        edge_shift = torch.tensor(shifts, dtype=DOUBLE)

        _, forces, _, _ = model(
            {"z": z, "pos": positions, "cell": cell,
             "edge_index": edge_index, "edge_shift": edge_shift},
            training=False, compute_stress=False,
        )
        step = 1.0e-6
        numerical = torch.zeros_like(positions)
        for atom in range(5):
            for axis in range(3):
                shifted = positions.clone()
                shifted[atom, axis] += step
                plus = model.atomic_energies(
                    z, shifted, cell, edge_index, edge_shift
                ).sum().item()
                shifted = positions.clone()
                shifted[atom, axis] -= step
                minus = model.atomic_energies(
                    z, shifted, cell, edge_index, edge_shift
                ).sum().item()
                numerical[atom, axis] = -(plus - minus) / (2.0 * step)
        assert float((forces - numerical).abs().max()) < 1e-8

        rotation = torch.linalg.qr(torch.randn(3, 3, dtype=DOUBLE))[0]
        rotation = rotation * torch.sign(torch.det(rotation))
        reference = model.atomic_energies(
            z, positions, cell, edge_index, edge_shift
        ).sum()
        rotated = model.atomic_energies(
            z, positions @ rotation.T, cell, edge_index, edge_shift
        ).sum()
        assert float((rotated - reference).abs().detach()) < 1e-12


class TestHomogeneousInvariants:
    """The invariant vector must be dimensionally homogeneous.

    ``J`` concatenates even scalars, degree one in the ACE features, with irrep
    norms.  Squaring the norms makes that block degree two, so a global rescaling
    of the features -- a different ``avg_num_neighbors``, a retuned radial
    network, or fine-tuning on a new dataset -- moves the two blocks by different
    powers and silently changes the learned balance between scalar and angular
    information.  No single linear map can undo that.
    """

    def _block(self, mode):
        torch.manual_seed(5)
        return MambaACEV2(
            r_max=6.0, l_max=2, num_radial=8, hidden_dim=16, num_layers=1,
            correlation_order=3, correlation_channels=8, num_shells=12,
            invariant_norm=mode, mamba_dim=16, mamba_d_state=8, mamba_headdim=8,
            readout_hidden=16, mamba_backend="torch",
        ).double().eval()

    @pytest.mark.parametrize(
        "mode,scalar_power,norm_power", [("squared", 1.0, 2.0), ("homogeneous", 1.0, 1.0)]
    )
    def test_scaling_degree_of_each_invariant_block(self, mode, scalar_power, norm_power):
        model = self._block(mode)
        layer = model.layers[0]
        torch.manual_seed(1)
        tokens = torch.randn(4, 12, model.ace.irreps_correlation.dim, dtype=DOUBLE)
        # Large amplitude so the smoothing epsilon is negligible.
        tokens = tokens * 10.0
        width = layer.token_scalar_dim
        reference = layer._token_invariants(tokens)
        for scale in (2.0, 4.0):
            scaled = layer._token_invariants(tokens * scale)
            scalar_ratio = float(
                scaled[..., :width].norm() / reference[..., :width].norm()
            )
            norm_ratio = float(
                scaled[..., width:].norm() / reference[..., width:].norm()
            )
            assert scalar_ratio == pytest.approx(scale**scalar_power, rel=1e-3)
            assert norm_ratio == pytest.approx(scale**norm_power, rel=1e-3)

    def test_smoothed_norm_is_twice_differentiable_at_the_origin(self):
        from mtace.ssm import IrrepInvariantNorm

        norm = IrrepInvariantNorm("2x1o", mode="homogeneous", eps=1.0e-4).double()
        features = torch.zeros(1, 6, dtype=DOUBLE, requires_grad=True)
        value = norm(features).sum()
        gradient = torch.autograd.grad(value, features, create_graph=True)[0]
        curvature = torch.autograd.grad(gradient.sum(), features)[0]
        assert torch.isfinite(gradient).all()
        assert torch.isfinite(curvature).all()
        # n(0) = 0 exactly: an empty shell must be invisible to the mixer.
        assert float(norm(torch.zeros(1, 6, dtype=DOUBLE)).abs().max()) == 0.0

    def test_rejects_unknown_mode(self):
        from mtace.ssm import IrrepInvariantNorm

        with pytest.raises(ValueError, match="invariant_norm"):
            IrrepInvariantNorm("1x1o", mode="linear")


class TestShellPairCorrelation:
    """Banded equivariant correlation between shells k and k + d.

    The linear shell term is the only route by which tokens reach the energy and
    it is linear in T_ik, so all many-body structure has to pass through a scalar
    gate.  The banded product is a strict sub-sum of
    ``TP(A_i, A_i) = sum_{k,k'} TP(T_ik, T_ik')``, so it carries information the
    direct ACE path does not have whenever L exceeds the ACE radial channel count.
    """

    def _model(self, **overrides):
        torch.manual_seed(7)
        settings = dict(
            r_max=6.0, l_max=2, num_radial=8, hidden_dim=16, num_layers=1,
            correlation_order=3, correlation_channels=8, num_shells=14,
            shell_r_min=1.5, shell_degree=5, avg_num_neighbors=12.0,
            invariant_norm="homogeneous", mamba_dim=16, mamba_d_state=8,
            mamba_headdim=8, readout_hidden=16, mamba_backend="torch",
        )
        settings.update(overrides)
        return MambaACEV2(**settings).double().eval()

    def _cluster(self, model):
        torch.manual_seed(2)
        positions = torch.randn(6, 3, dtype=DOUBLE) * 1.7
        z = torch.tensor([1, 6, 8, 1, 6, 8], dtype=torch.long)
        cell = torch.eye(3, dtype=DOUBLE) * 100.0
        senders, receivers, shifts = [], [], []
        for i in range(6):
            for j in range(6):
                if i != j and torch.linalg.norm(positions[j] - positions[i]) < 6.0:
                    senders.append(j)
                    receivers.append(i)
                    shifts.append([0.0, 0.0, 0.0])
        return (
            positions, z, cell,
            torch.tensor([senders, receivers], dtype=torch.long),
            torch.tensor(shifts, dtype=DOUBLE),
        )

    def test_term_changes_the_energy(self):
        with_pair = self._model(shell_pair_channels=4, shell_pair_width=2)
        without = self._model(shell_pair_channels=0)
        positions, z, cell, edge_index, edge_shift = self._cluster(with_pair)
        on = with_pair.atomic_energies(z, positions, cell, edge_index, edge_shift).sum()
        off = without.atomic_energies(z, positions, cell, edge_index, edge_shift).sum()
        assert float((on - off).abs()) > 1e-6

    def test_disabled_by_default(self):
        model = self._model()
        assert model.layers[0].shell_pair_products is None

    def test_preserves_every_symmetry(self):
        model = self._model(shell_pair_channels=4, shell_pair_width=2)
        positions, z, cell, edge_index, edge_shift = self._cluster(model)
        energy = lambda p, zz=z, ei=edge_index, es=edge_shift: model.atomic_energies(
            zz, p, cell, ei, es
        ).sum()
        reference = energy(positions)
        rotation = torch.linalg.qr(torch.randn(3, 3, dtype=DOUBLE))[0]
        rotation = rotation * torch.sign(torch.det(rotation))
        assert float((energy(positions + torch.tensor([1.0, -2.0, 3.0], dtype=DOUBLE))
                      - reference).abs()) < 1e-12
        assert float((energy(positions @ rotation.T) - reference).abs()) < 1e-12
        assert float((energy(-positions) - reference).abs()) < 1e-12

        order = torch.randperm(6)
        senders, receivers, shifts = [], [], []
        permuted = positions[order]
        for i in range(6):
            for j in range(6):
                if i != j and torch.linalg.norm(permuted[j] - permuted[i]) < 6.0:
                    senders.append(j)
                    receivers.append(i)
                    shifts.append([0.0, 0.0, 0.0])
        assert float((energy(
            permuted, z[order],
            torch.tensor([senders, receivers], dtype=torch.long),
            torch.tensor(shifts, dtype=DOUBLE),
        ) - reference).abs()) < 1e-12

    def test_forces_stay_conservative(self):
        model = self._model(shell_pair_channels=4, shell_pair_width=2)
        positions, z, cell, edge_index, edge_shift = self._cluster(model)
        _, forces, _, _ = model(
            {"z": z, "pos": positions, "cell": cell,
             "edge_index": edge_index, "edge_shift": edge_shift},
            training=False, compute_stress=False,
        )
        step = 1.0e-6
        numerical = torch.zeros_like(positions)
        for atom in range(6):
            for axis in range(3):
                shifted = positions.clone()
                shifted[atom, axis] += step
                plus = model.atomic_energies(
                    z, shifted, cell, edge_index, edge_shift
                ).sum().item()
                shifted = positions.clone()
                shifted[atom, axis] -= step
                minus = model.atomic_energies(
                    z, shifted, cell, edge_index, edge_shift
                ).sum().item()
                numerical[atom, axis] = -(plus - minus) / (2.0 * step)
        assert float((forces - numerical).abs().max()) < 1e-8

    def test_width_beyond_the_sequence_is_ignored_without_error(self):
        model = self._model(num_shells=6, shell_pair_channels=2, shell_pair_width=9)
        positions, z, cell, edge_index, edge_shift = self._cluster(model)
        energy = model.atomic_energies(z, positions, cell, edge_index, edge_shift).sum()
        assert torch.isfinite(energy)

    def test_rejects_negative_settings(self):
        with pytest.raises(ValueError, match="shell_pair_channels"):
            self._model(shell_pair_channels=-1)
        with pytest.raises(ValueError, match="shell_pair_width"):
            self._model(shell_pair_channels=2, shell_pair_width=-1)


class TestEquivariantMuon:
    """Muon's spectral normalization is defined per matrix.

    ``o3.Linear`` and ``o3.FullyConnectedTensorProduct`` flatten every path weight
    into one vector, so a shape test sends them to AdamW even though each path is
    a matrix between two representation spaces.  Orthogonalizing each path block
    separately is the correct generalization; merging blocks from different irreps
    into one matrix would be meaningless.
    """

    def test_annotation_matches_the_e3nn_weight_views(self):
        from e3nn import o3
        from mtace.optim import annotate_equivariant_blocks

        linear = o3.Linear(o3.Irreps("8x0e+4x1o"), o3.Irreps("6x0e+2x1o"))
        assert annotate_equivariant_blocks(linear) == 1
        blocks = linear.weight._irrep_blocks
        views = [tuple(view.shape) for view in linear.weight_views()]
        assert [(rows, columns) for _, rows, columns in blocks] == views
        # Offsets must tile the flat weight in order.
        expected = 0
        for (offset, rows, columns), shape in zip(blocks, views):
            assert offset == expected
            expected += rows * columns
        assert expected == linear.weight.numel()

    def test_tensor_product_paths_are_matricized(self):
        from e3nn import o3
        from mtace.optim import annotate_equivariant_blocks

        product = o3.FullyConnectedTensorProduct(
            o3.Irreps("4x0e+2x1o"), o3.Irreps("4x0e+2x1o"), o3.Irreps("4x0e+2x1o")
        )
        annotate_equivariant_blocks(product)
        blocks = product.weight._irrep_blocks
        for (_, rows, columns), view in zip(blocks, product.weight_views()):
            shape = tuple(view.shape)
            assert columns == shape[-1]
            assert rows == math.prod(shape[:-1])

    def test_blockwise_update_matches_an_independent_reference(self):
        from e3nn import o3
        from mtace.optim import (
            adjusted_muon_learning_rate,
            annotate_equivariant_blocks,
            muon_update_blockwise,
            zeropower_via_newton_schulz5,
        )

        torch.manual_seed(0)
        linear = o3.Linear(o3.Irreps("8x0e+4x1o"), o3.Irreps("6x0e+2x1o")).double()
        annotate_equivariant_blocks(linear)
        blocks = linear.weight._irrep_blocks
        gradient = torch.randn_like(linear.weight)

        buffer = torch.zeros_like(gradient)
        update = muon_update_blockwise(
            gradient, buffer, blocks, momentum=0.95, nesterov=True, learning_rate=0.01
        )

        reference_buffer = torch.zeros_like(gradient)
        reference_buffer.lerp_(gradient, 0.05)
        direction = gradient.lerp(reference_buffer, 0.95)
        reference = torch.zeros_like(gradient)
        for offset, rows, columns in blocks:
            block = direction[offset : offset + rows * columns].view(rows, columns)
            reference[offset : offset + rows * columns] = (
                zeropower_via_newton_schulz5(block).reshape(-1)
                * adjusted_muon_learning_rate(0.01, (rows, columns))
            )
        torch.testing.assert_close(update, reference, rtol=0.0, atol=1e-14)

    def test_equivariant_mode_covers_more_of_the_model_than_hidden(self):
        from mtace.optim import build_optimizer

        torch.manual_seed(0)
        model = MambaACEV2(
            r_max=6.0, l_max=2, num_radial=8, hidden_dim=32, num_layers=1,
            correlation_order=3, correlation_channels=8, num_shells=12,
            mamba_dim=32, mamba_d_state=8, mamba_headdim=8, readout_hidden=32,
            mamba_backend="torch",
        )
        covered = {}
        for mode in ("hidden", "equivariant"):
            optimizer = build_optimizer(
                model,
                {"optimizer": "muon", "learning_rate": 1e-3,
                 "muon_parameter_mode": mode},
            )
            covered[mode] = sum(
                parameter.numel()
                for group in optimizer.param_groups
                if group.get("use_muon")
                for parameter in group["params"]
            )
        assert covered["equivariant"] > covered["hidden"]

    def test_every_parameter_is_still_assigned_exactly_once(self):
        from mtace.optim import build_optimizer

        torch.manual_seed(0)
        model = MambaACEV2(
            r_max=6.0, l_max=2, num_radial=8, hidden_dim=32, num_layers=1,
            correlation_order=3, correlation_channels=8, num_shells=12,
            mamba_dim=32, mamba_d_state=8, mamba_headdim=8, readout_hidden=32,
            mamba_backend="torch",
        )
        optimizer = build_optimizer(
            model,
            {"optimizer": "muon", "learning_rate": 1e-3,
             "muon_parameter_mode": "equivariant"},
        )
        grouped = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        expected = [id(p) for p in model.parameters() if p.requires_grad]
        assert len(grouped) == len(set(grouped))
        assert set(grouped) == set(expected)

    def test_a_blockwise_step_actually_moves_the_weights(self):
        from e3nn import o3
        from mtace.optim import MuonWithAuxAdamW, annotate_equivariant_blocks

        torch.manual_seed(0)
        linear = o3.Linear(o3.Irreps("8x0e+4x1o"), o3.Irreps("6x0e+2x1o")).double()
        annotate_equivariant_blocks(linear)
        optimizer = MuonWithAuxAdamW(
            [{"params": [linear.weight], "use_muon": True, "lr": 1e-2}]
        )
        before = linear.weight.detach().clone()
        linear.weight.grad = torch.randn_like(linear.weight)
        optimizer.step()
        assert float((linear.weight.detach() - before).abs().max()) > 0.0
        assert torch.isfinite(linear.weight).all()


class TestExponentialMovingAverage:
    def test_matches_the_warmed_up_recursion(self):
        from mtace.optim import ExponentialMovingAverage

        torch.manual_seed(0)
        net = torch.nn.Linear(4, 3).double()
        ema = ExponentialMovingAverage(net, decay=0.9)
        manual = net.weight.detach().clone()
        for step in range(5):
            with torch.no_grad():
                net.weight.add_(torch.randn_like(net.weight) * 0.1)
            ema.update(net)
            decay = min(0.9, (1.0 + step + 1) / (10.0 + step + 1))
            manual = decay * manual + (1.0 - decay) * net.weight.detach()
        torch.testing.assert_close(
            ema.shadow["weight"], manual, rtol=0.0, atol=1e-14
        )

    def test_store_and_restore_are_exact(self):
        from mtace.optim import ExponentialMovingAverage

        torch.manual_seed(0)
        net = torch.nn.Linear(4, 3).double()
        ema = ExponentialMovingAverage(net, decay=0.9)
        with torch.no_grad():
            net.weight.add_(1.0)
        ema.update(net)
        raw = net.weight.detach().clone()
        ema.store(net)
        torch.testing.assert_close(
            net.weight.detach(), ema.shadow["weight"], rtol=0.0, atol=0.0
        )
        ema.restore(net)
        torch.testing.assert_close(net.weight.detach(), raw, rtol=0.0, atol=0.0)

    def test_state_round_trip(self):
        from mtace.optim import ExponentialMovingAverage

        torch.manual_seed(0)
        net = torch.nn.Linear(4, 3).double()
        first = ExponentialMovingAverage(net, decay=0.9)
        with torch.no_grad():
            net.weight.add_(0.5)
        first.update(net)
        second = ExponentialMovingAverage(net, decay=0.9)
        second.load_state_dict(first.state_dict())
        assert second.step == first.step
        torch.testing.assert_close(
            second.shadow["weight"], first.shadow["weight"], rtol=0.0, atol=0.0
        )

    def test_rejects_degenerate_decay(self):
        from mtace.optim import ExponentialMovingAverage

        net = torch.nn.Linear(2, 2)
        for decay in (0.0, 1.0, -0.5):
            with pytest.raises(ValueError, match="decay"):
                ExponentialMovingAverage(net, decay=decay)


def _labelled_frames(count: int = 8, seed: int = 0):
    """Self-contained labelled frames.

    Deliberately not read from ``examples/``: the repository excludes ``*.extxyz``
    from git, so a test that reads one passes locally and fails everywhere else.
    Two chemical species and a rattled lattice give a full-rank species-count
    matrix, which is what exercises the least-squares branch of
    ``target_statistics``.
    """

    import numpy as np
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    generator = np.random.RandomState(seed)
    lattice = 3.6
    frames = []
    for index in range(count):
        # Vary the composition so the species-count matrix has full column rank.
        symbols = "Cu3Au" if index % 2 else "Cu2Au2"
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
        ) * lattice
        positions = positions + generator.normal(scale=0.05, size=positions.shape)
        atoms = Atoms(
            symbols, positions=positions, cell=np.eye(3) * lattice, pbc=True
        )
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=float(generator.normal(scale=1.0) - 5.0 * len(atoms)),
            forces=generator.normal(scale=0.2, size=(len(atoms), 3)),
            stress=generator.normal(scale=1.0e-3, size=6),
        )
        frames.append(atoms)
    return frames


class TestDatasetStatistics:
    def test_target_statistics_are_positive_and_named(self):
        from mtace.data import target_statistics

        statistics = target_statistics(_labelled_frames())
        for key in (
            "energy_ev_per_atom",
            "forces_ev_per_angstrom",
            "stress_ev_per_angstrom3",
        ):
            assert statistics[key] > 0.0
            assert math.isfinite(statistics[key])

    def test_energy_scale_ignores_the_per_species_reference(self):
        """The energy scale must measure the learnable residual, not the offset.

        Adding a constant per-species reference energy shifts every total energy
        but changes nothing a potential has to learn, so the reported scale must
        be invariant to it.
        """

        from mtace.data import target_statistics

        frames = _labelled_frames()
        baseline = target_statistics(frames)["energy_ev_per_atom"]

        from ase.calculators.singlepoint import SinglePointCalculator

        shifted = []
        for atoms in frames:
            offsets = {29: -7.5, 79: 3.25}
            extra = sum(offsets[int(number)] for number in atoms.numbers)
            clone = atoms.copy()
            clone.calc = SinglePointCalculator(
                clone,
                energy=atoms.get_potential_energy() + extra,
                forces=atoms.get_forces(apply_constraint=False),
                stress=atoms.get_stress(voigt=True),
            )
            shifted.append(clone)
        assert target_statistics(shifted)["energy_ev_per_atom"] == pytest.approx(
            baseline, rel=1e-9
        )

    def test_minimum_edge_distance_matches_a_direct_scan(self):
        from ase.neighborlist import neighbor_list
        from mtace.data import minimum_edge_distance

        frames = _labelled_frames()
        shortest = min(
            float(neighbor_list("d", atoms, 6.0).min()) for atoms in frames
        )
        assert minimum_edge_distance(frames, 6.0) == pytest.approx(shortest)


class TestMixerEmittedPathWeights:
    """The mixer emits the weights of the equivariant map, not just a gate.

    With a fixed value map W_V and a scalar gate, the mixer can amplify an l = 2
    channel but never change how channels within an l combine.  In
    ``coupling_mode="path_weights"`` the mixer emits the weights of the map
    itself, which is the NequIP radial-MLP construction driven by the mixer state
    instead of by r.  Equivariance is exact because the emitted weights are O(3)
    invariants.
    """

    def _model(self, **overrides):
        torch.manual_seed(13)
        settings = dict(
            r_max=6.0, l_max=2, num_radial=8, hidden_dim=16, num_layers=2,
            correlation_order=3, correlation_channels=8, num_shells=14,
            shell_r_min=1.5, shell_degree=5, avg_num_neighbors=12.0,
            invariant_norm="homogeneous", mamba_dim=16, mamba_d_state=8,
            mamba_headdim=8, readout_hidden=16, mamba_backend="torch",
        )
        settings.update(overrides)
        return MambaACEV2(**settings).double().eval()

    def _cluster(self):
        torch.manual_seed(2)
        positions = torch.randn(6, 3, dtype=DOUBLE) * 1.7
        z = torch.tensor([1, 6, 8, 1, 6, 8], dtype=torch.long)
        cell = torch.eye(3, dtype=DOUBLE) * 100.0
        senders, receivers, shifts = [], [], []
        for i in range(6):
            for j in range(6):
                if i != j and torch.linalg.norm(positions[j] - positions[i]) < 6.0:
                    senders.append(j)
                    receivers.append(i)
                    shifts.append([0.0, 0.0, 0.0])
        return (
            positions, z, cell,
            torch.tensor([senders, receivers], dtype=torch.long),
            torch.tensor(shifts, dtype=DOUBLE),
        )

    def test_gate_is_the_default(self):
        assert self._model().layers[0].coupling_mode == "gate"
        assert self._model().layers[0].coupling_map is None

    def test_preserves_every_symmetry(self):
        model = self._model(coupling_mode="path_weights", coupling_channels=6)
        positions, z, cell, edge_index, edge_shift = self._cluster()
        energy = lambda p, zz=z, ei=edge_index, es=edge_shift: model.atomic_energies(
            zz, p, cell, ei, es
        ).sum()
        reference = energy(positions)
        rotation = torch.linalg.qr(torch.randn(3, 3, dtype=DOUBLE))[0]
        rotation = rotation * torch.sign(torch.det(rotation))
        assert float((energy(positions + torch.tensor([1.0, -2.0, 3.0], dtype=DOUBLE))
                      - reference).abs()) < 1e-12
        assert float((energy(positions @ rotation.T) - reference).abs()) < 1e-12
        assert float((energy(-positions) - reference).abs()) < 1e-12

        order = torch.randperm(6)
        permuted = positions[order]
        senders, receivers, shifts = [], [], []
        for i in range(6):
            for j in range(6):
                if i != j and torch.linalg.norm(permuted[j] - permuted[i]) < 6.0:
                    senders.append(j)
                    receivers.append(i)
                    shifts.append([0.0, 0.0, 0.0])
        assert float((energy(
            permuted, z[order],
            torch.tensor([senders, receivers], dtype=torch.long),
            torch.tensor(shifts, dtype=DOUBLE),
        ) - reference).abs()) < 1e-12

    def test_forces_stay_conservative(self):
        model = self._model(coupling_mode="path_weights", coupling_channels=6)
        positions, z, cell, edge_index, edge_shift = self._cluster()
        _, forces, _, _ = model(
            {"z": z, "pos": positions, "cell": cell,
             "edge_index": edge_index, "edge_shift": edge_shift},
            training=False, compute_stress=False,
        )
        step = 1.0e-6
        numerical = torch.zeros_like(positions)
        for atom in range(6):
            for axis in range(3):
                shifted = positions.clone()
                shifted[atom, axis] += step
                plus = model.atomic_energies(
                    z, shifted, cell, edge_index, edge_shift
                ).sum().item()
                shifted = positions.clone()
                shifted[atom, axis] -= step
                minus = model.atomic_energies(
                    z, shifted, cell, edge_index, edge_shift
                ).sum().item()
                numerical[atom, axis] = -(plus - minus) / (2.0 * step)
        assert float((forces - numerical).abs().max()) < 1e-8

    def test_removing_the_data_dependence_collapses_onto_the_ace_path(self):
        """The central identity, checked rather than assumed.

        Because ``sum_k T_ik = A_i``, a map whose weights do not vary across
        shells reduces exactly to a fixed equivariant map of the ACE density,
        which the direct path already contains.  Zeroing the state-dependent part
        of the emitted weights must therefore drive the residual to zero.
        """

        model = self._model(coupling_mode="path_weights", coupling_channels=6)
        positions, z, cell, edge_index, edge_shift = self._cluster()
        live = model.gate_shell_dependence(
            z, positions, cell, edge_index, edge_shift
        )[0]["residual_fraction"]
        assert live > 1e-3

        with torch.no_grad():
            for layer in model.layers:
                layer.coupling_path.weight.zero_()
        collapsed = model.gate_shell_dependence(
            z, positions, cell, edge_index, edge_shift
        )[0]["residual_fraction"]
        assert collapsed < 1e-12

    def test_emitted_weight_tensor_stays_small(self):
        """Per-sample weights for the full map would be prohibitive.

        At the production irreps the token -> node map has 2688 weights, i.e.
        328 MiB of per-sample weights for 1000 atoms at L = 32.  The reduced map
        must keep that within a few tens of MiB.
        """

        model = self._model(coupling_mode="path_weights", coupling_channels=8)
        numel = model.layers[0].coupling_map.weight_numel
        assert numel < 400
        megabytes = 1000 * 32 * numel * 4 / 1024**2
        assert megabytes < 64.0

    def test_rejects_bad_settings(self):
        with pytest.raises(ValueError, match="coupling_mode"):
            self._model(coupling_mode="attention")
        with pytest.raises(ValueError, match="coupling_channels"):
            self._model(coupling_mode="path_weights", coupling_channels=0)


class TestShellResolutionWarning:
    def test_calculator_warns_once_below_shell_r_min(self, tmp_path):
        import warnings as warnings_module
        from ase import Atoms
        from mtace.calculator import MambaACECalculator
        from mtace.checkpoint import save_checkpoint

        torch.manual_seed(0)
        config = dict(
            r_max=6.0, l_max=1, num_radial=6, hidden_dim=8, num_layers=1,
            correlation_order=2, correlation_channels=4, num_shells=10,
            shell_r_min=2.0, avg_num_neighbors=6.0, mamba_dim=8,
            mamba_d_state=8, mamba_headdim=8, readout_hidden=8,
            mamba_backend="torch",
        )
        model = MambaACEV2(**config)
        path = tmp_path / "model.pt"
        save_checkpoint(
            path, model, config, atomic_energies={1: 0.0}, atomic_numbers=[1]
        )
        calculator = MambaACECalculator(path, device="cpu")

        # 1.0 A is well inside shell_r_min = 2.0 A.
        close = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        close.calc = calculator
        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            close.get_potential_energy()
        messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
        assert any("shell_r_min" in message for message in messages)

        # The warning is issued once, not on every force evaluation of an MD run.
        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            close.calc.results = {}
            close.get_potential_energy()
        assert not [
            w for w in caught
            if w.category is RuntimeWarning and "shell_r_min" in str(w.message)
        ]

    def test_no_warning_when_the_geometry_is_safe(self, tmp_path):
        import warnings as warnings_module
        from ase import Atoms
        from mtace.calculator import MambaACECalculator
        from mtace.checkpoint import save_checkpoint

        torch.manual_seed(0)
        config = dict(
            r_max=6.0, l_max=1, num_radial=6, hidden_dim=8, num_layers=1,
            correlation_order=2, correlation_channels=4, num_shells=10,
            shell_r_min=2.0, avg_num_neighbors=6.0, mamba_dim=8,
            mamba_d_state=8, mamba_headdim=8, readout_hidden=8,
            mamba_backend="torch",
        )
        model = MambaACEV2(**config)
        path = tmp_path / "model.pt"
        save_checkpoint(
            path, model, config, atomic_energies={1: 0.0}, atomic_numbers=[1]
        )
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        atoms.calc = MambaACECalculator(path, device="cpu")
        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            atoms.get_potential_energy()
        assert not [
            w for w in caught
            if w.category is RuntimeWarning and "shell_r_min" in str(w.message)
        ]


class TestVersion10CheckpointCompatibility:
    """A v9 checkpoint must keep its trained function under v10 code.

    Every architecture-v10 setting defaults to its v9 behaviour, so no migration
    block is needed -- but that is a property of the defaults, not a guarantee,
    and it silently breaks the moment someone changes one.  This pins it.
    """

    def test_v9_defaults_are_the_v10_defaults_for_new_settings(self):
        from mtace.checkpoint import migrated_model_config

        stored = {
            "architecture": "mtace_v2",
            "architecture_version": 9,
            "model_config": {
                "r_max": 6.0, "l_max": 1, "num_radial": 6, "hidden_dim": 8,
                "num_layers": 1, "correlation_order": 2, "num_shells": 10,
                "mamba_mimo_rank": 4,
            },
        }
        migrated = migrated_model_config(stored)
        # A v9 checkpoint predates all four v10 settings, so the constructor
        # defaults must reproduce v9 behaviour rather than the new behaviour.
        for key, forbidden in (
            ("shell_degree", 5),
            ("invariant_norm", "homogeneous"),
            ("coupling_mode", "path_weights"),
        ):
            assert migrated.get(key, None) != forbidden
        assert migrated.get("shell_pair_channels", 0) == 0

        model = MambaACEV2(**migrated)
        layer = model.layers[0]
        assert model.ace.shell_degree == 3
        assert layer.invariant_norm == "squared"
        assert layer.coupling_mode == "gate"
        assert layer.shell_pair_products is None

    def test_current_architecture_version_is_recorded(self, tmp_path):
        from mtace.checkpoint import load_checkpoint, save_checkpoint

        torch.manual_seed(0)
        config = dict(
            r_max=6.0, l_max=1, num_radial=6, hidden_dim=8, num_layers=1,
            correlation_order=2, correlation_channels=4, num_shells=10,
            mamba_dim=8, mamba_d_state=8, mamba_headdim=8, readout_hidden=8,
            mamba_backend="torch",
        )
        model = MambaACEV2(**config)
        path = tmp_path / "model.pt"
        save_checkpoint(path, model, config, atomic_energies={1: 0.0},
                        atomic_numbers=[1])
        payload = load_checkpoint(path)
        assert payload["architecture_version"] == MambaACEV2.architecture_version
        # v11 added the per-layer mixer schedule and expert routing.  Both are
        # inert at their defaults, so this pin moves but no v10 behaviour does;
        # tests/test_mixer_schedule.py checks a v10 config still restores exactly.
        assert payload["architecture_version"] == 11
