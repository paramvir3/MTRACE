"""Architecture-v11: raising the polynomial degree of the shell branch.

The default coupling is provably limited.  With
``dh_{c l m} = g_{c l}(J) (W_V T)_{c l m}`` the equivariant Jacobian is nonzero
only for ``l_in == l_out``: the gate cannot transfer angular momentum, and the
update is degree one in ``T``.  Together with ``sum_k T_ik = A_i`` that makes the
shell branch a body-order-two model in the shell variables.

Three mechanisms lift it, and each assertion here is one of the mathematical
claims that justify them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from e3nn import o3

from mtace.model import MambaACEV2

DOUBLE = torch.float64
R_MAX = 6.0


def _model(**overrides):
    torch.manual_seed(9)
    settings = dict(
        r_max=R_MAX, l_max=2, num_radial=8, hidden_dim=16, num_layers=2,
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
    senders, receivers, shifts = [], [], []
    for i in range(count):
        for j in range(count):
            if i != j and torch.linalg.norm(positions[j] - positions[i]) < R_MAX:
                senders.append(j)
                receivers.append(i)
                shifts.append([0.0, 0.0, 0.0])
    return (
        positions, z, cell,
        torch.tensor([senders, receivers], dtype=torch.long),
        torch.tensor(shifts, dtype=DOUBLE),
    )


def _assert_symmetries(model, tolerance=1e-11):
    positions, z, cell, edge_index, edge_shift = _cluster()
    energy = lambda p, zz=z, ei=edge_index, es=edge_shift: float(
        model.atomic_energies(zz, p, cell, ei, es).sum()
    )
    reference = energy(positions)
    rotation = torch.linalg.qr(torch.randn(3, 3, dtype=DOUBLE))[0]
    rotation = rotation * torch.sign(torch.det(rotation))
    assert abs(energy(positions + torch.tensor([1.0, -2.0, 3.0], dtype=DOUBLE))
               - reference) < tolerance
    assert abs(energy(positions @ rotation.T) - reference) < tolerance
    assert abs(energy(-positions) - reference) < tolerance

    order = torch.randperm(positions.shape[0])
    permuted = positions[order]
    senders, receivers, shifts = [], [], []
    for i in range(positions.shape[0]):
        for j in range(positions.shape[0]):
            if i != j and torch.linalg.norm(permuted[j] - permuted[i]) < R_MAX:
                senders.append(j)
                receivers.append(i)
                shifts.append([0.0, 0.0, 0.0])
    assert abs(energy(
        permuted, z[order],
        torch.tensor([senders, receivers], dtype=torch.long),
        torch.tensor(shifts, dtype=DOUBLE),
    ) - reference) < tolerance


def _assert_conservative(model, tolerance=1e-8):
    positions, z, cell, edge_index, edge_shift = _cluster()
    _, forces, _, _ = model(
        {"z": z, "pos": positions, "cell": cell,
         "edge_index": edge_index, "edge_shift": edge_shift},
        training=False, compute_stress=False,
    )
    step = 1.0e-6
    numerical = torch.zeros_like(positions)
    for atom in range(positions.shape[0]):
        for axis in range(3):
            shifted = positions.clone()
            shifted[atom, axis] += step
            plus = float(model.atomic_energies(
                z, shifted, cell, edge_index, edge_shift).sum())
            shifted = positions.clone()
            shifted[atom, axis] -= step
            minus = float(model.atomic_energies(
                z, shifted, cell, edge_index, edge_shift).sum())
            numerical[atom, axis] = -(plus - minus) / (2.0 * step)
    assert float((forces - numerical).abs().max()) < tolerance


class TestShellAlignmentInvariants:
    """The mixer was fed only the diagonal of the shell Gram matrix.

    ``||T_ik||^2`` is ``G_kk`` for ``G_{kk'} = sum_m T_{ikclm} T_{ik'clm}``.  The
    off-diagonal band is an independent degree of freedom: measured on the five
    CsPbI3 polymorphs the adjacent-shell alignment spans [-0.16, 1.00] with
    standard deviation 0.43, so Cauchy-Schwarz is nowhere near saturated.
    """

    def test_disabled_by_default(self):
        assert _model().layers[0].invariant_overlap_width == 0

    def test_cosines_are_o3_invariant(self):
        model = _model(invariant_overlap_width=3)
        layer = model.layers[0]
        torch.manual_seed(1)
        tokens = torch.randn(3, 12, model.ace.irreps_correlation.dim, dtype=DOUBLE)
        rotation = o3.rand_matrix().to(DOUBLE)
        wigner = model.ace.irreps_correlation.D_from_matrix(rotation)
        torch.testing.assert_close(
            layer._shell_alignment(tokens),
            layer._shell_alignment(tokens @ wigner.T),
            rtol=0.0, atol=1e-13,
        )

    def test_cosines_are_bounded_and_scale_free(self):
        model = _model(invariant_overlap_width=2)
        layer = model.layers[0]
        torch.manual_seed(1)
        tokens = torch.randn(3, 12, model.ace.irreps_correlation.dim, dtype=DOUBLE)
        cosines = layer._shell_alignment(tokens)
        assert float(cosines.min()) >= -1.0 - 1e-12
        assert float(cosines.max()) <= 1.0 + 1e-12
        # Homogeneous of degree zero: an angle, not a magnitude.  This is why it
        # composes cleanly with the degree-one norms rather than reintroducing
        # the degree-one/degree-two clash.
        rescaled = layer._shell_alignment(tokens * 7.3)
        assert float((rescaled - cosines).abs().max()) < 1e-4

    def test_empty_shell_gives_zero_with_finite_curvature(self):
        model = _model(invariant_overlap_width=2)
        layer = model.layers[0]
        width = model.ace.irreps_correlation.dim
        zeros = torch.zeros(1, 12, width, dtype=DOUBLE)
        assert float(layer._shell_alignment(zeros).abs().max()) == 0.0
        # Force training differentiates twice; the eps-regularised denominator
        # must keep both derivatives finite at the origin.
        probe = torch.zeros(1, 12, width, dtype=DOUBLE, requires_grad=True)
        value = layer._shell_alignment(probe).sum()
        gradient = torch.autograd.grad(value, probe, create_graph=True)[0]
        curvature = torch.autograd.grad(gradient.sum(), probe)[0]
        assert torch.isfinite(gradient).all() and torch.isfinite(curvature).all()

    def test_physics_is_preserved(self):
        model = _model(invariant_overlap_width=3)
        _assert_symmetries(model)
        _assert_conservative(model)


class TestShellPairModes:
    @pytest.mark.parametrize("mode", ["banded", "exponential", "cg_ssm"])
    def test_symmetry_and_conservation(self, mode):
        model = _model(
            shell_pair_channels=6, shell_pair_width=2, shell_pair_mode=mode
        )
        _assert_symmetries(model)
        _assert_conservative(model)

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="shell_pair_mode"):
            _model(shell_pair_channels=4, shell_pair_mode="attention")

    def test_rejects_nonpositive_clip(self):
        with pytest.raises(ValueError, match="state_clip"):
            _model(shell_pair_channels=4, shell_pair_mode="cg_ssm",
                   shell_pair_state_clip=0.0)


class TestPolynomialDegree:
    """The central architecture-v11 claim, tested rather than asserted.

    ``exponential`` keeps the recurrence linear in the state, so the drive is the
    only nonlinearity and the result is exactly degree two in ``T``.  ``cg_ssm``
    puts a tensor product inside the recurrence, so ``deg_T(S_k) <= k + 1`` and
    the degree grows with the shell index.

    Probe: freeze the controls, scale the tokens by ``lambda``, and take finite
    differences in ``lambda`` at zero.  A polynomial of degree ``d`` has
    identically vanishing derivatives above order ``d``.
    """

    @staticmethod
    def _state_fn(model, layer, mode, tokens, controls):
        width = model.ace.irreps_correlation.dim

        def state(scale, index):
            reduced = layer.shell_pair_down(
                (scale * tokens).reshape(-1, width)
            ).reshape(1, tokens.shape[1], -1)
            if mode == "exponential":
                return layer._exponential_shell_state(reduced, controls)[0, index]
            return layer._cg_shell_state(reduced, controls)[0, index]

        return state

    @staticmethod
    def _derivatives(state, index, step=0.05):
        second = (state(step, index) - 2 * state(0.0, index)
                  + state(-step, index)) / step**2
        third = (state(2 * step, index) - 2 * state(step, index)
                 + 2 * state(-step, index) - state(-2 * step, index)) / (2 * step**3)
        fourth = (state(2 * step, index) - 4 * state(step, index)
                  + 6 * state(0.0, index) - 4 * state(-step, index)
                  + state(-2 * step, index)) / step**4
        return tuple(float(v.abs().max()) for v in (second, third, fourth))

    def _setup(self, mode):
        # A clip far above any value reached keeps the saturating normalisation
        # from contributing its own higher-order terms to this measurement.
        model = _model(
            num_shells=10, shell_pair_channels=6, shell_pair_mode=mode,
            shell_pair_state_clip=1.0e6, num_layers=1,
        )
        layer = model.layers[0]
        torch.manual_seed(1)
        tokens = torch.randn(
            1, 10, model.ace.irreps_correlation.dim, dtype=DOUBLE
        ) * 0.3
        controls = torch.randn(1, 10, 16, dtype=DOUBLE) * 0.1
        return self._state_fn(model, layer, mode, tokens, controls)

    def test_exponential_is_exactly_degree_two(self):
        state = self._setup("exponential")
        for index in (3, 5, 8):
            second, third, fourth = self._derivatives(state, index)
            assert second > 1e-4, "the degree-two term must be present"
            assert third < 1e-12, f"degree 3 leaked at k={index}: {third:.3e}"
            assert fourth < 1e-10, f"degree 4 leaked at k={index}: {fourth:.3e}"

    def test_cg_ssm_degree_grows_with_shell_index(self):
        state = self._setup("cg_ssm")
        # S_0 = gamma W T_0 is degree one, so S_1 reaches degree two and no more.
        _, third_first, _ = self._derivatives(state, 1)
        assert third_first < 1e-12, "S_1 must not exceed degree two"
        # Deeper shells accumulate products and must carry degree 3 and 4.
        for index in (3, 5):
            _, third, fourth = self._derivatives(state, index)
            assert third > 1e-6, f"no degree-3 content at k={index}"
            assert fourth > 1e-7, f"no degree-4 content at k={index}"


class TestLearnedRadialBand:
    def test_decay_coefficients_lie_strictly_inside_the_unit_interval(self):
        """alpha = exp(-softplus) is a decay, never a gain.

        Unrolled, the kernel is ``exp(-sum_s Delta_s a_s)`` with ``a > 0``, i.e. a
        genuine radial correlation length.  If alpha could reach or exceed one the
        recurrence would integrate without forgetting and the band would be
        meaningless.
        """

        model = _model(shell_pair_channels=6, shell_pair_mode="exponential")
        layer = model.layers[0]
        torch.manual_seed(0)
        controls = torch.randn(4, 12, 16, dtype=DOUBLE) * 5.0
        alpha, beta, gamma = layer._shell_dynamics(controls)
        assert float(alpha.min()) > 0.0
        assert float(alpha.max()) < 1.0
        assert float(beta.abs().max()) <= 1.0
        assert float(gamma.abs().max()) <= 1.0

    def test_response_decays_with_shell_separation(self):
        """dS_k/dT_{k'} must fall off with |k - k'|, i.e. the band is finite."""

        model = _model(
            num_shells=10, shell_pair_channels=6, shell_pair_mode="exponential",
            num_layers=1,
        )
        layer = model.layers[0]
        width = model.ace.irreps_correlation.dim
        torch.manual_seed(2)
        tokens = (torch.randn(1, 10, width, dtype=DOUBLE) * 0.3).requires_grad_(True)
        controls = torch.randn(1, 10, 16, dtype=DOUBLE) * 0.1
        reduced = layer.shell_pair_down(tokens.reshape(-1, width)).reshape(1, 10, -1)
        state = layer._exponential_shell_state(reduced, controls)
        kernel = np.zeros((10, 10))
        for k in range(10):
            gradient = torch.autograd.grad(
                state[0, k].pow(2).sum(), tokens, retain_graph=True
            )[0]
            kernel[k] = gradient[0].norm(dim=-1).detach().numpy()
        band = [
            float(np.mean([kernel[k, k + d] for k in range(10 - d)]))
            for d in range(4)
        ]
        assert band[0] > band[1] > band[2] > band[3], f"band does not decay: {band}"

    def test_cg_state_is_bounded_by_the_clip(self):
        """Repeated tensor products would grow geometrically without the clip."""

        clip = 2.0
        model = _model(
            num_shells=10, shell_pair_channels=6, shell_pair_mode="cg_ssm",
            shell_pair_state_clip=clip, num_layers=1,
        )
        layer = model.layers[0]
        width = model.ace.irreps_correlation.dim
        torch.manual_seed(4)
        # Deliberately far outside the trained range.
        tokens = torch.randn(1, 10, width, dtype=DOUBLE) * 50.0
        reduced = layer.shell_pair_down(tokens.reshape(-1, width)).reshape(1, 10, -1)
        state = layer._cg_shell_state(
            reduced, torch.randn(1, 10, 16, dtype=DOUBLE)
        )
        assert float(state.norm(dim=-1).max()) <= clip + 1e-9
        assert torch.isfinite(state).all()


class TestAffineShellScan:
    def test_matches_the_serial_recurrence(self):
        """The parallel scan must reproduce S_k = a_k S_{k-1} + d_k exactly."""

        from mtace.mamba3 import affine_shell_scan

        torch.manual_seed(0)
        drive = torch.randn(3, 16, 7, dtype=DOUBLE)
        transition = torch.rand(3, 16, 1, dtype=DOUBLE) * 0.9 + 0.05
        parallel = affine_shell_scan(drive, transition)

        serial = torch.zeros_like(drive)
        state = torch.zeros(3, 7, dtype=DOUBLE)
        for k in range(16):
            state = transition[:, k] * state + drive[:, k]
            serial[:, k] = state
        torch.testing.assert_close(parallel, serial, rtol=0.0, atol=1e-12)

    def test_rejects_mismatched_shapes(self):
        from mtace.mamba3 import affine_shell_scan

        with pytest.raises(ValueError, match="atoms, shells, features"):
            affine_shell_scan(torch.zeros(3, 4), torch.zeros(3, 4, 1))
        with pytest.raises(ValueError, match="atom and shell"):
            affine_shell_scan(torch.zeros(3, 4, 5), torch.zeros(2, 4, 1))


class TestKernelSpectroscopy:
    """The diagnostic that turns the mixer ablation into a yes/no question.

    Each mixer is a prior on the shell correlation kernel: the SSM is low-rank
    times a learned exponential band, attention is softmax-concentrated, dense is
    arbitrary.  Measuring the kernel directly is far cheaper than a converged
    accuracy comparison and can falsify the prior outright.
    """

    def _tool(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "experiments" / "mixer_kernel_spectroscopy.py"
        spec = importlib.util.spec_from_file_location("kernel_spectroscopy", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_identity_mixer_has_exactly_zero_mixing_kernel(self):
        """The probe must isolate mixing from the skip connection.

        Every mixer returns ``hidden + delta``.  Differentiating the raw output
        measures ``I + dDelta/du`` and reports full rank for *every* mixer,
        including the identity control.  Subtracting the skip connection is what
        makes the measurement meaningful, and the identity mixer is the test:
        it must give identically zero.
        """

        tool = self._tool()
        model = tool.build("identity", 10)
        layer = model.layers[0]
        kernel = tool.mixer_kernel(layer, 4, 10, layer.node_context.out_features)
        assert float(np.abs(kernel).max()) == 0.0
        assert tool.effective_rank(kernel) == 0.0

    def test_effective_rank_endpoints(self):
        tool = self._tool()
        # A rank-one kernel has participation ratio one.
        rank_one = np.outer(np.ones(8), np.ones(8))
        assert tool.effective_rank(rank_one) == pytest.approx(1.0, abs=1e-9)
        # An orthogonal kernel has a flat spectrum and participation ratio L.
        assert tool.effective_rank(np.eye(8)) == pytest.approx(8.0, abs=1e-9)

    def test_band_profile_recovers_a_known_decay_rate(self):
        tool = self._tool()
        length, rate = 24, 0.4
        index = np.arange(length)
        synthetic = np.exp(-rate * np.abs(index[:, None] - index[None, :]))
        profile, measured = tool.band_profile(synthetic)
        assert measured == pytest.approx(rate, rel=1e-6)
        assert profile[0] > profile[1] > profile[2]

    def test_state_space_kernel_is_more_banded_than_the_dense_one(self):
        """The taxonomy is not decoration: it is measurable at initialisation.

        The unconstrained dense mixer has no decay prior, so its kernel should be
        far less banded than the state-space kernel, whose unrolled form carries
        exp(-sum Delta a) by construction.
        """

        tool = self._tool()
        rates = {}
        for mixer in ("dense", "mamba"):
            model = tool.build(mixer, 16)
            layer = model.layers[0]
            kernel = tool.mixer_kernel(layer, 6, 16, layer.node_context.out_features)
            rates[mixer] = tool.band_profile(kernel)[1]
        assert rates["mamba"] > rates["dense"]


class TestScreenedDecay:
    """A physically constrained memory, and a predicted observable.

    The Mamba-3 default learns alpha = exp(Delta A) with both factors free per
    head and shell.  Constraining it to

        alpha_k = exp(-dr / lambda_i),   dr = shell spacing in Angstrom

    makes the unrolled kernel K(k,k') ~ exp(-|r_k - r_k'| / lambda_i), which is
    the Yukawa / Thomas-Fermi form, and turns lambda_i into a per-atom screening
    length in Angstrom that can be compared against known physics.
    """

    R_MAX, R_MIN, SHELLS = 6.0, 2.0, 16

    def _model(self, **overrides):
        torch.manual_seed(11)
        settings = dict(
            r_max=self.R_MAX, l_max=2, num_radial=8, hidden_dim=16, num_layers=1,
            correlation_order=3, correlation_channels=8, num_shells=self.SHELLS,
            shell_r_min=self.R_MIN, shell_degree=5, avg_num_neighbors=12.0,
            invariant_norm="homogeneous", mamba_dim=16, mamba_d_state=8,
            mamba_headdim=8, readout_hidden=16, mamba_backend="torch",
        )
        settings.update(overrides)
        return MambaACEV2(**settings).double().eval()

    def test_shell_spacing_is_physical(self):
        model = self._model(decay_mode="screening")
        expected = (self.R_MAX - self.R_MIN) / (self.SHELLS - 1)
        assert model.layers[0].shell_spacing_angstrom == pytest.approx(expected)

    @pytest.mark.parametrize("target", [0.4, 0.8, 1.5, 3.0])
    def test_decay_equals_exp_minus_dr_over_lambda(self, target):
        """The defining equation, checked numerically rather than assumed."""

        model = self._model(decay_mode="screening")
        layer = model.layers[0]
        with torch.no_grad():
            layer.screening_projection.weight.zero_()
            inverse = torch.log(torch.expm1(
                torch.tensor(target - layer.screening_min_angstrom, dtype=DOUBLE)
            ))
            layer.screening_projection.bias.fill_(float(inverse))
        invariants = torch.randn(4, layer.node_invariant_dim, dtype=DOUBLE)
        lengths = layer.screening_length(invariants)
        assert float(lengths.mean()) == pytest.approx(target, rel=1e-9)
        alpha = float(torch.exp(
            -layer.shell_spacing_angstrom / lengths
        ).mean())
        assert alpha == pytest.approx(
            math.exp(-layer.shell_spacing_angstrom / target), rel=1e-12
        )

    def test_length_is_invariant_and_bounded(self):
        model = self._model(decay_mode="screening")
        positions, z, cell, edge_index, edge_shift = _cluster()
        lengths = model.screening_lengths(z, positions, cell, edge_index, edge_shift)[0]
        rotation = torch.linalg.qr(torch.randn(3, 3, dtype=DOUBLE))[0]
        rotation = rotation * torch.sign(torch.det(rotation))
        rotated = model.screening_lengths(
            z, positions @ rotation.T, cell, edge_index, edge_shift
        )[0]
        # Relative, not absolute: lambda passes through sqrt and softplus on top
        # of the ACE frontend, so float64 accumulation is a few 1e-13 on an
        # Angstrom-scale value.  The energy is exact to 1e-16 because its sums
        # cancel; a per-atom nonlinear readout cannot be.
        torch.testing.assert_close(lengths, rotated, rtol=1e-10, atol=0.0)
        assert bool((lengths > model.layers[0].screening_min_angstrom).all())
        assert torch.isfinite(lengths).all()

    def test_constrained_decay_costs_almost_nothing(self):
        """One scalar per atom replaces a free rate per head and shell."""

        free = sum(p.numel() for p in self._model().parameters())
        screened = sum(
            p.numel() for p in self._model(decay_mode="screening").parameters()
        )
        assert 0 < screened - free < 100

    def test_physics_is_preserved(self):
        model = self._model(decay_mode="screening")
        _assert_symmetries(model)
        _assert_conservative(model)

    def test_diagnostic_refuses_when_decay_is_free(self):
        model = self._model()
        positions, z, cell, edge_index, edge_shift = _cluster()
        with pytest.raises(RuntimeError, match="screening"):
            model.screening_lengths(z, positions, cell, edge_index, edge_shift)

    def test_rejects_bad_settings(self):
        with pytest.raises(ValueError, match="decay_mode"):
            self._model(decay_mode="yukawa")
        with pytest.raises(ValueError, match="screening_min"):
            self._model(decay_mode="screening", screening_min_angstrom=0.0)


class TestMultiresolutionShells:
    """Dyadic scales, ordered coarse to fine.

    Scale s carries ``L_s = (L_0 - 1) 2^s + 1`` shells over the same physical
    interval, so *each* scale is its own exact partition of unity and the
    reconstruction identity holds per scale.  The resulting sequence has a
    canonical direction -- refining resolution -- which a single-scale radial
    axis does not, and it finally gives ``token_kind`` a job: the kind is the
    scale.
    """

    def _model(self, scales, base=7):
        torch.manual_seed(7)
        return MambaACEV2(
            r_max=6.0, l_max=2, num_radial=8, hidden_dim=16, num_layers=1,
            correlation_order=3, correlation_channels=8, num_shells=base,
            shell_r_min=2.0, shell_degree=5, shell_scales=scales,
            avg_num_neighbors=12.0, invariant_norm="homogeneous",
            mamba_dim=16, mamba_d_state=8, mamba_headdim=8, readout_hidden=16,
            mamba_backend="torch",
        ).double().eval()

    def test_dyadic_shell_counts(self):
        for scales, expected in ((1, [7]), (2, [7, 13]), (3, [7, 13, 25])):
            model = self._model(scales)
            assert model.ace.shell_counts == expected
            assert model.ace.sequence_length == sum(expected)
            assert model.ace.num_token_kinds == scales

    def test_reconstruction_identity_holds_per_scale(self):
        """sum_k T^(s)_ik = A_i for EVERY scale, not just in aggregate."""

        model = self._model(3)
        positions, z, cell, edge_index, edge_shift = _cluster()
        with torch.no_grad():
            vectors = positions[edge_index[0]] - positions[edge_index[1]]
            lengths = vectors.norm(dim=-1)
            from mtace.physics import ACEV2Descriptor

            _, edge_features, _ = ACEV2Descriptor.forward(
                model.ace, model.species_embedding(z), edge_index, vectors,
                lengths, return_edge_features=True,
            )
            density = torch.zeros(
                positions.shape[0], model.ace.irreps_correlation.dim, dtype=DOUBLE
            )
            density.index_add_(0, edge_index[1], edge_features)
            tokens = model.ace.pool_edge_features(
                edge_features, edge_index[1], lengths, positions.shape[0]
            )
        offset = 0
        for count in model.ace.shell_counts:
            partial = tokens[:, offset : offset + count].sum(dim=1)
            assert float((partial - density).abs().max()) < 1e-13
            offset += count

    def test_token_kind_encodes_the_scale(self):
        model = self._model(3)
        kinds = model.ace.token_kind.tolist()
        offset = 0
        for scale, count in enumerate(model.ace.shell_counts):
            assert set(kinds[offset : offset + count]) == {scale}
            offset += count

    @pytest.mark.parametrize("scales", [1, 2, 3])
    def test_physics_is_preserved(self, scales):
        model = self._model(scales)
        _assert_symmetries(model)
        _assert_conservative(model)

    def test_rejects_zero_scales(self):
        with pytest.raises(ValueError, match="shell_scales"):
            self._model(0)
