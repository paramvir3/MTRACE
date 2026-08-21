"""Tests for the architecture-v9 peer-review corrections.

Each test corresponds to a specific finding: density normalization, the folded
shell boundary and inner cutoff radius, the memory-bounded scan, the unified
rotary layout, the shell-resolution study, and the gate diagnostic.
"""

import math
import unittest

import numpy as np
import torch
from ase import Atoms

from mtace.data import average_num_neighbors, build_neighbor_tensors, shell_occupancy
from mtace.mamba3 import (
    Mamba3Direction,
    mamba3_mimo_scan_parallel,
    mamba3_mimo_scan_reference,
    mamba3_scan_parallel,
    mamba3_scan_reference,
)
from mtace.model import MambaACEV2
from mtace.physics import CompactRadialShellBasis

R_MAX = 6.0


def _disordered_cell(seed: int = 0, repeat: int = 2) -> Atoms:
    generator = np.random.RandomState(seed)
    lattice = 3.6
    cell = np.eye(3) * lattice * repeat
    positions = []
    for i in range(repeat):
        for j in range(repeat):
            for k in range(repeat):
                base = np.array([i, j, k], dtype=float) * lattice
                for offset in (
                    (0.0, 0.0, 0.0),
                    (0.5, 0.5, 0.0),
                    (0.5, 0.0, 0.5),
                    (0.0, 0.5, 0.5),
                ):
                    positions.append(base + np.asarray(offset) * lattice)
    positions = np.asarray(positions) + 0.1 * generator.randn(len(positions), 3)
    return Atoms("Cu" + str(len(positions)), positions=positions, cell=cell, pbc=True)


def _tensors(atoms, dtype=torch.float64):
    edge_index, edge_shift = build_neighbor_tensors(atoms, R_MAX, dtype)
    return (
        torch.as_tensor(atoms.numbers, dtype=torch.long),
        torch.as_tensor(atoms.positions, dtype=dtype),
        torch.as_tensor(atoms.cell.array, dtype=dtype),
        edge_index,
        edge_shift,
    )


def _model(**overrides) -> MambaACEV2:
    torch.manual_seed(overrides.pop("seed", 11))
    settings = dict(
        r_max=R_MAX,
        l_max=2,
        num_radial=6,
        hidden_dim=16,
        num_layers=1,
        correlation_order=3,
        correlation_channels=8,
        num_shells=12,
        mamba_dim=16,
        mamba_d_state=8,
        mamba_headdim=8,
        mamba_mimo_rank=4,
        readout_hidden=16,
        mamba_backend="torch",
    )
    settings.update(overrides)
    return MambaACEV2(**settings).double().eval()


class ShellBoundaryTests(unittest.TestCase):
    """The folded boundary is an exact partition of unity and stays C2."""

    def test_folded_weights_sum_to_one_everywhere(self):
        for r_min in (0.0, 1.5, 2.5):
            basis = CompactRadialShellBasis(
                R_MAX, 14, r_min=r_min, boundary_mode="fold"
            )
            distances = torch.linspace(-1.0, 7.0, 601, dtype=torch.float64)
            weights = basis.dense(distances)
            self.assertLess(float((weights.sum(dim=-1) - 1.0).abs().max()), 1.0e-12)

    def test_folded_weights_have_a_bounded_second_derivative(self):
        basis = CompactRadialShellBasis(R_MAX, 14, r_min=1.5, boundary_mode="fold")
        step = 1.0e-5
        # Sample straight across the inner boundary, where a clamped coordinate
        # would introduce a kink and therefore a discontinuous force.
        grid = torch.linspace(1.0, 2.0, 201, dtype=torch.float64)
        second = (
            basis.dense(grid + step) - 2.0 * basis.dense(grid) + basis.dense(grid - step)
        ) / step**2
        self.assertTrue(bool(torch.isfinite(second).all()))
        self.assertLess(float(second.abs().max()), 1.0e3)

    def test_inner_radius_requires_the_folded_boundary(self):
        with self.assertRaisesRegex(ValueError, "boundary_mode='fold'"):
            CompactRadialShellBasis(R_MAX, 8, r_min=1.0, boundary_mode="renormalize")

    def test_renormalizing_boundary_still_sums_to_one(self):
        basis = CompactRadialShellBasis(R_MAX, 9, boundary_mode="renormalize")
        distances = torch.linspace(0.0, R_MAX, 201, dtype=torch.float64)
        weights = basis.dense(distances)
        self.assertLess(float((weights.sum(dim=-1) - 1.0).abs().max()), 1.0e-12)

    def test_uniform_shells_on_the_full_range_waste_the_sequence(self):
        atoms = _disordered_cell()
        basis = CompactRadialShellBasis(R_MAX, 32, boundary_mode="fold")
        occupancy = shell_occupancy([atoms], R_MAX, basis)
        empty = int((occupancy == 0.0).sum())
        self.assertGreater(empty, 0)
        # Moving the inner edge above the repulsive wall removes the dead tokens.
        trimmed = CompactRadialShellBasis(
            R_MAX, 32, r_min=2.0, boundary_mode="fold"
        )
        trimmed_occupancy = shell_occupancy([atoms], R_MAX, trimmed)
        self.assertLess(int((trimmed_occupancy == 0.0).sum()), empty)


class ScanMemoryTests(unittest.TestCase):
    """The chunked scan is the same recurrence with fewer retained tensors."""

    @staticmethod
    def _mimo_inputs(requires_grad=False):
        torch.manual_seed(5)
        shape = (2, 16, 3, 2, 4)
        state = (2, 16, 3, 2, 6)
        make = lambda s: torch.randn(*s, dtype=torch.float64, requires_grad=requires_grad)
        alpha = torch.rand(2, 16, 2, dtype=torch.float64) * 0.8 + 0.1
        return (
            make(shape),
            make(state),
            make(state),
            alpha.requires_grad_(requires_grad),
            make((2, 16, 2)),
            make((2, 16, 2)),
            make((2,)),
            make(shape),
        )

    def test_chunked_mimo_scan_matches_the_serial_recurrence(self):
        arguments = self._mimo_inputs()
        expected = mamba3_mimo_scan_reference(*arguments)
        for chunk in (1, 2, 4, 5, 16, 64):
            actual = mamba3_mimo_scan_parallel(*arguments, chunk_size=chunk)
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_chunked_siso_scan_matches_the_serial_recurrence(self):
        torch.manual_seed(6)
        x = torch.randn(2, 16, 2, 4, dtype=torch.float64)
        k = torch.randn(2, 16, 2, 6, dtype=torch.float64)
        q = torch.randn(2, 16, 2, 6, dtype=torch.float64)
        alpha = torch.rand(2, 16, 2, dtype=torch.float64) * 0.8 + 0.1
        beta = torch.randn(2, 16, 2, dtype=torch.float64)
        gamma = torch.randn(2, 16, 2, dtype=torch.float64)
        expected = mamba3_scan_reference(x, k, q, alpha, beta, gamma)
        for chunk in (1, 3, 8, 16):
            actual = mamba3_scan_parallel(
                x, k, q, alpha, beta, gamma, chunk_size=chunk
            )
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_chunked_scan_supports_double_backward(self):
        arguments = self._mimo_inputs(requires_grad=True)
        gradients = []
        for chunk in (None, 4):
            output = mamba3_mimo_scan_parallel(*arguments, chunk_size=chunk)
            first = torch.autograd.grad(
                output.square().sum(), arguments[0], create_graph=True
            )[0]
            second = torch.autograd.grad(first.square().sum(), arguments[0])[0]
            gradients.append((first.detach(), second))
        torch.testing.assert_close(gradients[0][0], gradients[1][0], rtol=1e-8, atol=1e-8)
        torch.testing.assert_close(gradients[0][1], gradients[1][1], rtol=1e-6, atol=1e-6)

    def test_scan_mode_is_validated(self):
        with self.assertRaisesRegex(ValueError, "scan_mode"):
            Mamba3Direction(d_model=8, d_state=8, headdim=4, scan_mode="banana")

    def test_auto_scan_mode_uses_the_single_pass(self):
        # Blocking is opt-in: it is measurably not a memory win (see
        # benchmarks/benchmark_scan_memory.py) and costs latency.
        auto = Mamba3Direction(
            d_model=8, d_state=8, headdim=4, chunk_size=8, scan_mode="auto"
        )
        self.assertIsNone(auto._resolved_scan_chunk(12))
        self.assertIsNone(auto._resolved_scan_chunk(1024))
        chunked = Mamba3Direction(
            d_model=8, d_state=8, headdim=4, chunk_size=8, scan_mode="chunked"
        )
        self.assertEqual(chunked._resolved_scan_chunk(64), 8)


class RotaryLayoutTests(unittest.TestCase):
    """One portable rotary layout for every MIMO rank, and a fused-path guard."""

    def test_rank_one_and_rank_four_share_the_rotary_layout(self):
        for rank in (1, 4):
            direction = Mamba3Direction(
                d_model=8, d_state=8, headdim=4, mimo_rank=rank
            )
            self.assertEqual(direction.rotary_layout, "halves")

    def test_pairs_layout_blocks_the_fused_kernels(self):
        direction = Mamba3Direction(
            d_model=8, d_state=8, headdim=4, rotary_layout="pairs"
        )
        self.assertIsNotNone(direction.fused_configuration_error)
        self.assertIn("halves", direction.fused_configuration_error)

    def test_halves_layout_has_no_layout_related_fused_error(self):
        direction = Mamba3Direction(
            d_model=8, d_state=8, headdim=4, rotary_layout="halves"
        )
        self.assertIsNone(direction.fused_configuration_error)

    def test_layout_changes_the_predicted_energy(self):
        # Not a bug but a hazard: the layouts differ by a permutation of state
        # coordinates, so a checkpoint trained under one must never be evaluated
        # under the other.
        atoms = _disordered_cell()
        z, pos, cell, edge_index, edge_shift = _tensors(atoms)
        energies = []
        for layout in ("halves", "pairs"):
            model = _model(mamba_rotary_layout=layout, mamba_mimo_rank=1)
            with torch.no_grad():
                energies.append(
                    float(
                        model.atomic_energies(z, pos, cell, edge_index, edge_shift).sum()
                    )
                )
        self.assertNotAlmostEqual(energies[0], energies[1], places=8)


class DensityNormalizationTests(unittest.TestCase):
    """Density normalization removes most of the coordination dependence."""

    @staticmethod
    def _feature_scale(model, atoms):
        z, pos, cell, edge_index, edge_shift = _tensors(atoms)
        with torch.no_grad():
            vector = pos[edge_index[0]] - pos[edge_index[1]] + edge_shift @ cell
            length = torch.linalg.vector_norm(vector, dim=-1)
            features, _, _, _ = model.ace(
                model.species_embedding(z), edge_index, vector, length
            )
        return float(features.pow(2).mean().sqrt())

    def test_normalization_reduces_sensitivity_to_coordination(self):
        sparse = _disordered_cell(seed=1)
        dense = _disordered_cell(seed=1)
        dense.set_cell(dense.cell.array * 0.8, scale_atoms=True)
        plain = _model(avg_num_neighbors=1.0)
        normalized = _model(avg_num_neighbors=average_num_neighbors([sparse], R_MAX))
        plain_ratio = self._feature_scale(plain, dense) / self._feature_scale(
            plain, sparse
        )
        normalized_ratio = self._feature_scale(
            normalized, dense
        ) / self._feature_scale(normalized, sparse)
        self.assertGreater(abs(math.log(plain_ratio)), abs(math.log(normalized_ratio)))

    def test_normalization_preserves_the_reconstruction_identity(self):
        atoms = _disordered_cell()
        z, pos, cell, edge_index, edge_shift = _tensors(atoms)
        model = _model(avg_num_neighbors=48.0)
        with torch.no_grad():
            vector = pos[edge_index[0]] - pos[edge_index[1]] + edge_shift @ cell
            length = torch.linalg.vector_norm(vector, dim=-1)
            density, edges, _ = model.ace._density(
                model.species_embedding(z), edge_index, vector, length
            )
            tokens = model.ace.pool_edge_features(
                edges, edge_index[1], length, z.numel()
            )
        torch.testing.assert_close(tokens.sum(dim=1), density, rtol=1e-11, atol=1e-11)

    def test_average_num_neighbors_is_positive_for_isolated_atoms(self):
        isolated = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 20.0, pbc=True)
        self.assertGreaterEqual(average_num_neighbors([isolated], R_MAX), 1.0)


class ResolutionTests(unittest.TestCase):
    """Re-gridding is a valid research operation with a *bounded*, not vanishing, drift.

    The measured study in ``docs/RESOLUTION_STUDY.md`` shows that refining the
    shell mesh of a fixed parameter vector changes the energy by an amount that
    grows slowly instead of converging.  The shell count is therefore a genuine
    hyperparameter and a re-gridded model must be revalidated.  What these tests
    do guarantee is that re-gridding is *structurally* sound: the reconstruction
    identity survives, forces stay conservative on the new mesh, and the drift
    stays small enough to be usable for the resolution study itself.
    """

    def test_mesh_refinement_drift_stays_bounded(self):
        atoms = _disordered_cell()
        z, pos, cell, edge_index, edge_shift = _tensors(atoms)
        model = _model(num_shells=16, shell_r_min=1.5)
        energies = []
        for shells in (16, 31, 61, 121):
            model.set_num_shells(shells)
            with torch.no_grad():
                energies.append(
                    float(
                        model.atomic_energies(z, pos, cell, edge_index, edge_shift).sum()
                    )
                    / len(atoms)
                )
        drift = max(abs(value - energies[0]) for value in energies)
        # Sub-meV/atom over an eightfold refinement: small, but explicitly not a
        # convergence claim.
        self.assertLess(drift, 1.0e-3)

    def test_regridding_preserves_the_reconstruction_identity(self):
        atoms = _disordered_cell()
        z, pos, cell, edge_index, edge_shift = _tensors(atoms)
        model = _model(num_shells=16, shell_r_min=1.0)
        model.set_num_shells(41)
        with torch.no_grad():
            vector = pos[edge_index[0]] - pos[edge_index[1]] + edge_shift @ cell
            length = torch.linalg.vector_norm(vector, dim=-1)
            density, edges, _ = model.ace._density(
                model.species_embedding(z), edge_index, vector, length
            )
            tokens = model.ace.pool_edge_features(
                edges, edge_index[1], length, z.numel()
            )
        self.assertEqual(tokens.shape[1], 41)
        torch.testing.assert_close(tokens.sum(dim=1), density, rtol=1e-11, atol=1e-11)

    def test_regridding_rejects_the_dense_mixer(self):
        model = _model(mixer_type="dense", num_shells=12)
        with self.assertRaisesRegex(ValueError, "dense"):
            model.set_num_shells(24)

    def test_refined_mesh_keeps_conservative_forces(self):
        atoms = _disordered_cell(repeat=1)
        z, pos, cell, edge_index, edge_shift = _tensors(atoms)
        model = _model(num_shells=12, shell_r_min=1.5)
        model.set_num_shells(29)
        data = {
            "z": z,
            "pos": pos,
            "cell": cell,
            "edge_index": edge_index,
            "edge_shift": edge_shift,
        }
        _, forces, _, _ = model(data, training=False, compute_stress=False)
        step = 1.0e-6
        for atom in (0, 2):
            for axis in range(3):
                shifted = pos.clone()
                shifted[atom, axis] += step
                plus = float(
                    model.atomic_energies(
                        z, shifted, cell, edge_index, edge_shift
                    ).sum()
                )
                shifted[atom, axis] -= 2.0 * step
                minus = float(
                    model.atomic_energies(
                        z, shifted, cell, edge_index, edge_shift
                    ).sum()
                )
                self.assertAlmostEqual(
                    float(forces[atom, axis]), -(plus - minus) / (2.0 * step), places=6
                )


class GateDiagnosticTests(unittest.TestCase):
    """The shell-dependence diagnostic detects a decorative mixer."""

    def test_identity_mixer_and_shell_dependence(self):
        atoms = _disordered_cell()
        z, pos, cell, edge_index, edge_shift = _tensors(atoms)
        report = _model(mixer_type="mamba").gate_shell_dependence(
            z, pos, cell, edge_index, edge_shift
        )
        self.assertEqual(len(report), 1)
        entry = report[0]
        for key in (
            "gate_abs_mean",
            "gate_std_over_shells",
            "gate_std_over_channels",
            "residual_fraction",
            "update_norm",
        ):
            self.assertIn(key, entry)
        self.assertGreaterEqual(entry["residual_fraction"], 0.0)
        self.assertLessEqual(entry["residual_fraction"], 2.0)

    def test_a_constant_gate_reproduces_the_direct_ace_path(self):
        # With a zeroed gate projection the gate is tanh(b_g), constant over
        # shells, and the update collapses onto W_O W_V A_i.
        atoms = _disordered_cell()
        z, pos, cell, edge_index, edge_shift = _tensors(atoms)
        model = _model()
        with torch.no_grad():
            model.layers[0].gate_projection.weight.zero_()
        report = model.gate_shell_dependence(z, pos, cell, edge_index, edge_shift)
        self.assertLess(report[0]["residual_fraction"], 1.0e-10)


class ConfigurationGuardTests(unittest.TestCase):
    """Silently ignored settings now fail loudly."""

    def test_mamba_d_conv_is_rejected_for_mamba3(self):
        with self.assertRaisesRegex(ValueError, "mamba_d_conv"):
            _model(mamba_d_conv=5)

    def test_mamba_d_conv_is_accepted_for_mamba1(self):
        model = _model(mamba_variant="mamba1", mamba_d_conv=5, mamba_mimo_rank=1)
        self.assertEqual(model.layers[0].mixer.d_conv, 5)

    def test_degenerate_automatic_headdim_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "headdim"):
            _model(mamba_dim=7, mamba_expand=1, mamba_headdim=None)

    def test_ffn_type_is_independent_of_the_mixer(self):
        swiglu = _model(mamba_variant="mamba1", mamba_mimo_rank=1, ffn_type="swiglu")
        self.assertTrue(swiglu.layers[0].use_swiglu)
        mlp = _model(ffn_type="mlp")
        self.assertFalse(mlp.layers[0].use_swiglu)

    def test_invariant_pair_channels_add_cross_channel_invariants(self):
        plain = _model(invariant_pair_channels=0)
        enriched = _model(invariant_pair_channels=4)
        self.assertGreater(
            enriched.layers[0].token_invariant_dim,
            plain.layers[0].token_invariant_dim,
        )
        atoms = _disordered_cell()
        z, pos, cell, edge_index, edge_shift = _tensors(atoms)
        with torch.no_grad():
            value = enriched.atomic_energies(z, pos, cell, edge_index, edge_shift)
        self.assertTrue(bool(torch.isfinite(value).all()))

    def test_enriched_invariants_stay_rotation_invariant(self):
        atoms = _disordered_cell(repeat=1)
        z, pos, cell, edge_index, edge_shift = _tensors(atoms)
        model = _model(invariant_pair_channels=4)
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
        with torch.no_grad():
            plain = model.atomic_energies(z, pos, cell, edge_index, edge_shift).sum()
            rotated = model.atomic_energies(
                z, pos @ rotation.T, cell @ rotation.T, edge_index, edge_shift
            ).sum()
        self.assertAlmostEqual(float(plain), float(rotated), places=10)


if __name__ == "__main__":
    unittest.main()
