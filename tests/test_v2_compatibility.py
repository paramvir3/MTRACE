import sys
import unittest
from pathlib import Path

import torch

from mtace.model import MambaACE, MambaACEV2
from mtace.mamba3 import Mamba3SequenceMixer
from mtace.physics import (
    ACEV2Descriptor,
    ACEV2MambaTokenizer,
    BesselRadialBasis,
    CompactRadialShellBasis,
    V2BesselBasis,
)


FLASH_ACE_ROOT = Path(__file__).resolve().parents[2] / "Flash-ACE"


def geometry():
    attributes = torch.randn(4, 8)
    edge_index = torch.tensor(
        [[1, 2, 3, 0, 2, 3, 0, 1, 3], [0, 0, 0, 1, 1, 1, 2, 2, 2]],
        dtype=torch.long,
    )
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]]
    )
    edge_vector = positions[edge_index[0]] - positions[edge_index[1]]
    return attributes, edge_index, edge_vector, torch.linalg.vector_norm(edge_vector, dim=1)


class TraceV2CompatibilityTests(unittest.TestCase):
    def test_bessel_channels_use_the_regular_analytic_limit_at_zero(self):
        r_max = 4.5
        distances = torch.tensor(
            [0.0, 1.0e-16, 1.0e-12, 1.0e-9],
            dtype=torch.float64,
            requires_grad=True,
        )
        for basis in (
            V2BesselBasis(r_max, 4).double(),
            BesselRadialBasis(r_max, 4).double(),
        ):
            with self.subTest(basis=basis.__class__.__name__):
                values = basis(distances)
                torch.testing.assert_close(
                    values[1], values[0], atol=2.0e-14, rtol=2.0e-14
                )
                self.assertTrue(torch.isfinite(values).all())
                first = torch.autograd.grad(
                    values.sum(), distances, create_graph=True, retain_graph=True
                )[0]
                second = torch.autograd.grad(first.sum(), distances, retain_graph=True)[0]
                self.assertTrue(torch.isfinite(first).all())
                self.assertTrue(torch.isfinite(second).all())

    def reference_descriptor(self):
        sys.path.insert(0, str(FLASH_ACE_ROOT))
        from flashace.physics import ACEV2Descriptor as TraceACEV2Descriptor

        return TraceACEV2Descriptor

    @unittest.skipUnless(FLASH_ACE_ROOT.exists(), "Sibling Flash-ACE reference is unavailable")
    def test_descriptor_state_and_numerics_match_trace_v2(self):
        torch.manual_seed(101)
        parameters = dict(
            r_max=4.5,
            l_max=2,
            num_radial=4,
            hidden_dim=8,
            correlation_order=4,
            correlation_channels=4,
            radial_basis_type="bessel",
            radial_trainable=False,
            gaussian_width=0.5,
            radial_mlp_hidden=10,
            radial_mlp_layers=2,
        )
        ours = ACEV2MambaTokenizer(**parameters).eval()
        reference = self.reference_descriptor()(**parameters).eval()
        ours_state = ours.state_dict()
        self.assertTrue(set(reference.state_dict()).issubset(ours_state))
        reference.load_state_dict(
            {name: ours_state[name] for name in reference.state_dict()}, strict=True
        )

        attributes, edge_index, edge_vector, edge_length = geometry()
        ours_density, ours_edges, ours_cutoff = ours._density(
            attributes, edge_index, edge_vector, edge_length
        )
        ref_density, ref_edges, ref_cutoff = reference._density(
            attributes, edge_index, edge_vector, edge_length
        )
        torch.testing.assert_close(ours_density, ref_density, atol=0.0, rtol=0.0)
        torch.testing.assert_close(ours_edges, ref_edges, atol=0.0, rtol=0.0)
        torch.testing.assert_close(ours_cutoff, ref_cutoff, atol=0.0, rtol=0.0)

        ours_output = ACEV2Descriptor.forward(
            ours, attributes, edge_index, edge_vector, edge_length
        )
        reference_output = reference(attributes, edge_index, edge_vector, edge_length)
        torch.testing.assert_close(ours_output, reference_output, atol=0.0, rtol=0.0)

    def test_mamba_tokens_are_only_commutative_pooling_of_v2_edges(self):
        torch.manual_seed(102)
        descriptor = ACEV2MambaTokenizer(
            r_max=4.5,
            l_max=2,
            num_radial=4,
            hidden_dim=8,
            correlation_order=4,
            correlation_channels=4,
            tokenizer_type="legacy_basis",
        ).eval()
        attributes, edge_index, edge_vector, edge_length = geometry()
        node_features, tokens, token_kind, _ = descriptor(
            attributes, edge_index, edge_vector, edge_length
        )
        exact_node, edge_features, _ = ACEV2Descriptor.forward(
            descriptor,
            attributes,
            edge_index,
            edge_vector,
            edge_length,
            return_edge_features=True,
        )
        radial = descriptor.radial_basis(edge_length)
        expected = torch.zeros_like(tokens)
        for shell in range(descriptor.num_radial):
            expected[:, shell].index_add_(
                0,
                edge_index[1],
                radial[:, shell : shell + 1] * edge_features,
            )
        torch.testing.assert_close(node_features, exact_node, atol=0.0, rtol=0.0)
        torch.testing.assert_close(tokens, expected, atol=0.0, rtol=0.0)
        self.assertTrue(token_kind.eq(0).all())
        self.assertFalse(hasattr(descriptor, "remove_pair_self_contractions"))

        permutation = torch.randperm(edge_index.shape[1])
        reordered = descriptor(
            attributes,
            edge_index[:, permutation],
            edge_vector[permutation],
            edge_length[permutation],
        )
        torch.testing.assert_close(reordered[0], node_features, atol=2.0e-6, rtol=2.0e-6)
        torch.testing.assert_close(reordered[1], tokens, atol=2.0e-6, rtol=2.0e-6)

    def test_public_default_is_v2(self):
        self.assertIs(MambaACE, MambaACEV2)
        model = MambaACE(
            r_max=3.5,
            l_max=1,
            num_radial=3,
            hidden_dim=6,
            num_layers=1,
            correlation_order=3,
            correlation_channels=3,
            mamba_dim=8,
            mamba_d_state=4,
            mamba_backend="torch",
        )
        self.assertEqual(model.architecture, "mtace_v2")
        self.assertIsInstance(model.ace, ACEV2MambaTokenizer)
        self.assertEqual(model.species_embedding.num_embeddings, 119)
        self.assertEqual(model.species_embedding(torch.tensor([118])).shape, (1, 6))
        self.assertEqual(model.ace.tokenizer_type, "physical_shells")
        self.assertEqual(model.ace.sequence_length, 32)
        self.assertIsInstance(model.layers[0].mixer, Mamba3SequenceMixer)
        self.assertIsNotNone(model.layers[0].mixer.backward_direction)
        direction = model.layers[0].mixer.forward_direction
        self.assertEqual(direction.mimo_rank, 4)
        self.assertEqual(direction.chunk_size, 16)

    def test_legacy_tokenizer_infers_historical_shell_reduction(self):
        model = MambaACE(
            r_max=3.5,
            l_max=1,
            num_radial=3,
            num_shells=3,
            hidden_dim=6,
            num_layers=1,
            correlation_order=3,
            correlation_channels=3,
            tokenizer_type="legacy_basis",
            mamba_dim=8,
            mamba_d_state=4,
            mamba_backend="torch",
        )
        self.assertEqual(model.shell_coupling_mode, "legacy")
        self.assertEqual(model.ace.shell_coupling_mode, "legacy")
        self.assertEqual(model.layers[0].token_reduction, "sqrt_length")

    def test_legacy_gaussian_token_coordinates_match_basis_centers(self):
        descriptor = ACEV2MambaTokenizer(
            r_max=5.0,
            l_max=1,
            num_radial=4,
            hidden_dim=8,
            correlation_order=3,
            correlation_channels=4,
            radial_basis_type="gaussian",
            tokenizer_type="legacy_basis",
        )
        expected = descriptor.radial_basis.basis.centers / descriptor.r_max
        torch.testing.assert_close(descriptor.token_coordinate, expected)

    def test_trainable_radial_scales_remain_positive(self):
        descriptor = ACEV2MambaTokenizer(
            r_max=5.0,
            l_max=1,
            num_radial=4,
            hidden_dim=8,
            correlation_order=3,
            correlation_channels=4,
            radial_basis_type="gaussian",
            radial_trainable=True,
        )
        with torch.no_grad():
            descriptor.radial_basis.basis.raw_widths.fill_(-100.0)
        self.assertTrue(torch.all(descriptor.radial_basis.basis.positive_widths > 0.0))
        centers = descriptor.radial_basis.basis.radial_centers
        self.assertTrue(torch.all(centers[1:] > centers[:-1]))
        self.assertGreater(float(centers[0].detach()), 0.0)
        self.assertLess(float(centers[-1].detach()), descriptor.r_max)
        values = descriptor.radial_basis(torch.linspace(0.0, 5.0, 11))
        self.assertTrue(torch.isfinite(values).all())

        bessel = ACEV2MambaTokenizer(
            r_max=5.0,
            l_max=1,
            num_radial=4,
            hidden_dim=8,
            correlation_order=3,
            correlation_channels=4,
            radial_basis_type="bessel",
            radial_trainable=True,
        )
        frequencies = bessel.radial_basis.basis.frequencies
        self.assertTrue(torch.all(frequencies[1:] > frequencies[:-1]))

    def test_physical_shells_are_sparse_partition_of_unity(self):
        basis = CompactRadialShellBasis(5.0, 17).double()
        distances = torch.linspace(0.0, 5.0, 101, dtype=torch.float64)
        weights = basis.dense(distances)
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones_like(distances),
            atol=2.0e-15,
            rtol=2.0e-15,
        )
        self.assertTrue(torch.all(weights >= 0.0))
        self.assertLessEqual(int((weights > 0.0).sum(dim=-1).max()), 4)
        torch.testing.assert_close(
            basis.centers,
            torch.linspace(0.0, 1.0, 17, dtype=torch.float64),
        )

        legacy = CompactRadialShellBasis(
            5.0, 17, shell_coupling_mode="legacy"
        ).double()
        torch.testing.assert_close(
            legacy.dense(distances).sum(dim=-1),
            legacy.cutoff(distances),
            atol=2.0e-15,
            rtol=2.0e-15,
        )

    def test_physical_shells_are_c2_at_knots(self):
        basis = CompactRadialShellBasis(4.0, 9).double()

        def values(radius):
            return basis.dense(radius.reshape(1))[0]

        knot = torch.tensor(1.5, dtype=torch.float64)
        epsilon = 1.0e-7
        left = knot - epsilon
        right = knot + epsilon
        for order in range(3):
            if order == 0:
                left_value, right_value = values(left), values(right)
            elif order == 1:
                left_value = torch.autograd.functional.jacobian(values, left)
                right_value = torch.autograd.functional.jacobian(values, right)
            else:
                left_value = torch.stack(
                    [
                        torch.autograd.functional.hessian(
                            lambda radius, index=index: values(radius)[index], left
                        )
                        for index in range(basis.num_shells)
                    ]
                )
                right_value = torch.stack(
                    [
                        torch.autograd.functional.hessian(
                            lambda radius, index=index: values(radius)[index], right
                        )
                        for index in range(basis.num_shells)
                    ]
                )
            torch.testing.assert_close(left_value, right_value, atol=2.0e-5, rtol=2.0e-5)

    def test_physical_tokens_reconstruct_density_and_remain_c2_at_cutoff(self):
        torch.manual_seed(105)
        descriptor = ACEV2MambaTokenizer(
            r_max=4.0,
            l_max=1,
            num_radial=4,
            num_shells=13,
            hidden_dim=8,
            correlation_order=3,
            correlation_channels=4,
            tokenizer_type="physical_shells",
            shell_coupling_mode="conservative",
        ).double().eval()
        attributes, edge_index, edge_vector, edge_length = geometry()
        density, edge_features, _ = descriptor._density(
            attributes.double(),
            edge_index,
            edge_vector.double(),
            edge_length.double(),
        )
        tokens = descriptor.pool_edge_features(
            edge_features,
            edge_index[1],
            edge_length.double(),
            attributes.shape[0],
        )
        torch.testing.assert_close(
            tokens.sum(dim=1), density, atol=2.0e-12, rtol=2.0e-12
        )

        pair_attributes = torch.randn(2, 8, dtype=torch.float64)
        pair_index = torch.tensor([[1], [0]], dtype=torch.long)
        zero = torch.zeros((), dtype=torch.float64)

        def token_values(radius):
            pair_vector = torch.stack((radius, zero, zero)).reshape(1, 3)
            return descriptor(
                pair_attributes,
                pair_index,
                pair_vector,
                radius.reshape(1),
            )[1].reshape(-1)

        cutoff = torch.tensor(4.0, dtype=torch.float64)
        value = token_values(cutoff)
        first = torch.autograd.functional.jacobian(token_values, cutoff)
        second = torch.stack(
            [
                torch.autograd.functional.hessian(
                    lambda radius, index=index: token_values(radius)[index], cutoff
                )
                for index in range(value.numel())
            ]
        )
        torch.testing.assert_close(value, torch.zeros_like(value), atol=0.0, rtol=0.0)
        torch.testing.assert_close(first, torch.zeros_like(first), atol=0.0, rtol=0.0)
        torch.testing.assert_close(second, torch.zeros_like(second), atol=0.0, rtol=0.0)

    def test_physical_tokenizer_is_end_to_end_invariant_to_radial_basis_gauge(self):
        parameters = dict(
            r_max=4.5,
            l_max=1,
            num_radial=4,
            num_shells=11,
            hidden_dim=8,
            correlation_order=3,
            correlation_channels=4,
            tokenizer_type="physical_shells",
        )
        torch.manual_seed(104)
        descriptor = ACEV2MambaTokenizer(**parameters).double().eval()
        transformed = ACEV2MambaTokenizer(**parameters).double().eval()
        transformed.load_state_dict(descriptor.state_dict())

        orthogonal, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))

        class OrthogonalBasis(torch.nn.Module):
            def __init__(self, basis, matrix):
                super().__init__()
                self.basis = basis
                self.register_buffer("matrix", matrix)

            def forward(self, distances):
                return self.basis(distances) @ self.matrix.T

        transformed.radial_basis.basis = OrthogonalBasis(
            transformed.radial_basis.basis, orthogonal
        )
        with torch.no_grad():
            first_weight = transformed.radial_net.layer0.weight
            first_weight.copy_(orthogonal @ first_weight)

        attributes, edge_index, edge_vector, edge_length = geometry()
        inputs = (
            attributes.double(),
            edge_index,
            edge_vector.double(),
            edge_length.double(),
        )
        expected = descriptor(*inputs)
        actual = transformed(*inputs)
        torch.testing.assert_close(actual[0], expected[0], atol=2.0e-11, rtol=2.0e-11)
        torch.testing.assert_close(actual[1], expected[1], atol=2.0e-11, rtol=2.0e-11)
        torch.testing.assert_close(actual[2], expected[2], atol=0.0, rtol=0.0)
        torch.testing.assert_close(actual[3], expected[3], atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
