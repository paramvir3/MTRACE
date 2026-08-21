import unittest

import torch

from mtace.mixers import (
    AttentionSequenceMixer,
    DeepSetsSequenceMixer,
    DenseRadialSequenceMixer,
    IdentitySequenceMixer,
)
from mtace.model import MambaACEV2


class MatchedMixerTests(unittest.TestCase):
    def test_attention_dropout_is_separate_from_common_block_dropout(self):
        common = dict(
            r_max=4.0,
            l_max=1,
            num_radial=3,
            num_shells=7,
            hidden_dim=8,
            num_layers=1,
            correlation_order=3,
            correlation_channels=4,
            mixer_type="attention",
            attention_heads=2,
            mamba_dim=8,
            dropout=0.4,
        )
        matched = MambaACEV2(**common)
        self.assertEqual(matched.layers[0].mixer.attention_dropout.p, 0.0)
        self.assertEqual(matched.layers[0].dropout.probability, 0.4)

        regularized = MambaACEV2(**common, attention_dropout=0.2)
        self.assertEqual(regularized.layers[0].mixer.attention_dropout.p, 0.2)
        with self.assertRaisesRegex(ValueError, "attention_dropout"):
            MambaACEV2(**common, attention_dropout=1.0)

    def test_baseline_mixers_preserve_shape_and_second_derivatives(self):
        mixers = [
            AttentionSequenceMixer(8, num_heads=2),
            DenseRadialSequenceMixer(8, sequence_length=7),
            DeepSetsSequenceMixer(8, expand=2),
            IdentitySequenceMixer(8),
        ]
        for mixer in mixers:
            with self.subTest(mixer=mixer.__class__.__name__):
                hidden = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
                mixer = mixer.double()
                output = mixer(hidden, require_higher_order=True)
                first = torch.autograd.grad(
                    output.square().sum(), hidden, create_graph=True
                )[0]
                second = torch.autograd.grad(first.square().sum(), hidden)[0]
                self.assertEqual(output.shape, hidden.shape)
                self.assertTrue(torch.isfinite(second).all())

    def test_attention_and_deepsets_are_token_permutation_equivariant(self):
        hidden = torch.randn(2, 9, 8)
        permutation = torch.randperm(9)
        for mixer in (AttentionSequenceMixer(8, 2), DeepSetsSequenceMixer(8)):
            mixer.eval()
            expected = mixer(hidden)[:, permutation]
            actual = mixer(hidden[:, permutation])
            torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-6)

    def test_all_model_mixers_receive_identical_physical_tokens(self):
        common = dict(
            r_max=4.0,
            l_max=1,
            num_radial=3,
            num_shells=9,
            hidden_dim=8,
            num_layers=1,
            correlation_order=3,
            correlation_channels=4,
            tokenizer_type="physical_shells",
            attention_heads=2,
            mamba_dim=8,
            mamba_d_state=4,
            mamba_headdim=4,
            mamba_mimo_rank=1,
            mamba_backend="torch",
        )
        torch.manual_seed(81)
        reference = MambaACEV2(**common, mixer_type="mamba")
        models = [reference]
        for mixer_type in ("attention", "dense", "mlp", "identity"):
            model = MambaACEV2(**common, mixer_type=mixer_type)
            model.species_embedding.load_state_dict(reference.species_embedding.state_dict())
            model.ace.load_state_dict(reference.ace.state_dict())
            models.append(model)

        z = torch.tensor([6, 8, 1], dtype=torch.long)
        positions = torch.tensor(
            [[0.1, 0.2, 0.3], [1.1, 0.4, 0.8], [0.6, 1.2, 0.7]]
        )
        edge_index = torch.tensor(
            [[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]], dtype=torch.long
        )
        edge_vector = positions[edge_index[0]] - positions[edge_index[1]]
        edge_length = torch.linalg.vector_norm(edge_vector, dim=-1)
        outputs = [
            model.ace(
                model.species_embedding(z), edge_index, edge_vector, edge_length
            )
            for model in models
        ]
        for output in outputs[1:]:
            torch.testing.assert_close(output[0], outputs[0][0], atol=0.0, rtol=0.0)
            torch.testing.assert_close(output[1], outputs[0][1], atol=0.0, rtol=0.0)
            torch.testing.assert_close(output[3], outputs[0][3], atol=0.0, rtol=0.0)
        self.assertTrue(all(model.ace.sequence_length == 9 for model in models))

    def test_conservative_token_reduction_is_shell_resolution_independent(self):
        common = dict(
            r_max=4.0,
            l_max=1,
            num_radial=3,
            num_shells=8,
            hidden_dim=8,
            num_layers=1,
            correlation_order=3,
            correlation_channels=4,
            mixer_type="identity",
            mamba_dim=8,
        )
        conservative = MambaACEV2(
            **common, shell_coupling_mode="conservative"
        ).layers[0]
        legacy = MambaACEV2(**common, shell_coupling_mode="legacy").layers[0]
        value = torch.randn(2, conservative.node_irreps.dim)
        for length in (4, 16, 31):
            partition = torch.rand(2, length)
            partition = partition / partition.sum(dim=1, keepdim=True)
            weighted = partition[:, :, None] * value[:, None, :]
            torch.testing.assert_close(
                conservative._reduce_token_updates(weighted), value
            )
            torch.testing.assert_close(
                legacy._reduce_token_updates(weighted), value / length**0.5
            )


if __name__ == "__main__":
    unittest.main()
