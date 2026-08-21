import unittest

import torch

from mtace.mamba3 import (
    Mamba3Direction,
    Mamba3RMSNorm,
    Mamba3SequenceMixer,
    heavy_tail_activation,
    mamba3_angle_increments,
    mamba3_mimo_scan_parallel,
    mamba3_mimo_scan_reference,
    mamba3_scan_parallel,
    mamba3_scan_reference,
    rotate_state_pairs,
    rotate_state_halves,
)
from mtace.model import MambaACEV2


class Mamba3Tests(unittest.TestCase):
    def test_model_rejects_invalid_rank_and_chunk_before_default_resolution(self):
        with self.assertRaisesRegex(ValueError, "mamba_mimo_rank"):
            MambaACEV2(mamba_mimo_rank=0)
        with self.assertRaisesRegex(ValueError, "mamba_chunk_size"):
            MambaACEV2(mamba_chunk_size=0)

    def test_unsupported_fused_mimo_chunk_is_caught_before_kernel_dispatch(self):
        direction = Mamba3Direction(
            d_model=8,
            d_state=4,
            expand=2,
            headdim=8,
            chunk_size=4,
            mimo_rank=4,
            backend="auto",
        )
        self.assertIn("chunk_size >= 8", direction.fused_configuration_error)
        hidden = torch.randn(1, 5, 8)
        portable = direction(hidden)
        self.assertTrue(torch.isfinite(portable).all())

        required = Mamba3Direction(
            d_model=8,
            d_state=4,
            expand=2,
            headdim=8,
            chunk_size=4,
            mimo_rank=4,
            backend="cuda",
        )
        with self.assertRaisesRegex(RuntimeError, "chunk_size >= 8"):
            required(hidden)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_fp32_cuda_inference_autocasts_only_fused_mamba_projection(self):
        direction = Mamba3Direction(
            d_model=32,
            d_state=16,
            expand=2,
            headdim=64,
            chunk_size=64,
            backend="cuda",
        ).cuda()
        if not direction.accelerated_backend_available:
            self.skipTest("official fused Mamba-3 SISO kernel is unavailable")
        hidden = torch.randn(2, 16, 32, device="cuda", requires_grad=True)
        output = direction(hidden)
        self.assertEqual(output.dtype, torch.float32)
        gradient = torch.autograd.grad(output.square().sum(), hidden)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_fp32_cuda_mimo_inference_has_finite_first_backward(self):
        direction = Mamba3Direction(
            d_model=32,
            d_state=64,
            expand=2,
            headdim=64,
            chunk_size=16,
            mimo_rank=4,
            backend="cuda",
        ).cuda()
        if not direction.accelerated_backend_available:
            self.skipTest("official fused Mamba-3 MIMO kernel is unavailable")
        hidden = torch.randn(1, 32, 32, device="cuda", requires_grad=True)
        output = direction(hidden)
        self.assertEqual(output.dtype, torch.float32)
        gradient = torch.autograd.grad(output.square().sum(), hidden)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    def test_rms_norm_preserves_low_precision_output_dtype(self):
        norm = Mamba3RMSNorm(8)
        values = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        self.assertEqual(norm(values).dtype, torch.bfloat16)

    def test_heavy_tail_activation_matches_piecewise_definition(self):
        values = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=torch.float64)
        expected = torch.tensor([0.25, 2.0 / 3.0, 1.0, 1.5, 4.0], dtype=torch.float64)
        torch.testing.assert_close(heavy_tail_activation(values), expected)

    def test_rotary_pairs_preserve_norm_and_leave_tail_unchanged(self):
        vectors = torch.randn(2, 4, 3, 6, dtype=torch.float64)
        angles = torch.randn(2, 4, 3, 2, dtype=torch.float64)
        rotated = rotate_state_pairs(vectors, angles)
        torch.testing.assert_close(
            rotated[..., :4].square().sum(dim=-1),
            vectors[..., :4].square().sum(dim=-1),
            atol=2.0e-12,
            rtol=2.0e-12,
        )
        torch.testing.assert_close(rotated[..., 4:], vectors[..., 4:])

        mimo_vectors = torch.randn(2, 4, 5, 3, 6, dtype=torch.float64)
        mimo_angles = torch.randn(2, 4, 5, 3, 2, dtype=torch.float64)
        mimo_rotated = rotate_state_halves(mimo_vectors, mimo_angles)
        torch.testing.assert_close(
            mimo_rotated.square().sum(dim=-1),
            mimo_vectors.square().sum(dim=-1),
            atol=2.0e-12,
            rtol=2.0e-12,
        )
        torch.testing.assert_close(mimo_rotated[..., 2], mimo_vectors[..., 2])
        torch.testing.assert_close(mimo_rotated[..., 5], mimo_vectors[..., 5])
        cosine, sine = torch.cos(mimo_angles), torch.sin(mimo_angles)
        torch.testing.assert_close(
            mimo_rotated[..., :2],
            mimo_vectors[..., :2] * cosine - mimo_vectors[..., 3:5] * sine,
        )
        torch.testing.assert_close(
            mimo_rotated[..., 3:5],
            mimo_vectors[..., :2] * sine + mimo_vectors[..., 3:5] * cosine,
        )

    def test_official_angles_are_raw_and_legacy_mode_is_explicit(self):
        raw = torch.tensor([-4.0, -0.5, 0.0, 0.75, 5.0], dtype=torch.float64)
        torch.testing.assert_close(mamba3_angle_increments(raw), raw)
        torch.testing.assert_close(
            mamba3_angle_increments(raw, "legacy_bounded"),
            torch.pi * torch.tanh(raw),
        )
        with self.assertRaisesRegex(ValueError, "angle_mode"):
            mamba3_angle_increments(raw, "bounded")

    def test_exponential_trapezoidal_recurrence_matches_hand_calculation(self):
        x = torch.tensor([[[[1.0]], [[2.0]], [[-0.5]]]], dtype=torch.float64)
        k = torch.tensor([[[[0.5, -0.2]], [[0.3, 0.7]], [[-0.4, 0.1]]]], dtype=torch.float64)
        q = torch.tensor([[[[0.8, 0.4]], [[-0.1, 0.6]], [[0.2, -0.9]]]], dtype=torch.float64)
        alpha = torch.tensor([[[0.9], [0.8], [0.7]]], dtype=torch.float64)
        beta = torch.tensor([[[0.0], [0.12], [0.08]]], dtype=torch.float64)
        gamma = torch.tensor([[[0.2], [0.15], [0.1]]], dtype=torch.float64)
        actual = mamba3_scan_reference(x, k, q, alpha, beta, gamma)

        state = torch.zeros(2, dtype=torch.float64)
        expected = []
        previous = torch.zeros(2, dtype=torch.float64)
        for step in range(3):
            current = x[0, step, 0, 0] * k[0, step, 0]
            state = alpha[0, step, 0] * state
            state = state + beta[0, step, 0] * previous + gamma[0, step, 0] * current
            expected.append(torch.dot(state, q[0, step, 0]))
            previous = current
        expected = torch.stack(expected).reshape_as(actual)
        torch.testing.assert_close(actual, expected, atol=1.0e-14, rtol=1.0e-14)

    def test_parallel_scan_matches_reference_and_all_gradients(self):
        torch.manual_seed(71)
        tensors = [
            torch.randn(2, 7, 3, 2, dtype=torch.float64, requires_grad=True),
            torch.randn(2, 7, 3, 4, dtype=torch.float64, requires_grad=True),
            torch.randn(2, 7, 3, 4, dtype=torch.float64, requires_grad=True),
            torch.sigmoid(torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True)),
            torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True),
            torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True),
            torch.randn(3, dtype=torch.float64, requires_grad=True),
            torch.randn(2, 7, 3, 2, dtype=torch.float64, requires_grad=True),
        ]
        reference = mamba3_scan_reference(*tensors)
        parallel = mamba3_scan_parallel(*tensors)
        torch.testing.assert_close(parallel, reference, atol=3.0e-12, rtol=3.0e-12)
        reference_gradients = torch.autograd.grad(reference.square().sum(), tensors, retain_graph=True)
        parallel_gradients = torch.autograd.grad(parallel.square().sum(), tensors)
        for actual, expected in zip(parallel_gradients, reference_gradients):
            torch.testing.assert_close(actual, expected, atol=5.0e-11, rtol=5.0e-11)

    def test_mimo_parallel_scan_matches_reference_and_all_gradients(self):
        torch.manual_seed(73)
        tensors = [
            torch.randn(2, 7, 3, 2, 4, dtype=torch.float64, requires_grad=True),
            torch.randn(2, 7, 3, 2, 6, dtype=torch.float64, requires_grad=True),
            torch.randn(2, 7, 3, 2, 6, dtype=torch.float64, requires_grad=True),
            torch.sigmoid(torch.randn(2, 7, 2, dtype=torch.float64, requires_grad=True)),
            torch.randn(2, 7, 2, dtype=torch.float64, requires_grad=True),
            torch.randn(2, 7, 2, dtype=torch.float64, requires_grad=True),
            torch.randn(2, dtype=torch.float64, requires_grad=True),
            torch.randn(2, 7, 3, 2, 4, dtype=torch.float64, requires_grad=True),
        ]
        reference = mamba3_mimo_scan_reference(*tensors)
        parallel = mamba3_mimo_scan_parallel(*tensors)
        torch.testing.assert_close(parallel, reference, atol=5.0e-12, rtol=5.0e-12)
        reference_gradients = torch.autograd.grad(
            reference.square().sum(), tensors, retain_graph=True
        )
        parallel_gradients = torch.autograd.grad(parallel.square().sum(), tensors)
        for actual, expected in zip(parallel_gradients, reference_gradients):
            torch.testing.assert_close(actual, expected, atol=8.0e-11, rtol=8.0e-11)

    def test_direction_matches_official_projection_layout(self):
        direction = Mamba3Direction(
            d_model=8,
            d_state=6,
            expand=2,
            headdim=8,
            rope_fraction=0.5,
            backend="torch",
        )
        expected = 2 * 16 + 2 * 6 + 3 * 2 + 1
        self.assertEqual(direction.in_proj.out_features, expected)
        self.assertFalse(hasattr(direction, "conv1d"))
        initialized_dt = torch.nn.functional.softplus(direction.dt_bias.detach())
        self.assertTrue(torch.all(initialized_dt >= 1.0e-3))
        self.assertTrue(torch.all(initialized_dt <= 1.0e-1))

        mimo = Mamba3Direction(
            d_model=8,
            d_state=6,
            expand=2,
            headdim=8,
            rope_fraction=0.5,
            mimo_rank=4,
            backend="torch",
        )
        mimo_expected = 2 * 16 + 2 * 6 * 4 + 3 * 2 + 1
        self.assertEqual(mimo.in_proj.out_features, mimo_expected)
        self.assertEqual(mimo.B_bias.shape, (2, 4, 6))
        self.assertEqual(mimo.mimo_x.shape, (2, 4, 8))
        torch.testing.assert_close(
            mimo.mimo_x,
            torch.full_like(mimo.mimo_x, 0.25),
        )

    def test_bidirectional_mixer_has_finite_second_derivatives(self):
        torch.manual_seed(72)
        mixer = Mamba3SequenceMixer(
            d_model=8,
            d_state=4,
            expand=2,
            headdim=8,
            backend="torch",
        ).double()
        hidden = torch.randn(2, 6, 8, dtype=torch.float64, requires_grad=True)
        output = mixer(hidden, require_higher_order=True)
        first = torch.autograd.grad(output.square().sum(), hidden, create_graph=True)[0]
        second = torch.autograd.grad(first.square().sum(), hidden)[0]
        self.assertEqual(output.shape, hidden.shape)
        self.assertTrue(torch.isfinite(second).all())
        self.assertIsNotNone(mixer.backward_direction)

        mimo = Mamba3SequenceMixer(
            d_model=8,
            d_state=4,
            expand=2,
            headdim=8,
            mimo_rank=3,
            backend="torch",
        ).double()
        mimo_hidden = hidden.detach().clone().requires_grad_(True)
        mimo_output = mimo(mimo_hidden, require_higher_order=True)
        mimo_first = torch.autograd.grad(
            mimo_output.square().sum(), mimo_hidden, create_graph=True
        )[0]
        mimo_second = torch.autograd.grad(mimo_first.square().sum(), mimo_hidden)[0]
        self.assertTrue(torch.isfinite(mimo_second).all())

    def test_tied_mixer_is_reversal_equivariant(self):
        mixer = Mamba3SequenceMixer(
            d_model=8,
            d_state=4,
            expand=2,
            headdim=8,
            bidirectional_tied=True,
            backend="torch",
        )
        hidden = torch.randn(2, 5, 8)
        output = mixer(hidden)
        reflected = torch.flip(mixer(torch.flip(hidden, dims=(1,))), dims=(1,))
        torch.testing.assert_close(output, reflected, atol=3.0e-6, rtol=3.0e-6)


if __name__ == "__main__":
    unittest.main()
