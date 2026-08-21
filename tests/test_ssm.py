import math
import unittest

import torch
from e3nn import o3

from mtace.ssm import (
    IrrepDropout,
    MambaSequenceMixer,
    selective_scan_parallel,
    selective_scan_reference,
)


class SelectiveScanTests(unittest.TestCase):
    def test_scalar_recurrence_matches_hand_calculation(self):
        u = torch.tensor([[[1.0, 2.0, -1.0]]], dtype=torch.float64)
        delta = torch.tensor([[[0.2, 0.1, 0.3]]], dtype=torch.float64)
        A = torch.tensor([[-2.0]], dtype=torch.float64)
        B = torch.tensor([[[0.5, 1.0, -0.25]]], dtype=torch.float64)
        C = torch.tensor([[[1.5, -0.5, 2.0]]], dtype=torch.float64)
        result = selective_scan_reference(
            u, delta, A, B, C, delta_softplus=False
        )
        state = 0.0
        expected = []
        for step in range(3):
            dt = float(delta[0, 0, step])
            state = math.exp(dt * float(A[0, 0])) * state
            state += dt * float(B[0, 0, step]) * float(u[0, 0, step])
            expected.append(state * float(C[0, 0, step]))
        torch.testing.assert_close(
            result,
            torch.tensor(expected, dtype=result.dtype).reshape(1, 1, 3),
            atol=1.0e-6,
            rtol=1.0e-6,
        )

    def test_scan_has_finite_first_and_second_derivatives(self):
        torch.manual_seed(4)
        u = torch.randn(2, 3, 5, requires_grad=True)
        delta = torch.randn(2, 3, 5, requires_grad=True)
        A = -torch.exp(torch.randn(3, 4, requires_grad=True))
        B = torch.randn(2, 4, 5, requires_grad=True)
        C = torch.randn(2, 4, 5, requires_grad=True)
        output = selective_scan_reference(u, delta, A, B, C)
        gradient = torch.autograd.grad(output.square().sum(), u, create_graph=True)[0]
        second = torch.autograd.grad(gradient.square().sum(), u)[0]
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.isfinite(second).all())

    def test_associative_scan_matches_reference_and_second_derivatives(self):
        torch.manual_seed(44)
        inputs = torch.randn(2, 4, 11, requires_grad=True)
        delta = torch.randn(2, 4, 11, requires_grad=True)
        A = -torch.exp(torch.randn(4, 3))
        B = torch.randn(2, 3, 11)
        C = torch.randn(2, 3, 11)
        D = torch.randn(4)
        z = torch.randn(2, 4, 11)
        reference = selective_scan_reference(inputs, delta, A, B, C, D, z)
        parallel = selective_scan_parallel(inputs, delta, A, B, C, D, z)
        torch.testing.assert_close(parallel, reference, atol=2.0e-5, rtol=2.0e-5)
        first = torch.autograd.grad(parallel.square().sum(), inputs, create_graph=True)[0]
        second = torch.autograd.grad(first.square().sum(), inputs)[0]
        self.assertTrue(torch.isfinite(second).all())

    def test_parallel_scan_matches_all_reference_gradients_in_float64(self):
        torch.manual_seed(45)
        tensors = [
            torch.randn(1, 3, 7, dtype=torch.float64, requires_grad=True),
            torch.randn(1, 3, 7, dtype=torch.float64, requires_grad=True),
            -torch.rand(3, 4, dtype=torch.float64, requires_grad=True),
            torch.randn(1, 4, 7, dtype=torch.float64, requires_grad=True),
            torch.randn(1, 4, 7, dtype=torch.float64, requires_grad=True),
            torch.randn(3, dtype=torch.float64, requires_grad=True),
            torch.randn(1, 3, 7, dtype=torch.float64, requires_grad=True),
            torch.randn(3, dtype=torch.float64, requires_grad=True),
        ]
        reference = selective_scan_reference(*tensors[:5], *tensors[5:7], tensors[7])
        parallel = selective_scan_parallel(*tensors[:5], *tensors[5:7], tensors[7])
        torch.testing.assert_close(parallel, reference, atol=2.0e-12, rtol=2.0e-12)
        reference_gradients = torch.autograd.grad(reference.square().sum(), tensors, retain_graph=True)
        parallel_gradients = torch.autograd.grad(parallel.square().sum(), tensors)
        for actual, expected in zip(parallel_gradients, reference_gradients):
            torch.testing.assert_close(actual, expected, atol=2.0e-11, rtol=2.0e-11)

    def test_float64_scan_is_not_silently_demoted(self):
        u = torch.tensor([[[1.00000003, -0.99999997, 0.50000001]]], dtype=torch.float64)
        delta = torch.tensor([[[0.123456789, 0.234567891, 0.345678912]]], dtype=torch.float64)
        A = torch.tensor([[-0.876543219]], dtype=torch.float64)
        B = torch.tensor([[[0.333333337, -0.222222229, 0.111111113]]], dtype=torch.float64)
        C = torch.tensor([[[0.777777779, -0.666666667, 0.555555557]]], dtype=torch.float64)
        result = selective_scan_parallel(u, delta, A, B, C, delta_softplus=False)
        state = torch.zeros((), dtype=torch.float64)
        expected = []
        for step in range(u.shape[-1]):
            state = torch.exp(delta[0, 0, step] * A[0, 0]) * state
            state = state + delta[0, 0, step] * B[0, 0, step] * u[0, 0, step]
            expected.append(state * C[0, 0, step])
        expected = torch.stack(expected).reshape_as(result)
        torch.testing.assert_close(result, expected, atol=1.0e-14, rtol=1.0e-14)

    def test_bidirectional_mixer_shape_and_gradients(self):
        torch.manual_seed(5)
        mixer = MambaSequenceMixer(12, d_state=4, d_conv=3, backend="torch")
        hidden = torch.randn(3, 9, 12, requires_grad=True)
        output = mixer(hidden)
        self.assertEqual(output.shape, hidden.shape)
        output.sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertIsNotNone(mixer.backward_A_log.grad)

    def test_tied_bidirectional_compatibility_mode(self):
        mixer = MambaSequenceMixer(
            8, d_state=3, d_conv=2, bidirectional_tied=True, backend="torch"
        )
        self.assertFalse(hasattr(mixer, "backward_A_log"))
        hidden = torch.randn(2, 5, 8)
        output = mixer(hidden)
        reflected = torch.flip(mixer(torch.flip(hidden, dims=(1,))), dims=(1,))
        torch.testing.assert_close(output, reflected, atol=2.0e-6, rtol=2.0e-6)

    def test_dropout_ties_each_irrep_copy_over_magnetic_components(self):
        irreps = o3.Irreps("2x1o + 1x2e")
        dropout = IrrepDropout(irreps, 0.5).train()
        torch.manual_seed(12)
        output = dropout(torch.ones(32, irreps.dim))
        for (multiplicity, irrep), component_slice in zip(irreps, irreps.slices()):
            block = output[:, component_slice].reshape(32, multiplicity, irrep.dim)
            torch.testing.assert_close(block, block[..., :1].expand_as(block))


if __name__ == "__main__":
    unittest.main()
