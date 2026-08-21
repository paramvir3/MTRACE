import unittest

import torch

from mtace.physics import SmoothPolynomialCutoff


class CutoffTests(unittest.TestCase):
    def test_value_first_and_second_derivatives_vanish(self):
        cutoff = SmoothPolynomialCutoff(5.0)
        radius = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        value = cutoff(radius)
        first = torch.autograd.grad(value, radius, create_graph=True)[0]
        second = torch.autograd.grad(first, radius)[0]
        torch.testing.assert_close(value, torch.zeros_like(value))
        torch.testing.assert_close(first, torch.zeros_like(first))
        torch.testing.assert_close(second, torch.zeros_like(second))


if __name__ == "__main__":
    unittest.main()
