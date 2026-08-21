import unittest

import torch

from mtace.model import MambaACE


def model():
    torch.manual_seed(17)
    return MambaACE(
        r_max=4.5,
        l_max=2,
        num_radial=4,
        hidden_dim=8,
        num_layers=1,
        correlation_order=4,
        correlation_channels=4,
        mamba_dim=12,
        mamba_d_state=4,
        mamba_backend="torch",
    ).eval()


def data(positions, species=None, edge_permutation=None):
    count = positions.shape[0]
    senders, receivers = [], []
    for receiver in range(count):
        for sender in range(count):
            if sender != receiver:
                senders.append(sender)
                receivers.append(receiver)
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    if edge_permutation is not None:
        edge_index = edge_index[:, edge_permutation]
    return {
        "z": torch.tensor(species or [8, 1, 1, 8][:count], dtype=torch.long),
        "pos": positions,
        "cell": torch.eye(3) * 9.0,
        "edge_index": edge_index,
        "edge_shift": torch.zeros((edge_index.shape[1], 3)),
        "volume": torch.tensor(729.0),
    }


POSITIONS = torch.tensor(
    [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]]
)


class SymmetryTests(unittest.TestCase):
    def test_rotation_inversion_translation_and_force_covariance(self):
        network = model()
        energy, forces, _, _ = network(data(POSITIONS), compute_stress=False)
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        rotated_energy, rotated_forces, _, _ = network(
            data(POSITIONS @ rotation.T), compute_stress=False
        )
        torch.testing.assert_close(rotated_energy, energy, atol=2.0e-5, rtol=2.0e-5)
        torch.testing.assert_close(
            rotated_forces, forces @ rotation.T, atol=2.0e-4, rtol=2.0e-4
        )
        inverted_energy, inverted_forces, _, _ = network(
            data(-POSITIONS), compute_stress=False
        )
        torch.testing.assert_close(inverted_energy, energy, atol=2.0e-5, rtol=2.0e-5)
        torch.testing.assert_close(inverted_forces, -forces, atol=2.0e-4, rtol=2.0e-4)
        translated_energy, translated_forces, _, _ = network(
            data(POSITIONS + torch.tensor([2.3, -1.1, 0.4])), compute_stress=False
        )
        torch.testing.assert_close(translated_energy, energy, atol=2.0e-5, rtol=2.0e-5)
        torch.testing.assert_close(translated_forces, forces, atol=2.0e-4, rtol=2.0e-4)

    def test_atom_and_neighbor_list_permutations(self):
        network = model()
        species = [8, 1, 6, 14]
        energy = network(data(POSITIONS, species), compute_stress=False)[0]
        atom_permutation = torch.tensor([2, 0, 3, 1])
        permuted_species = [species[index] for index in atom_permutation.tolist()]
        permuted_energy = network(
            data(POSITIONS[atom_permutation], permuted_species), compute_stress=False
        )[0]
        torch.testing.assert_close(permuted_energy, energy, atol=2.0e-5, rtol=2.0e-5)

        edge_count = data(POSITIONS)["edge_index"].shape[1]
        edge_permutation = torch.randperm(edge_count, generator=torch.Generator().manual_seed(9))
        reordered_energy = network(
            data(POSITIONS, species, edge_permutation=edge_permutation),
            compute_stress=False,
        )[0]
        torch.testing.assert_close(reordered_energy, energy, atol=2.0e-5, rtol=2.0e-5)

    def test_central_species_changes_energy(self):
        network = model()
        oxygen = network(data(POSITIONS, [8, 1, 1, 8]), compute_stress=False)[0]
        carbon = network(data(POSITIONS, [6, 1, 1, 8]), compute_stress=False)[0]
        self.assertGreater(float((oxygen - carbon).detach().abs()), 1.0e-7)

    def test_periodic_image_gauge_leaves_energy_and_forces_unchanged(self):
        network = model()
        original = data(POSITIONS)
        energy, forces, _, _ = network(original, compute_stress=False)

        wrapped_positions = POSITIONS.clone()
        wrapped_positions[0] += original["cell"][0]
        wrapped = data(wrapped_positions)
        sender, receiver = wrapped["edge_index"]
        wrapped["edge_shift"][:, 0] += (receiver == 0).to(
            wrapped["edge_shift"].dtype
        )
        wrapped["edge_shift"][:, 0] -= (sender == 0).to(
            wrapped["edge_shift"].dtype
        )
        wrapped_energy, wrapped_forces, _, _ = network(
            wrapped, compute_stress=False
        )
        torch.testing.assert_close(wrapped_energy, energy, atol=2.0e-5, rtol=2.0e-5)
        torch.testing.assert_close(wrapped_forces, forces, atol=2.0e-4, rtol=2.0e-4)

    def test_stress_is_symmetric_and_matches_finite_difference(self):
        network = model()
        _, _, stress, _ = network(data(POSITIONS), compute_stress=True)
        torch.testing.assert_close(stress, stress.T)
        step = 1.0e-3
        for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
            plus = torch.eye(3)
            minus = torch.eye(3)
            plus[a, b] += step
            minus[a, b] -= step
            if a != b:
                plus[b, a] += step
                minus[b, a] -= step
            e_plus = network(data(POSITIONS @ plus), compute_stress=False)[0]
            e_minus = network(data(POSITIONS @ minus), compute_stress=False)[0]
            derivative = (e_plus - e_minus) / (2.0 * step * 729.0)
            expected = stress[a, b] * (2.0 if a != b else 1.0)
            torch.testing.assert_close(derivative, expected, atol=3.0e-5, rtol=3.0e-2)

        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        _, _, rotated_stress, _ = network(
            data(POSITIONS @ rotation.T), compute_stress=True
        )
        torch.testing.assert_close(
            rotated_stress,
            rotation @ stress @ rotation.T,
            atol=3.0e-5,
            rtol=3.0e-2,
        )

    def test_force_loss_backpropagates_second_derivatives(self):
        network = model().train()
        energy, forces, _, _ = network(data(POSITIONS), training=True, compute_stress=False)
        loss = energy.square() + forces.square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in network.parameters() if parameter.requires_grad]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients if g is not None))

    def test_full_model_preserves_float64_energy_and_forces(self):
        network = model().double()
        energy, forces, _, _ = network(
            data(POSITIONS.double()), training=True, compute_stress=False
        )
        self.assertEqual(energy.dtype, torch.float64)
        self.assertEqual(forces.dtype, torch.float64)
        dynamics = network.layers[0].mixer.forward_direction.dt_bias
        second = torch.autograd.grad(forces.square().sum(), dynamics)[0]
        self.assertTrue(torch.isfinite(second).all())

    def test_atomic_number_domain_is_validated(self):
        network = model()
        with self.assertRaisesRegex(ValueError, "1 <= Z <= 118"):
            network(data(POSITIONS, [119, 1, 1, 8]), compute_stress=False)

    def test_singular_stress_cell_and_invalid_edges_are_rejected(self):
        network = model().double()
        singular = data(POSITIONS.double())
        singular["cell"] = torch.zeros(3, 3, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "full-rank cell"):
            network(singular, compute_stress=True)

        invalid_edge = data(POSITIONS.double())
        invalid_edge["edge_index"] = invalid_edge["edge_index"].clone()
        invalid_edge["edge_index"][0, 0] = len(POSITIONS)
        with self.assertRaisesRegex(ValueError, "outside the structure"):
            network(invalid_edge, compute_stress=False)


if __name__ == "__main__":
    unittest.main()
