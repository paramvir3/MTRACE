"""Batched evaluation must be identical to the sequential loop.

The model mixes atoms only through ``edge_index``, so concatenating structures
into one disconnected graph cannot change any per-atom quantity.  That is the
same property as the additivity check ``E(A u B) = E(A) + E(B)``; this module
turns it into the contract the training loop relies on.

A batched forward that is merely *close* to the loop is not acceptable: the
energy is a physical observable and the forces are its exact gradient, so the
tests below compare in FP64 at round-off tolerances rather than at "good
enough" ones.
"""

import unittest

import torch

from mtace.data import collate_structures
from mtace.model import MambaACEV2


def model(seed=17, **overrides):
    torch.manual_seed(seed)
    settings = dict(
        r_max=4.5, l_max=2, num_radial=4, hidden_dim=8, num_layers=1,
        correlation_order=4, correlation_channels=4, mamba_dim=12,
        mamba_d_state=4, mamba_backend="torch", readout_hidden=8,
    )
    settings.update(overrides)
    return MambaACEV2(**settings).double().eval()


def structure(positions, species, cell_size):
    count = positions.shape[0]
    senders, receivers = [], []
    for receiver in range(count):
        for sender in range(count):
            if sender != receiver:
                senders.append(sender)
                receivers.append(receiver)
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    cell = torch.eye(3, dtype=torch.float64) * cell_size
    return {
        "z": torch.tensor(species, dtype=torch.long),
        "pos": positions,
        "cell": cell,
        "edge_index": edge_index,
        "edge_shift": torch.zeros((edge_index.shape[1], 3), dtype=torch.float64),
        "volume": torch.tensor(float(cell_size) ** 3, dtype=torch.float64),
    }


def structures():
    """Three deliberately dissimilar structures: different sizes and cells."""

    torch.manual_seed(3)
    a = structure(
        torch.tensor([[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8]],
                     dtype=torch.float64),
        [8, 1, 1], 9.0,
    )
    b = structure(
        torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.3, 0.2], [0.4, 1.5, 0.1],
                      [0.9, 0.8, 1.6]], dtype=torch.float64),
        [6, 1, 1, 8], 11.0,
    )
    c = structure(
        torch.tensor([[0.2, 0.1, 0.4], [1.1, 1.0, 0.5]], dtype=torch.float64),
        [8, 8], 7.0,
    )
    return [a, b, c]


class BatchedEqualsSequentialTests(unittest.TestCase):
    def test_energies_and_forces_match_the_loop(self):
        network = model()
        items = structures()

        energies, forces = [], []
        for item in items:
            e, f, _, _ = network(item, compute_stress=False)
            energies.append(e)
            forces.append(f)

        batched = collate_structures(items)
        be, bf, _, _ = network(batched, compute_stress=False)

        self.assertEqual(tuple(be.shape), (len(items),))
        torch.testing.assert_close(
            be, torch.stack(energies), atol=1.0e-12, rtol=1.0e-12
        )
        torch.testing.assert_close(
            bf, torch.cat(forces), atol=1.0e-12, rtol=1.0e-12
        )

    def test_stress_matches_the_loop(self):
        network = model()
        items = structures()

        stresses = []
        for item in items:
            _, _, s, _ = network(item, compute_stress=True)
            stresses.append(s)

        batched = collate_structures(items)
        _, _, bs, _ = network(batched, compute_stress=True)

        self.assertEqual(tuple(bs.shape), (len(items), 3, 3))
        torch.testing.assert_close(
            bs, torch.stack(stresses), atol=1.0e-12, rtol=1.0e-12
        )
        for index in range(len(items)):
            torch.testing.assert_close(bs[index], bs[index].T)

    def test_parameter_gradients_match_the_loop(self):
        """The point of batching: one backward must equal the accumulated ones."""

        items = structures()

        loop = model()
        loop.zero_grad()
        for item in items:
            e, f, _, _ = loop(item, training=True, compute_stress=False)
            (e.square() + f.square().sum()).backward()

        batched_model = model()
        batched_model.zero_grad()
        batched = collate_structures(items)
        be, bf, _, _ = batched_model(batched, training=True, compute_stress=False)
        (be.square().sum() + bf.square().sum()).backward()

        compared = 0
        for (name, left), (_, right) in zip(
            loop.named_parameters(), batched_model.named_parameters()
        ):
            if left.grad is None and right.grad is None:
                continue
            self.assertIsNotNone(left.grad, name)
            self.assertIsNotNone(right.grad, name)
            torch.testing.assert_close(
                right.grad, left.grad, atol=1.0e-10, rtol=1.0e-10,
                msg=f"gradient mismatch for {name}",
            )
            compared += 1
        self.assertGreater(compared, 0)

    def test_a_single_structure_batch_reproduces_the_scalar_path(self):
        network = model()
        item = structures()[0]
        e, f, s, _ = network(item, compute_stress=True)
        be, bf, bs, _ = network(collate_structures([item]), compute_stress=True)
        torch.testing.assert_close(be[0], e, atol=1.0e-12, rtol=1.0e-12)
        torch.testing.assert_close(bf, f, atol=1.0e-12, rtol=1.0e-12)
        torch.testing.assert_close(bs[0], s, atol=1.0e-12, rtol=1.0e-12)

    def test_batch_order_does_not_change_results(self):
        network = model()
        items = structures()
        forward = network(collate_structures(items), compute_stress=True)
        reversed_order = network(
            collate_structures(items[::-1]), compute_stress=True
        )
        torch.testing.assert_close(
            reversed_order[0].flip(0), forward[0], atol=1.0e-12, rtol=1.0e-12
        )

    def test_hybrid_and_routed_models_batch_correctly(self):
        """Batching must hold for the schedule and the routed experts too."""

        network = model(
            num_layers=3, mixer_schedule=["mamba", "attention"],
            num_experts=3, expert_hidden=8, router_tau=0.5,
        )
        items = structures()
        energies = [network(item, compute_stress=False)[0] for item in items]
        be, _, _, _ = network(collate_structures(items), compute_stress=False)
        torch.testing.assert_close(
            be, torch.stack(energies), atol=1.0e-12, rtol=1.0e-12
        )


class CollateValidationTests(unittest.TestCase):
    def test_edge_indices_are_offset_and_never_cross_structures(self):
        items = structures()
        batched = collate_structures(items)
        counts = [int(item["z"].numel()) for item in items]
        self.assertEqual(int(batched["z"].numel()), sum(counts))
        self.assertEqual(tuple(batched["cell"].shape), (len(items), 3, 3))
        sender_graph = batched["batch"][batched["edge_index"][0]]
        receiver_graph = batched["batch"][batched["edge_index"][1]]
        self.assertTrue(bool((sender_graph == receiver_graph).all()))

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            collate_structures([])

    def test_a_mismatched_batch_index_is_rejected(self):
        network = model()
        batched = collate_structures(structures())
        batched["batch"] = batched["batch"][:-1]
        with self.assertRaisesRegex(ValueError, "batch must be"):
            network(batched, compute_stress=False)


if __name__ == "__main__":
    unittest.main()
