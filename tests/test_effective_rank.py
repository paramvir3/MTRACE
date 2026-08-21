"""Equivariant effective rank of the ACE token features."""

import math
import unittest

import torch
from e3nn import o3

from mtace.diagnostics import (
    IrrepGramAccumulator,
    format_effective_rank,
    participation_ratio,
    token_effective_rank,
)
from mtace.model import MambaACEV2


def data(positions, species=None):
    count = positions.shape[0]
    senders, receivers = [], []
    for receiver in range(count):
        for sender in range(count):
            if sender != receiver:
                senders.append(sender)
                receivers.append(receiver)
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    return {
        "z": torch.tensor(species or [8, 1, 1, 8][:count], dtype=torch.long),
        "pos": positions,
        "cell": torch.eye(3, dtype=positions.dtype) * 9.0,
        "edge_index": edge_index,
        "edge_shift": torch.zeros((edge_index.shape[1], 3), dtype=positions.dtype),
    }


POSITIONS = torch.tensor(
    [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]],
    dtype=torch.float64,
)


def random_rotation(seed):
    """A generic SO(3) matrix, orthogonal to float64 rather than to float32.

    ``o3.rand_matrix`` builds in the default dtype, so casting its result to
    double leaves an orthogonality defect around 1e-7 -- large enough to swamp
    the exact invariance this module claims.  A float64 QR does not.
    """

    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    rotation, upper = torch.linalg.qr(matrix)
    # Fix the QR sign convention, then make it a proper rotation.
    rotation = rotation * torch.sign(torch.diagonal(upper))
    if float(torch.linalg.det(rotation)) < 0.0:
        rotation[:, 0] = -rotation[:, 0]
    torch.testing.assert_close(
        rotation @ rotation.T, torch.eye(3, dtype=torch.float64),
        atol=1.0e-14, rtol=1.0e-14,
    )
    return rotation


def model():
    torch.manual_seed(17)
    return MambaACEV2(
        r_max=4.5, l_max=2, num_radial=4, hidden_dim=8, num_layers=1,
        correlation_order=4, correlation_channels=4, mamba_dim=12,
        mamba_d_state=4, mamba_backend="torch", readout_hidden=8,
    ).double().eval()


class ParticipationRatioTests(unittest.TestCase):
    def test_flat_spectrum_counts_every_direction(self):
        for n in (1, 3, 8):
            self.assertAlmostEqual(
                participation_ratio(torch.ones(n, dtype=torch.float64)),
                float(n), places=12,
            )

    def test_one_direction_carrying_everything_gives_one(self):
        values = torch.tensor([5.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        self.assertAlmostEqual(participation_ratio(values), 1.0, places=12)

    def test_scale_invariance_and_the_empty_spectrum(self):
        values = torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64)
        self.assertAlmostEqual(
            participation_ratio(values), participation_ratio(1.0e6 * values), places=10
        )
        self.assertEqual(participation_ratio(torch.zeros(4, dtype=torch.float64)), 0.0)

    def test_ratio_is_bounded_by_the_number_of_directions(self):
        generator = torch.Generator().manual_seed(4)
        for _ in range(20):
            values = torch.rand(7, generator=generator, dtype=torch.float64)
            ratio = participation_ratio(values)
            self.assertGreaterEqual(ratio, 1.0)
            self.assertLessEqual(ratio, 7.0 + 1.0e-12)


class GramAccumulatorTests(unittest.TestCase):
    def test_gram_singular_values_reproduce_the_direct_svd(self):
        torch.manual_seed(11)
        irreps = o3.Irreps("3x0e + 2x1o")
        features = torch.randn(50, irreps.dim, dtype=torch.float64)
        accumulator = IrrepGramAccumulator(irreps).update(features)
        # The l = 1 block, columns 3 .. 9, against a direct SVD of the same slice.
        expected = torch.linalg.svdvals(features[:, 3:9])
        self.assertAlmostEqual(
            accumulator.report()[1]["r_eff_full"],
            participation_ratio(expected), places=9,
        )

    def test_batched_updates_match_a_single_pass(self):
        torch.manual_seed(12)
        irreps = o3.Irreps("2x0e + 1x2e")
        features = torch.randn(60, irreps.dim, dtype=torch.float64)
        one_pass = IrrepGramAccumulator(irreps).update(features).report()
        batched = IrrepGramAccumulator(irreps)
        for chunk in features.split(7):
            batched.update(chunk)
        for left, right in zip(one_pass, batched.report()):
            self.assertEqual(left["rows"], right["rows"])
            self.assertAlmostEqual(left["r_eff_full"], right["r_eff_full"], places=10)
            self.assertAlmostEqual(
                left["r_eff_channel"], right["r_eff_channel"], places=10
            )

    def test_leading_axes_are_flattened_into_the_row_index(self):
        torch.manual_seed(13)
        irreps = o3.Irreps("2x1o")
        tokens = torch.randn(5, 7, irreps.dim, dtype=torch.float64)
        self.assertEqual(
            IrrepGramAccumulator(irreps).update(tokens).report()[0]["rows"], 35
        )

    def test_mismatched_width_is_rejected(self):
        accumulator = IrrepGramAccumulator(o3.Irreps("2x0e"))
        with self.assertRaisesRegex(ValueError, "last dimension"):
            accumulator.update(torch.zeros(3, 5))


class SchurRelationTests(unittest.TestCase):
    """``r_eff_full = (2l+1) * r_eff_channel`` for an isotropic distribution.

    If the feature distribution is O(3) invariant then Schur's lemma forces
    ``E[x_{clm} x_{c'lm'}] = delta_{mm'} M_{cc'}``, so the full Gram is
    ``M kron I_{2l+1}``, its spectrum is that of ``M`` repeated ``2l+1`` times,
    and the two entropies differ by exactly ``log(2l+1)``.
    """

    def test_isotropic_features_satisfy_the_schur_relation(self):
        generator = torch.Generator().manual_seed(21)
        for l in (1, 2):
            dimension = 2 * l + 1
            channels = 5
            irreps = o3.Irreps([(channels, (l, (-1) ** l))])
            # x = A g with g standard normal on (channels, 2l+1) is isotropic in
            # m by construction, and correlated across channels through A.
            mixing = torch.randn(
                (channels, channels), generator=generator, dtype=torch.float64
            )
            samples = torch.randn(
                (40000, channels, dimension), generator=generator, dtype=torch.float64
            )
            features = torch.einsum("cd,ndm->ncm", mixing, samples)
            record = (
                IrrepGramAccumulator(irreps)
                .update(features.reshape(-1, irreps.dim))
                .report()[0]
            )
            self.assertAlmostEqual(
                record["r_eff_full"] / record["r_eff_channel"],
                float(dimension), delta=0.02,
                msg=f"Schur relation violated for l={l}",
            )
            self.assertAlmostEqual(
                record["r_eff_full"], record["r_eff_full_predicted"],
                delta=0.02 * dimension,
            )

    def test_an_anisotropic_distribution_departs_from_the_relation(self):
        """The identity has content: it fails when orientations are not covered."""

        generator = torch.Generator().manual_seed(22)
        irreps = o3.Irreps("4x1o")
        samples = torch.randn((4000, 4, 3), generator=generator, dtype=torch.float64)
        # Crush two of the three magnetic components: no rotational coverage.
        samples[:, :, 1] *= 0.01
        samples[:, :, 2] *= 0.01
        record = (
            IrrepGramAccumulator(irreps)
            .update(samples.reshape(-1, irreps.dim))
            .report()[0]
        )
        self.assertLess(record["r_eff_full"], 0.8 * record["r_eff_full_predicted"])


class TokenEffectiveRankTests(unittest.TestCase):
    def test_effective_rank_is_exactly_rotation_invariant(self):
        """``X -> X (I_c kron D^(l)(R))^T`` is orthogonal, so sigma is untouched."""

        network = model()
        rotation = random_rotation(31)
        reference = token_effective_rank(network, [data(POSITIONS)])
        rotated = token_effective_rank(network, [data(POSITIONS @ rotation.T)])
        self.assertEqual(len(reference), len(rotated))
        for left, right in zip(reference, rotated):
            self.assertEqual(left["irrep"], right["irrep"])
            self.assertAlmostEqual(
                left["r_eff_full"], right["r_eff_full"], places=8,
                msg=f"r_eff_full not invariant for {left['irrep']}",
            )
            self.assertAlmostEqual(
                left["r_eff_channel"], right["r_eff_channel"], places=8,
                msg=f"r_eff_channel not invariant for {left['irrep']}",
            )

    def test_effective_rank_is_invariant_under_translation_and_inversion(self):
        network = model()
        reference = token_effective_rank(network, [data(POSITIONS)])
        shift = torch.tensor([2.3, -1.1, 0.4], dtype=torch.float64)
        for positions in (POSITIONS + shift, -POSITIONS):
            for left, right in zip(
                reference, token_effective_rank(network, [data(positions)])
            ):
                self.assertAlmostEqual(
                    left["r_eff_channel"], right["r_eff_channel"], places=8
                )

    def test_report_is_well_formed_and_bounded_by_the_channel_count(self):
        network = model()
        records = token_effective_rank(
            network, [data(POSITIONS), data(POSITIONS * 1.05)]
        )
        self.assertEqual(
            [record["irrep"] for record in records],
            [str(irrep) for _, irrep in o3.Irreps(network.ace.irreps_correlation)],
        )
        self.assertEqual(
            [record["channels"] for record in records],
            [int(mul) for mul, _ in o3.Irreps(network.ace.irreps_correlation)],
        )
        for record in records:
            self.assertGreaterEqual(record["r_eff_channel"], 0.0)
            # r_eff can never exceed the number of directions available.
            self.assertLessEqual(
                record["r_eff_channel"], record["channels"] + 1.0e-9
            )
            self.assertLessEqual(record["r_eff_full"], record["columns"] + 1.0e-9)
            self.assertGreater(record["channel_utilisation"], 0.0)
            self.assertLessEqual(record["channel_utilisation"], 1.0 + 1.0e-9)
            # Rows are atoms x shells, accumulated over both supplied frames.
            self.assertEqual(
                record["rows"], 2 * POSITIONS.shape[0] * network.ace.sequence_length
            )

    def test_accumulating_more_frames_only_adds_rows(self):
        network = model()
        one = token_effective_rank(network, [data(POSITIONS)])
        two = token_effective_rank(network, [data(POSITIONS), data(POSITIONS * 1.05)])
        self.assertEqual(two[0]["rows"], 2 * one[0]["rows"])

    def test_table_renders_every_block(self):
        network = model()
        table = format_effective_rank(token_effective_rank(network, [data(POSITIONS)]))
        lines = table.splitlines()
        self.assertEqual(len(lines), 2 + len(o3.Irreps(network.ace.irreps_correlation)))
        self.assertIn("r_eff_ch", lines[0])


if __name__ == "__main__":
    unittest.main()
