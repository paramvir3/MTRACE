"""Per-layer mixer schedules: a Mamba stack with sparse attention anchors."""

import tempfile
import unittest
from pathlib import Path

import torch

from mtace.checkpoint import restore_model, save_checkpoint
from mtace.model import MambaACEV2
from mtace.schedule import anchored_schedule, resolve_mixer_schedule


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
        "volume": torch.tensor(729.0, dtype=positions.dtype),
    }


POSITIONS = torch.tensor(
    [[0.1, 0.2, 0.3], [1.2, 0.4, 0.7], [0.5, 1.3, 0.8], [0.8, 0.7, 1.7]],
    dtype=torch.float64,
)

SETTINGS = dict(
    r_max=4.5,
    l_max=2,
    num_radial=4,
    hidden_dim=8,
    num_layers=3,
    correlation_order=4,
    correlation_channels=4,
    mamba_dim=12,
    mamba_d_state=4,
    mamba_backend="torch",
    readout_hidden=8,
)


def hybrid_model(seed=17, **overrides):
    torch.manual_seed(seed)
    settings = dict(SETTINGS)
    settings.update(overrides)
    return MambaACEV2(**settings).double().eval()


class ScheduleResolutionTests(unittest.TestCase):
    def test_none_broadcasts_the_scalar_shorthand(self):
        self.assertEqual(
            resolve_mixer_schedule("attention", None, 3),
            ("attention", "attention", "attention"),
        )

    def test_short_schedules_cycle_and_exact_ones_are_used_as_written(self):
        self.assertEqual(
            resolve_mixer_schedule("mamba", ["mamba", "mamba", "attention"], 6),
            ("mamba", "mamba", "attention", "mamba", "mamba", "attention"),
        )
        self.assertEqual(
            resolve_mixer_schedule("mamba", ["attention", "mamba"], 2),
            ("attention", "mamba"),
        )
        # A bare string is the one-entry schedule, not a character sequence.
        self.assertEqual(
            resolve_mixer_schedule("mamba", "attention", 2),
            ("attention", "attention"),
        )

    def test_invalid_schedules_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown mixer"):
            resolve_mixer_schedule("mamba", ["mamba", "transformer"], 2)
        with self.assertRaisesRegex(ValueError, "never longer"):
            resolve_mixer_schedule("mamba", ["mamba"] * 4, 3)
        with self.assertRaisesRegex(ValueError, "at least one"):
            resolve_mixer_schedule("mamba", [], 3)
        with self.assertRaises(ValueError):
            resolve_mixer_schedule("mamba", None, 0)

    def test_anchored_schedule_matches_the_published_hybrid(self):
        # 52 layers, 4 attention: the Nemotron 3 Super ratio, 7.7% attention.
        schedule = anchored_schedule(52, 4)
        positions = [i for i, name in enumerate(schedule) if name == "attention"]
        self.assertEqual(positions, [6, 19, 32, 45])
        self.assertEqual(len(schedule), 52)
        self.assertEqual(schedule.count("attention"), 4)

    def test_anchored_schedule_never_loses_or_doubles_an_anchor(self):
        for layers in range(1, 40):
            for anchors in range(0, layers + 1):
                schedule = anchored_schedule(layers, anchors)
                self.assertEqual(len(schedule), layers)
                self.assertEqual(
                    schedule.count("attention"), anchors,
                    f"{anchors} anchors in {layers} layers collided",
                )
        with self.assertRaises(ValueError):
            anchored_schedule(4, 5)


class HybridStackTests(unittest.TestCase):
    def test_layers_receive_their_scheduled_mixer(self):
        network = hybrid_model(mixer_schedule=["mamba", "attention"])
        self.assertEqual(
            [layer.mixer_type for layer in network.layers],
            ["mamba", "attention", "mamba"],
        )
        self.assertEqual(network.mixer_schedule, ("mamba", "attention", "mamba"))
        # The attention layer really is attention, not a renamed Mamba block.
        self.assertTrue(hasattr(network.layers[1].mixer, "qkv_projection"))
        self.assertFalse(hasattr(network.layers[0].mixer, "qkv_projection"))

    def test_scheduled_layers_share_one_scalar_residual_block(self):
        """A hybrid must differ in the mixer and *only* in the mixer.

        ``ffn_type=None`` used to be resolved inside each block from that block's
        own mixer, which would hand attention layers a SwiGLU residual and
        mamba1 layers a plain MLP.  The stack would then differ in two ways at
        once and no comparison across it would be controlled.

        The uniform choice for a mixed stack is SwiGLU: attention layers have
        always had it, and it is what the default mamba3 variant resolves to for
        every mixer, so nothing that previously existed changes shape.  (An
        earlier revision of this test asserted ``mlp`` here, which was an
        artifact of resolving ``ffn_type`` from the ``mixer_type`` shorthand
        rather than from the schedule.)
        """

        network = hybrid_model(
            mixer_schedule=["mamba", "attention"],
            mamba_variant="mamba1",
            mamba_d_conv=3,
        )
        self.assertEqual({layer.ffn_type for layer in network.layers}, {"swiglu"})
        self.assertEqual({layer.use_swiglu for layer in network.layers}, {True})

    def test_broadcast_schedule_reproduces_the_scalar_setting_exactly(self):
        scalar = hybrid_model(mixer_type="attention")
        broadcast = hybrid_model(mixer_schedule=["attention"])
        explicit = hybrid_model(mixer_schedule=["attention"] * 3)
        structure = data(POSITIONS)
        reference = scalar(structure, compute_stress=False)[0]
        for network in (broadcast, explicit):
            torch.testing.assert_close(
                network(structure, compute_stress=False)[0], reference,
                atol=0.0, rtol=0.0,
            )

    def test_the_two_spellings_agree_for_every_mixer_and_variant(self):
        """Regression: ``ffn_type`` was resolved from the wrong thing.

        ``mixer_type="attention"`` and ``mixer_schedule=["attention"]`` describe
        the same stack, but resolving ``ffn_type`` from the ``mixer_type``
        shorthand read its *default* ("mamba") when only the schedule was given.
        With ``mamba_variant="mamba1"`` that produced an MLP residual block on
        one spelling and SwiGLU on the other.  The earlier test above missed it
        because the default variant is mamba3, where both paths give SwiGLU.
        """

        for variant in ("mamba1", "mamba3"):
            for mixer in ("mamba", "attention", "identity"):
                extra = {"mamba_variant": variant}
                if variant == "mamba1":
                    extra["mamba_d_conv"] = 3
                scalar = hybrid_model(mixer_type=mixer, **extra)
                schedule = hybrid_model(mixer_schedule=[mixer], **extra)
                self.assertEqual(
                    [layer.ffn_type for layer in scalar.layers],
                    [layer.ffn_type for layer in schedule.layers],
                    f"ffn_type differs between spellings for {variant}/{mixer}",
                )
                torch.testing.assert_close(
                    schedule(data(POSITIONS), compute_stress=False)[0],
                    scalar(data(POSITIONS), compute_stress=False)[0],
                    atol=0.0, rtol=0.0,
                )

    def test_a_heterogeneous_stack_keeps_one_residual_block_under_mamba1(self):
        network = hybrid_model(
            mixer_schedule=["mamba", "attention"],
            mamba_variant="mamba1",
            mamba_d_conv=3,
        )
        self.assertEqual({layer.ffn_type for layer in network.layers}, {"swiglu"})

    def test_hybrid_stack_preserves_symmetry_and_force_covariance(self):
        network = hybrid_model(mixer_schedule=["mamba", "attention"])
        energy, forces, _, _ = network(data(POSITIONS), compute_stress=False)
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )
        rotated = network(data(POSITIONS @ rotation.T), compute_stress=False)
        torch.testing.assert_close(rotated[0], energy, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(
            rotated[1], forces @ rotation.T, atol=1.0e-10, rtol=1.0e-10
        )
        inverted = network(data(-POSITIONS), compute_stress=False)
        torch.testing.assert_close(inverted[0], energy, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(inverted[1], -forces, atol=1.0e-10, rtol=1.0e-10)
        translated = network(
            data(POSITIONS + torch.tensor([2.3, -1.1, 0.4], dtype=torch.float64)),
            compute_stress=False,
        )
        torch.testing.assert_close(translated[0], energy, atol=1.0e-10, rtol=1.0e-10)
        torch.testing.assert_close(translated[1], forces, atol=1.0e-10, rtol=1.0e-10)

    def test_hybrid_stack_conserves_energy(self):
        network = hybrid_model(mixer_schedule=["mamba", "attention"])
        _, forces, _, _ = network(data(POSITIONS), compute_stress=False)
        step = 1.0e-6
        for atom in range(POSITIONS.shape[0]):
            for axis in range(3):
                plus = POSITIONS.clone()
                minus = POSITIONS.clone()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                derivative = (
                    network(data(plus), compute_stress=False)[0]
                    - network(data(minus), compute_stress=False)[0]
                ) / (2.0 * step)
                self.assertAlmostEqual(
                    float(-derivative), float(forces[atom, axis]), places=6
                )

    def test_atom_permutation_leaves_a_hybrid_energy_unchanged(self):
        network = hybrid_model(mixer_schedule=["mamba", "attention"])
        species = [8, 1, 6, 14]
        energy = network(data(POSITIONS, species), compute_stress=False)[0]
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = [species[index] for index in permutation.tolist()]
        torch.testing.assert_close(
            network(data(POSITIONS[permutation], permuted), compute_stress=False)[0],
            energy, atol=1.0e-10, rtol=1.0e-10,
        )

    def test_hybrid_and_pure_stacks_are_genuinely_different_models(self):
        structure = data(POSITIONS)
        pure = hybrid_model(mixer_type="mamba")(structure, compute_stress=False)[0]
        hybrid = hybrid_model(mixer_schedule=["mamba", "attention"])(
            structure, compute_stress=False
        )[0]
        self.assertGreater(float((pure - hybrid).abs()), 1.0e-9)


class HybridCheckpointTests(unittest.TestCase):
    def test_schedule_survives_a_checkpoint_round_trip(self):
        config = dict(SETTINGS)
        config["mixer_schedule"] = ["mamba", "attention"]
        config["num_experts"] = 2
        config["router_tau"] = 0.5
        torch.manual_seed(17)
        network = MambaACEV2(**config).double().eval()
        self.assertEqual(network.architecture_version, 11)
        reference = network(data(POSITIONS), compute_stress=False)[0]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.pt"
            save_checkpoint(path, network, config, atomic_numbers=[1, 8])
            restored, checkpoint = restore_model(path)

        self.assertEqual(int(checkpoint["architecture_version"]), 11)
        self.assertEqual(
            [layer.mixer_type for layer in restored.layers],
            ["mamba", "attention", "mamba"],
        )
        self.assertEqual(
            [layer.routed_ffn is not None for layer in restored.layers], [True] * 3
        )
        torch.testing.assert_close(
            restored(data(POSITIONS), compute_stress=False)[0], reference,
            atol=0.0, rtol=0.0,
        )

    def test_a_v10_configuration_still_loads_without_migration(self):
        """v11 is additive: neither new setting appears in a v10 config."""

        config = dict(SETTINGS)
        torch.manual_seed(17)
        network = MambaACEV2(**config).double().eval()
        reference = network(data(POSITIONS), compute_stress=False)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            save_checkpoint(path, network, config, atomic_numbers=[1, 8])
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["architecture_version"] = 10
            torch.save(payload, path)
            restored, _ = restore_model(path)
        self.assertEqual(restored.mixer_schedule, ("mamba",) * 3)
        self.assertIsNone(restored.layers[0].routed_ffn)
        torch.testing.assert_close(
            restored(data(POSITIONS), compute_stress=False)[0], reference,
            atol=0.0, rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
