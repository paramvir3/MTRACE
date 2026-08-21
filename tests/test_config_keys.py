"""The config whitelist must cover the constructors it feeds.

``train.py`` builds a model configuration by filtering the YAML against
``MODEL_KEYS``:

    requested_model_config = {
        key: config[key] for key in MODEL_KEYS if key in config and key not in excluded
    }

A key absent from ``MODEL_KEYS`` is therefore **silently dropped**.  The run
trains a different model from the one the file describes and reports nothing,
which is the worst available failure mode: no exception, no warning, and a
plausible-looking result.

These tests pin the whitelist against the constructor signatures so a new model
setting cannot be added without also being made reachable from a config file.
This is not hypothetical -- the ten routing and schedule settings were added to
``MambaACEV2`` and omitted here, leaving every one of them inert in training.
"""

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train  # noqa: E402
from mtace.model import CanonicalMambaACE, MambaACEV2  # noqa: E402


def constructor_parameters(cls):
    return {
        name
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name != "self"
        and parameter.kind
        not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    }


class ModelKeyCoverageTests(unittest.TestCase):
    def test_every_v2_setting_is_reachable_from_a_config_file(self):
        missing = sorted(constructor_parameters(MambaACEV2) - train.MODEL_KEYS)
        self.assertEqual(
            missing, [],
            "MambaACEV2 settings missing from train.MODEL_KEYS; a config file "
            "setting any of these would be silently ignored",
        )

    def test_every_canonical_setting_is_reachable_from_a_config_file(self):
        missing = sorted(constructor_parameters(CanonicalMambaACE) - train.MODEL_KEYS)
        self.assertEqual(missing, [], "CanonicalMambaACE settings missing from MODEL_KEYS")

    def test_the_whitelist_contains_no_settings_that_exist_nowhere(self):
        """A key no constructor accepts would raise TypeError at construction."""

        known = constructor_parameters(MambaACEV2) | constructor_parameters(
            CanonicalMambaACE
        )
        self.assertEqual(sorted(train.MODEL_KEYS - known), [])

    def test_the_new_routing_and_schedule_settings_are_present(self):
        for key in (
            "mixer_schedule",
            "num_experts",
            "expert_hidden",
            "expert_latent_dim",
            "router_tau",
            "router_switch",
            "router_threshold_init",
            "router_balance_rate",
            "router_balance_target",
            "routing_backend",
        ):
            self.assertIn(key, train.MODEL_KEYS)


class CanonicalExclusionTests(unittest.TestCase):
    """Keys the canonical architecture cannot accept must be excluded there.

    ``train.py`` filters with ``key not in excluded`` for
    ``architecture: mtace_canonical``.  A v2-only key that is in MODEL_KEYS but
    not excluded would be forwarded to ``CanonicalMambaACE`` and raise
    ``TypeError`` at construction.
    """

    @staticmethod
    def canonical_excluded():
        source = Path(train.__file__).read_text()
        block = source.split('model_class = CanonicalMambaACE', 1)[1]
        block = block.split("excluded = {", 1)[1].split("}", 1)[0]
        return {
            line.strip().strip(",").strip('"')
            for line in block.splitlines()
            if line.strip().startswith('"')
        }

    # Pre-existing gap, deliberately pinned rather than closed.  These v2-only
    # keys are in MODEL_KEYS but not excluded for the canonical architecture, so
    # a canonical config that sets one raises TypeError naming the argument.
    # That is a *loud* failure, and adding them to ``excluded`` would convert it
    # into a silent one -- the config would be accepted and the setting ignored.
    # Which is preferable is a call for the author, so this test freezes the set
    # instead of changing behaviour: it fails if the gap grows.
    KNOWN_UNEXCLUDED = {
        "coupling_channels", "coupling_mode", "decay_mode", "invariant_norm",
        "invariant_norm_eps", "invariant_overlap_width", "screening_min_angstrom",
        "shell_degree", "shell_pair_channels", "shell_pair_mode",
        "shell_pair_state_clip", "shell_pair_width", "shell_scales",
    }

    def test_routing_and_schedule_settings_are_excluded_for_canonical(self):
        """The keys added with the hybrid work must not reach the canonical class."""

        excluded = self.canonical_excluded()
        for key in (
            "mixer_schedule", "num_experts", "expert_hidden", "expert_latent_dim",
            "router_tau", "router_switch", "router_threshold_init",
            "router_balance_rate", "router_balance_target", "routing_backend",
        ):
            self.assertIn(key, excluded, f"{key} would be forwarded to CanonicalMambaACE")

    def test_the_unexcluded_v2_only_set_has_not_grown(self):
        excluded = self.canonical_excluded()
        v2_only = constructor_parameters(MambaACEV2) - constructor_parameters(
            CanonicalMambaACE
        )
        forwarded = (v2_only & train.MODEL_KEYS) - excluded
        self.assertEqual(
            forwarded, self.KNOWN_UNEXCLUDED,
            "the set of v2-only keys that reach CanonicalMambaACE has changed; "
            "a new one raises TypeError for canonical configs",
        )

    def test_the_exclusion_list_is_itself_well_formed(self):
        excluded = self.canonical_excluded()
        # Nothing in the exclusion list should be a canonical-only setting.
        canonical_only = constructor_parameters(CanonicalMambaACE) - constructor_parameters(
            MambaACEV2
        )
        self.assertEqual(sorted(excluded & canonical_only), [])


if __name__ == "__main__":
    unittest.main()
