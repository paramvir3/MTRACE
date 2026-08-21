"""Public MTACE API."""

from .calculator import MambaACECalculator
from .deployment import export_lammps_model
from .mamba3 import (
    Mamba3SequenceMixer,
    mamba3_mimo_scan_parallel,
    mamba3_mimo_scan_reference,
    mamba3_scan_parallel,
    mamba3_scan_reference,
)
from .mixers import (
    AttentionSequenceMixer,
    DeepSetsSequenceMixer,
    DenseRadialSequenceMixer,
    IdentitySequenceMixer,
)
from .diagnostics import (
    IrrepGramAccumulator,
    format_effective_rank,
    participation_ratio,
    token_effective_rank,
)
from .model import CanonicalMambaACE, MambaACE, MambaACEV2
from .optim import MuonWithAuxAdamW, build_optimizer, get_muon_param_groups
from .physics import CompactRadialShellBasis
from .routing import (
    CompactSupportRouter,
    RoutedScalarFFN,
    resolve_switch_contract,
    routing_capacity,
    switch_polynomial,
)
from .schedule import (
    MIXER_NAMES,
    MIXERS_PER_ANCHOR,
    anchor_count_for,
    anchored_schedule,
    nemotron_style_schedule,
    resolve_mixer_schedule,
)
from .ssm import MambaSequenceMixer, selective_scan_parallel, selective_scan_reference

__all__ = [
    "MambaACE",
    "MambaACEV2",
    "CanonicalMambaACE",
    "MIXER_NAMES",
    "MIXERS_PER_ANCHOR",
    "anchor_count_for",
    "anchored_schedule",
    "nemotron_style_schedule",
    "resolve_mixer_schedule",
    "CompactSupportRouter",
    "RoutedScalarFFN",
    "switch_polynomial",
    "resolve_switch_contract",
    "routing_capacity",
    "IrrepGramAccumulator",
    "participation_ratio",
    "token_effective_rank",
    "format_effective_rank",
    "MambaACECalculator",
    "export_lammps_model",
    "MuonWithAuxAdamW",
    "build_optimizer",
    "get_muon_param_groups",
    "MambaSequenceMixer",
    "Mamba3SequenceMixer",
    "AttentionSequenceMixer",
    "DenseRadialSequenceMixer",
    "DeepSetsSequenceMixer",
    "IdentitySequenceMixer",
    "CompactRadialShellBasis",
    "mamba3_scan_parallel",
    "mamba3_scan_reference",
    "mamba3_mimo_scan_parallel",
    "mamba3_mimo_scan_reference",
    "selective_scan_parallel",
    "selective_scan_reference",
]
