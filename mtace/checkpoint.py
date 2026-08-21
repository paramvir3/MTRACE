"""Checkpoint helpers with explicit architecture metadata."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch


_DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}
_NAME_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}


def _floating_dtype(tensors) -> torch.dtype:
    dtypes = {tensor.dtype for tensor in tensors if tensor.is_floating_point()}
    if not dtypes:
        return torch.get_default_dtype()
    if len(dtypes) != 1:
        names = ", ".join(sorted(str(dtype) for dtype in dtypes))
        raise ValueError(f"MTACE checkpoints require one floating dtype, got {names}")
    dtype = dtypes.pop()
    if dtype not in _DTYPE_NAMES:
        raise ValueError(f"Unsupported MTACE checkpoint dtype {dtype}")
    return dtype


def _normalize_atomic_numbers(values: Sequence[int] | None) -> list[int]:
    numbers = sorted({int(value) for value in (values or [])})
    invalid = [number for number in numbers if not 1 <= number <= 118]
    if invalid:
        raise ValueError(f"Atomic numbers must satisfy 1 <= Z <= 118, got {invalid}")
    return numbers


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_config: dict,
    training_config: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    atomic_energies: dict[int, float] | None = None,
    atomic_numbers: Sequence[int] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    metrics: dict[str, float] | None = None,
    training_metrics: dict[str, float] | None = None,
    best_validation_loss: float | None = None,
    checkpoint_role: str | None = None,
    stale_epochs: int = 0,
    rng_state: dict | None = None,
    run_signature: dict | None = None,
    ema_state: dict | None = None,
) -> None:
    references = {int(k): float(v) for k, v in (atomic_energies or {}).items()}
    invalid_references = sorted(key for key in references if not 1 <= key <= 118)
    if invalid_references:
        raise ValueError(f"Invalid atomic-reference keys Z={invalid_references}")
    if any(not math.isfinite(value) for value in references.values()):
        raise ValueError("Atomic reference energies must be finite")
    species = _normalize_atomic_numbers(atomic_numbers)
    if not species and references:
        species = sorted(references)
    if references:
        missing = sorted(set(species) - set(references))
        if missing:
            raise ValueError(f"Missing atomic reference energies for Z={missing}")
    model_dtype = _floating_dtype(model.state_dict().values())
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validation_metrics = {
        str(name): float(value) for name, value in (metrics or {}).items()
    }
    epoch_training_metrics = {
        str(name): float(value) for name, value in (training_metrics or {}).items()
    }
    payload = {
        "format_version": 1,
        "training_objective_version": 2,
        "architecture": getattr(model, "architecture", "mtace_v2"),
        "architecture_version": int(getattr(model, "architecture_version", 2)),
        "model_config": dict(model_config),
        "training_config": dict(training_config or {}),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "training_metrics": epoch_training_metrics,
        "validation_metrics": validation_metrics,
        "best_validation_loss": (
            None if best_validation_loss is None else float(best_validation_loss)
        ),
        "checkpoint_role": checkpoint_role,
        "stale_epochs": int(stale_epochs),
        "rng_state": dict(rng_state or {}),
        "ema_state_dict": ema_state,
        "run_signature": dict(run_signature or {}),
        "atomic_energies": references,
        "atomic_numbers": species,
        "model_dtype": _DTYPE_NAMES[model_dtype],
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if checkpoint.get("architecture") not in {
        "mtace",
        "mtace_v2",
        "mtace_canonical",
    }:
        raise ValueError("Checkpoint is not a MTACE checkpoint")
    if int(checkpoint.get("format_version", 0)) != 1:
        raise ValueError("Unsupported MTACE checkpoint format")
    return checkpoint


def migrated_model_config(
    checkpoint: dict,
    *,
    mamba_backend: str | None = None,
) -> dict:
    """Return the constructor configuration for any supported checkpoint."""

    architecture = checkpoint.get("architecture", "mtace")
    model_config = dict(checkpoint["model_config"])
    architecture_version = int(checkpoint.get("architecture_version", 1))
    # Note: there is deliberately no "< 11" migration block either.  Both v11
    # features are inert at their defaults -- ``mixer_schedule=None`` broadcasts
    # the stored scalar ``mixer_type`` to every layer, which is what v10 did, and
    # ``num_experts=0`` builds no router and no experts, so the block's state dict
    # keys are unchanged.  A v10 checkpoint therefore restores bit-for-bit.  A
    # checkpoint that *was* trained with a schedule or with experts carries both
    # settings in its own ``model_config`` and needs no migration for the same
    # reason.
    # Note: there is deliberately no "< 10" migration block.  Every
    # architecture-v10 setting defaults to its v9 behaviour -- shell_degree=3,
    # invariant_norm="squared", shell_pair_channels=0, coupling_mode="gate" --
    # so a v9 checkpoint reproduces its trained energy with no migration at all.
    mamba3_version = 4 if architecture == "mtace_v2" else 2
    official_angle_version = 5 if architecture == "mtace_v2" else 3
    if architecture_version < mamba3_version:
        model_config.setdefault("mamba_variant", "mamba1")
    else:
        model_config.setdefault("mamba_variant", "mamba3")
        if architecture_version < official_angle_version:
            model_config.setdefault("mamba_angle_mode", "legacy_bounded")
    if architecture == "mtace_v2" and architecture_version < 3:
        model_config.setdefault("mamba_bidirectional_tied", True)
    if architecture == "mtace_v2" and architecture_version < 6:
        model_config.setdefault("tokenizer_type", "legacy_basis")
        model_config.setdefault("num_shells", int(model_config["num_radial"]))
        model_config.setdefault("mixer_type", "mamba")
        model_config.setdefault("mamba_mimo_rank", 1)
    if architecture == "mtace_v2" and architecture_version < 7:
        # Version 6 passed the common block dropout into attention weights.
        # Preserve that training-time behavior when resuming historical runs.
        model_config.setdefault(
            "attention_dropout", float(model_config.get("dropout", 0.0))
        )
    if architecture == "mtace_v2" and architecture_version < 8:
        # Versions 6 and 7 applied a second cutoff to physical shell weights and
        # divided the integrated shell update by sqrt(L). Older frequency-token
        # models used the same reduction. Preserve every historical energy and
        # derivative exactly when those checkpoints are restored.
        model_config.setdefault("shell_coupling_mode", "legacy")
    if architecture == "mtace_v2" and architecture_version < 9:
        # Architecture v9 changes four defaults.  Every one of them is pinned to
        # its historical value here so a v8-or-earlier checkpoint reproduces its
        # trained energy, force, and stress bit-for-bit.
        #
        #  * ``avg_num_neighbors``: v8 had no density normalization.
        #  * ``shell_boundary_mode``: v8 dropped out-of-range spline weights and
        #    renormalized; v9 folds them onto the boundary shell instead.
        #  * ``continuum_mode``: v8 had no radial-metric quadrature.  (It is a
        #    no-op at the reference resolution, but pinning it keeps the stored
        #    configuration self-describing.)
        #  * ``mamba_rotary_layout``: v8 SISO used adjacent pairs and v8 MIMO used
        #    blockwise halves.  v9 unifies both on halves, so the historical
        #    layout depends on the stored rank.
        model_config.setdefault("avg_num_neighbors", 1.0)
        model_config.setdefault("shell_r_min", 0.0)
        model_config.setdefault("shell_boundary_mode", "renormalize")
        model_config.setdefault("continuum_mode", False)
        model_config.setdefault("invariant_pair_channels", 0)
        # v8 always used the single-pass Hillis-Steele scan.  Pin it so a restored
        # checkpoint reproduces its trained energy bitwise, not merely to
        # floating-point association order.
        model_config.setdefault("mamba_scan_mode", "parallel")
        historical_rank = int(model_config.get("mamba_mimo_rank", 1))
        model_config.setdefault(
            "mamba_rotary_layout", "halves" if historical_rank > 1 else "pairs"
        )
        # ``ffn_type=None`` reproduces the historical coupling of the scalar
        # residual block to the mixer, so it needs no explicit migration.  New
        # controlled comparisons should set it explicitly; see docs/BENCHMARKS.md.
        historical_variant = str(model_config.get("mamba_variant", "mamba3")).lower()
        # v8 built a Mamba-3 mixer while still accepting mamba_d_conv; v9 rejects
        # that combination, so drop the inert setting rather than fail the load.
        if historical_variant == "mamba3":
            model_config.pop("mamba_d_conv", None)
    if mamba_backend is not None:
        model_config["mamba_backend"] = mamba_backend
    return model_config


def restore_model(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    mamba_backend: str | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Reconstruct a model, including migrations for historical checkpoints."""

    from .model import CanonicalMambaACE, MambaACEV2

    device = torch.device(device)
    checkpoint = load_checkpoint(path, map_location=device)
    architecture = checkpoint.get("architecture", "mtace")
    model_class = MambaACEV2 if architecture == "mtace_v2" else CanonicalMambaACE
    model_config = migrated_model_config(
        checkpoint, mamba_backend=mamba_backend
    )
    state_dict = dict(checkpoint["model_state_dict"])
    architecture_version = int(checkpoint.get("architecture_version", 1))
    if architecture == "mtace_v2" and architecture_version < 3:
        embedding_key = "species_embedding.weight"
        if embedding_key in state_dict and state_dict[embedding_key].shape[0] == 118:
            old_embedding = state_dict[embedding_key]
            state_dict[embedding_key] = torch.cat(
                (
                    old_embedding,
                    old_embedding.new_zeros((1, old_embedding.shape[1])),
                ),
                dim=0,
            )
    dtype_name = checkpoint.get("model_dtype")
    if dtype_name is None:
        dtype = _floating_dtype(state_dict.values())
    else:
        try:
            dtype = _NAME_DTYPES[str(dtype_name)]
        except KeyError as exception:
            raise ValueError(f"Unsupported checkpoint model_dtype {dtype_name!r}") from exception
    model = model_class(**model_config).to(device=device, dtype=dtype)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint
