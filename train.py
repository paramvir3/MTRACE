#!/usr/bin/env python3
"""Train MTACE on an ASE-readable trajectory."""

from __future__ import annotations

import argparse
import hashlib
import os
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from ase.io import read
from torch.utils.data import DataLoader

from mtace.checkpoint import (
    load_checkpoint,
    migrated_model_config,
    restore_model,
    save_checkpoint,
)
from mtace.data import (
    AtomisticDataset,
    collate_structures,
    minimum_edge_distance,
    target_statistics,
    average_num_neighbors,
    parse_atomic_energies,
    shell_occupancy,
    solve_atomic_energies,
    split_frames,
    stress_mse,
    stress_to_mandel,
)
from mtace.model import CanonicalMambaACE, MambaACEV2
from mtace.optim import (
    ExponentialMovingAverage,
    adamw_parameter_groups,
    build_optimizer,
    optimizer_group_summary,
)


MODEL_KEYS = {
    "r_max", "l_max", "num_radial", "hidden_dim", "num_layers",
    "correlation_order", "correlation_channels", "radial_basis_type",
    "radial_trainable", "gaussian_width", "remove_pair_self_contractions",
    "radial_mlp_hidden", "radial_mlp_layers", "avg_num_neighbors",
    "tokenizer_type", "num_shells", "shell_coupling_mode", "shell_degree", "shell_scales", "mixer_type",
    "shell_r_min", "shell_boundary_mode", "continuum_mode",
    "attention_heads",
    "mamba_dim", "mamba_d_state", "mamba_d_conv", "mamba_expand",
    "mamba_bidirectional_tied", "mamba_variant", "mamba_headdim",
    "mamba_rope_fraction", "mamba_a_floor", "mamba_chunk_size",
    "mamba_angle_mode", "mamba_mimo_rank", "mamba_rotary_layout",
    "mamba_scan_mode",
    "mamba_backend", "ffn_hidden", "ffn_type", "invariant_pair_channels", "invariant_norm", "invariant_norm_eps",
    "invariant_overlap_width",
    "shell_pair_channels", "shell_pair_width", "shell_pair_mode",
    "decay_mode", "screening_min_angstrom",
    "shell_pair_state_clip",
    "coupling_mode", "coupling_channels",
    "gate_norm",
    "mixer_schedule",
    "num_experts", "expert_hidden", "expert_latent_dim",
    "router_tau", "router_switch", "router_threshold_init",
    "router_balance_rate", "router_balance_target", "routing_backend",
    "dropout", "attention_dropout",
    "layer_scale_init", "readout_hidden",
}
# Every MambaACEV2 constructor argument must appear above, or a setting placed in
# a config file is silently dropped rather than rejected.  That failure mode is
# invisible -- the run trains a different model than the file asks for and says
# nothing -- so tests/test_config_keys.py pins the invariant against the
# constructor signature.

EV_A3_TO_GPA = 160.21766208

OPTIMIZER_CONTRACT_KEYS = (
    "learning_rate",
    "weight_decay",
    "muon_learning_rate",
    "muon_aux_learning_rate",
    "muon_momentum",
    "muon_ns_steps",
    "muon_newton_schulz_steps",
    "muon_nesterov",
    "muon_ns_coefficients",
    "muon_eps",
    "muon_ns_precision",
    "muon_adjust_lr_fn",
    "muon_aux_betas",
    "muon_aux_eps",
    "muon_parameter_mode",
    "ema_decay",
    "normalize_loss_weights",
    "muon_include",
    "muon_exclude",
)



def _batched_forward_groups(batch, weights, report_stress_metrics,
                            max_batch_atoms=None):
    """Split a batch into runs that can share one forward pass.

    ``compute_stress`` is a per-call flag, so structures that disagree about it
    cannot be merged.  Datasets are usually uniform, in which case this returns
    a single group holding the whole batch; a mixed dataset degrades to two
    groups rather than back to the per-structure loop.

    ``max_batch_atoms`` caps how many atoms enter one forward pass.  Activation
    memory scales with the atom count -- the state-space scan alone holds
    O(N L d d_s) -- so on large systems an unrestricted batch exhausts the
    device: measured on 768-atom liquid water, a Mamba mixer needs 8.6 GiB for a
    single structure and two do not fit on a 16 GiB card.  Chunking keeps the
    gradient identical, because the per-structure losses are summed either way;
    only the number of forward passes changes.  ``None`` means no cap.
    """

    groups = {}
    for raw in batch:
        include_stress = bool(raw["has_stress"]) and (
            report_stress_metrics or weights["stress"] > 0.0
        )
        groups.setdefault(include_stress, []).append(raw)
    out = []
    for flag, raws in groups.items():
        if not max_batch_atoms:
            out.append((raws, flag))
            continue
        chunk, total = [], 0
        for raw in raws:
            size = int(raw["z"].numel())
            if chunk and total + size > int(max_batch_atoms):
                out.append((chunk, flag))
                chunk, total = [], 0
            chunk.append(raw)
            total += size
        if chunk:
            out.append((chunk, flag))
    return out


def _split_batched_outputs(items, merged, energies, forces, stresses):
    """Return per-structure views of a batched forward.

    The views keep the autograd graph, so a loss assembled from them produces
    exactly the gradients a per-structure loop would accumulate.
    """

    counts = [int(item["z"].numel()) for item in items]
    force_parts = torch.split(forces, counts)
    out = []
    for index, item in enumerate(items):
        stress = stresses[index] if stresses.ndim == 3 else stresses
        out.append((item, energies[index], force_parts[index], stress))
    return out




def cuda_determinism_supported() -> bool:
    """Probe whether the scatter kernels this model needs are deterministic.

    The ACE density and the shell tokenizer both use ``index_add_``.  PyTorch
    historically shipped no deterministic CUDA kernel for it, which is why this
    file used to reject ``deterministic_algorithms: true`` on CUDA outright.
    That is no longer true -- measured under torch 2.11.0+cu128, ``index_add_``,
    ``scatter_add_`` and the ``index_add`` backward all run under strict
    determinism and give bitwise identical results.

    Probing beats asserting in either direction: the answer depends on the
    installed PyTorch, so ask the installed PyTorch.
    """

    if not torch.cuda.is_available():
        return False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        index = torch.zeros(8, 2, device="cuda")
        index.index_add_(0, torch.zeros(4, dtype=torch.long, device="cuda"),
                         torch.ones(4, 2, device="cuda"))
        return True
    except RuntimeError:
        return False
    finally:
        torch.use_deterministic_algorithms(previous)


def configure_determinism(enabled: bool, device: torch.device) -> str:
    """Make runs reproducible, which on CUDA they are not by default.

    Scatter and reduction kernels on CUDA accumulate in a nondeterministic
    order, so two runs from the same seed diverge.  Measured on an RTX 5060 Ti
    the immediate spread is 2.8e-16 eV over five evaluations of one model,
    7.5e-15 relative -- negligible in itself, but training compounds it, so
    "same seed, same result" does not hold and a diverging run cannot be
    reproduced for debugging.

    Cost is nil where it is available: 171.1 ms/step with determinism against
    173.9 ms/step without, a factor 0.98.  Strict mode is used rather than
    ``warn_only``, which would quietly permit a nondeterministic kernel back in.

    ``CUBLAS_WORKSPACE_CONFIG`` must be set before cuBLAS initialises, which is
    why this runs before any device work.
    """

    if not enabled:
        torch.use_deterministic_algorithms(False)
        return "off"
    if device.type == "cuda" and not cuda_determinism_supported():
        raise ValueError(
            "deterministic_algorithms: true is not supported by this PyTorch on "
            "CUDA, because the ACE density and shell pooling use index_add_ and "
            "this build provides no deterministic CUDA kernel for it. Use "
            "device: cpu for a bitwise-deterministic reference run, upgrade "
            "PyTorch, or set deterministic_algorithms: false."
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return "on"


def configure_worker_sharing(num_workers: int) -> None:
    """Make ``num_workers > 0`` survive a default file-descriptor limit.

    PyTorch shares worker tensors through the ``file_descriptor`` strategy on
    Linux, spending one descriptor per shared tensor.  With precomputed
    neighbour lists each dataset item carries several tensors and a prefetching
    loader holds many in flight, so a stock ``ulimit -n`` of 1024 is exhausted
    and the run dies with ``OSError: [Errno 24] Too many open files`` part-way
    through an epoch -- late enough to waste the work already done.

    Raising the soft limit to the hard limit fixes that outright and adds no new
    machinery.  The ``file_system`` strategy is the other common remedy and is
    *deliberately not* the default here: it routes sharing through a
    ``torch_shm_manager`` helper process, and on a busy or core-starved host
    that helper fails to answer in time, turning the descriptor error into
    ``RuntimeError: Shared memory manager connection has timed out``.  Measured
    on a contended four-core machine, switching strategy replaced one crash with
    another.  It is kept as a fallback for hosts whose hard limit is too low for
    the descriptor strategy to work at all.
    """

    if num_workers <= 0:
        return
    hard_limit = 0
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        hard_limit = hard
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except (ImportError, ValueError, OSError):
        return
    if hard_limit and hard_limit < 4096:
        # The descriptor strategy cannot be made to fit; accept the shared
        # memory manager and its risks rather than a certain crash.
        try:
            import torch.multiprocessing as multiprocessing

            multiprocessing.set_sharing_strategy("file_system")
        except (ImportError, RuntimeError, ValueError):
            pass


def move(item, device, non_blocking=False):
    return {
        key: value.to(device, non_blocking=non_blocking)
        if torch.is_tensor(value)
        else value
        for key, value in item.items()
    }


def resolve_config_path(value: str | Path, directory: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (directory / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_training_dtype(value) -> torch.dtype:
    name = str(value or "float32").lower()
    choices = {"float32": torch.float32, "float64": torch.float64}
    try:
        return choices[name]
    except KeyError as exception:
        raise ValueError("dtype must be 'float32' or 'float64'") from exception


def finite_nonnegative(name: str, value) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


def conflicting_model_settings(checkpoint: dict, requested: dict) -> list[str]:
    """Return explicitly requested settings that differ from the saved model."""

    saved = migrated_model_config(checkpoint)
    return sorted(
        key for key, value in requested.items() if key not in saved or saved[key] != value
    )


def capture_rng_state(
    data_generator: torch.Generator,
    include_cuda: bool | None = None,
) -> dict:
    numpy_generator, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached = (
        np.random.get_state()
    )
    python_version, python_state, python_gaussian = random.getstate()
    state = {
        "torch_cpu": torch.get_rng_state(),
        "data_loader": data_generator.get_state(),
        "numpy": {
            "generator": numpy_generator,
            "keys": torch.as_tensor(numpy_keys.astype(np.int64, copy=False)).clone(),
            "position": int(numpy_position),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(numpy_cached),
        },
        "python": {
            "version": int(python_version),
            "state": torch.tensor(python_state, dtype=torch.int64),
            "gaussian": python_gaussian,
        },
    }
    if include_cuda is None:
        include_cuda = torch.cuda.is_available()
    if include_cuda:
        if not torch.cuda.is_available():
            raise ValueError("Cannot capture CUDA RNG state because CUDA is unavailable")
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, data_generator: torch.Generator) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch_cpu"].cpu())
    data_generator.set_state(state["data_loader"].cpu())
    if "numpy" in state:
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                str(numpy_state["generator"]),
                numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    if "python" in state:
        python_state = state["python"]
        random.setstate(
            (
                int(python_state["version"]),
                tuple(int(value) for value in python_state["state"].tolist()),
                python_state["gaussian"],
            )
        )
    if "torch_cuda" in state:
        cuda_states = [value.cpu() for value in state["torch_cuda"]]
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "Cannot exactly resume with a different number of visible CUDA devices"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def last_checkpoint_path(best_path: str | Path) -> Path:
    """Derive model_last.pt from model.pt without assuming a suffix."""

    best_path = Path(best_path)
    if best_path.suffix:
        return best_path.with_name(f"{best_path.stem}_last{best_path.suffix}")
    return best_path.with_name(f"{best_path.name}_last")


def atomic_reference_tensor(
    atomic_energies: dict[int, float],
    required_atomic_numbers: set[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    missing = sorted(required_atomic_numbers - set(atomic_energies)) if atomic_energies else []
    if missing:
        raise ValueError(f"Missing atomic reference energies for Z={missing}")
    values = torch.zeros(119, dtype=dtype, device=device)
    for atomic_number, energy in atomic_energies.items():
        values[int(atomic_number)] = float(energy)
    return values


def validate_training_species(training_frames, validation_frames) -> list[int]:
    """Return trained species and reject validation-only chemical elements."""

    training = {
        int(number) for atoms in training_frames for number in atoms.numbers
    }
    validation = {
        int(number) for atoms in validation_frames for number in atoms.numbers
    }
    unseen = sorted(validation - training)
    if unseen:
        raise ValueError(
            "Validation contains atomic species absent from training: "
            f"Z={unseen}. Move representative structures into training or use a "
            "different split; an unseen elemental embedding cannot be validated."
        )
    return sorted(training)


def new_metric_accumulator(device: torch.device) -> dict:
    return {
        "sums": torch.zeros(7, dtype=torch.float64, device=device),
        "structures": 0,
        "force_components": 0,
        "stress_components": 0,
    }


def update_metric_accumulator(
    totals: dict,
    loss: torch.Tensor,
    energy_error: torch.Tensor,
    force_error: torch.Tensor,
    stress_error: torch.Tensor | None,
    loss_terms: dict[str, torch.Tensor] | None = None,
) -> None:
    with torch.no_grad():
        zero = energy_error.detach().new_zeros(())
        stress_sum = (
            stress_to_mandel(stress_error.detach()).square().sum()
            if stress_error is not None
            else zero
        )
        terms = loss_terms or {}
        contribution = torch.stack(
            (
                loss.detach(),
                energy_error.detach().square(),
                force_error.detach().square().sum(),
                stress_sum,
                terms.get("energy", zero).detach(),
                terms.get("forces", zero).detach(),
                terms.get("stress", zero).detach(),
            )
        )
        totals["sums"].add_(contribution.to(dtype=torch.float64))
    totals["structures"] += 1
    totals["force_components"] += force_error.numel()
    if stress_error is not None:
        totals["stress_components"] += 6


def finalize_metrics(totals: dict) -> dict[str, float]:
    if totals["structures"] < 1 or totals["force_components"] < 1:
        raise ValueError("Cannot finalize metrics for an empty dataset")
    (
        loss_sum,
        energy_sq,
        force_sq,
        stress_sq,
        energy_term,
        force_term,
        stress_term,
    ) = totals["sums"].detach().cpu().tolist()
    stress_rmse = (
        math.sqrt(stress_sq / totals["stress_components"])
        if totals["stress_components"]
        else float("nan")
    )
    return {
        "loss": loss_sum / totals["structures"],
        "energy_rmse_mev_atom": 1000.0 * math.sqrt(energy_sq / totals["structures"]),
        "force_rmse_ev_a": math.sqrt(force_sq / totals["force_components"]),
        "stress_rmse_ev_a3": stress_rmse,
        "stress_rmse_mev_a3": 1000.0 * stress_rmse,
        "stress_rmse_gpa": EV_A3_TO_GPA * stress_rmse,
        "stress_structures": totals["stress_components"] // 6,
        # Weighted contribution of each objective term.  A term that is orders of
        # magnitude below the others is effectively unconstrained, which is easy
        # to create accidentally with hand-tuned weights.
        "loss_energy_term": energy_term / totals["structures"],
        "loss_forces_term": force_term / totals["structures"],
        "loss_stress_term": stress_term / totals["structures"],
    }


def supervised_loss(
    energy: torch.Tensor,
    forces: torch.Tensor,
    stress: torch.Tensor,
    item: dict[str, torch.Tensor],
    reference_values: torch.Tensor,
    weights: dict[str, float],
    include_stress: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    dict[str, torch.Tensor],
]:
    target_energy = item["target_energy"] - reference_values[item["z"]].sum()
    energy_error = (energy - target_energy) / item["z"].numel()
    force_error = forces - item["target_forces"]
    energy_term = weights["energy"] * energy_error.square()
    force_term = weights["forces"] * force_error.square().mean()
    loss = energy_term + force_term
    stress_error = stress - item["target_stress"] if include_stress else None
    stress_term = energy_term.new_zeros(())
    if stress_error is not None and weights["stress"] > 0.0:
        stress_term = weights["stress"] * stress_mse(stress, item["target_stress"])
        loss = loss + stress_term
    terms = {"energy": energy_term, "forces": force_term, "stress": stress_term}
    return loss, energy_error, force_error, stress_error, terms


def format_metric_line(name: str, metrics: dict[str, float]) -> str:
    if math.isfinite(metrics["stress_rmse_gpa"]):
        stress = (
            f"S_RMSE={metrics['stress_rmse_mev_a3']:.4f} meV/A^3 "
            f"({metrics['stress_rmse_gpa']:.4f} GPa)"
        )
    else:
        stress = "S_RMSE=n/a"
    return (
        f"  {name}: loss={metrics['loss']:.6e} "
        f"E_RMSE={metrics['energy_rmse_mev_atom']:.3f} meV/atom "
        f"F_RMSE={metrics['force_rmse_ev_a']:.4f} eV/A {stress}\n"
        f"  {name}: loss_terms E={metrics['loss_energy_term']:.3e} "
        f"F={metrics['loss_forces_term']:.3e} "
        f"S={metrics['loss_stress_term']:.3e}"
    )


def evaluate(model, loader, device, reference_values, weights,
             report_stress_metrics=True, max_batch_atoms=None):
    model.eval()
    totals = new_metric_accumulator(device)
    for batch in loader:
        # Batched exactly as the training loop is; validation was otherwise the
        # remaining per-structure bottleneck once training was batched.
        for raws, include_stress in _batched_forward_groups(
            batch, weights, report_stress_metrics, max_batch_atoms
        ):
            items = [
                move(raw, device, non_blocking=device.type == "cuda") for raw in raws
            ]
            if len(items) > 1:
                merged = collate_structures(items, device=device)
                energies, forces, stresses, _ = model(
                    merged, training=False, compute_stress=include_stress
                )
                per_structure = _split_batched_outputs(
                    items, merged, energies, forces, stresses
                )
            else:
                energy, force, stress, _ = model(
                    items[0], training=False, compute_stress=include_stress
                )
                per_structure = [(items[0], energy, force, stress)]
            for item, energy, forces, stress in per_structure:
                loss, energy_error, force_error, stress_error, terms = supervised_loss(
                    energy,
                    forces,
                    stress,
                    item,
                    reference_values,
                    weights,
                    include_stress,
                )
                update_metric_accumulator(
                    totals, loss, energy_error, force_error, stress_error, terms
                )
    return finalize_metrics(totals)


def require_finite_metrics(metrics: dict[str, float], stage: str) -> None:
    required = ("loss", "energy_rmse_mev_atom", "force_rmse_ev_a")
    if metrics["stress_structures"]:
        required += ("stress_rmse_ev_a3",)
    invalid = [name for name in required if not math.isfinite(metrics[name])]
    if invalid:
        raise FloatingPointError(
            f"Nonfinite {stage} metrics: {', '.join(invalid)}. "
            "No checkpoint was written for this epoch."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument(
        "--resume",
        help="Resume exactly from a best- or last-epoch MTACE checkpoint",
    )
    continuation.add_argument(
        "--restart",
        help=(
            "Warm-start model weights from a MTACE checkpoint while resetting "
            "optimizer, scheduler, epoch, and early-stopping state"
        ),
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping")
    config_directory = config_path.parent

    output = resolve_config_path(
        config.get("model_save_path", "mtace.pt"), config_directory
    )
    save_last = bool(config.get("save_last_checkpoint", True))
    configured_last = config.get("last_model_save_path")
    last_output = (
        resolve_config_path(configured_last, config_directory)
        if configured_last
        else last_checkpoint_path(output)
    )
    if save_last and output == last_output:
        raise ValueError("best and last checkpoint paths must be different")
    if args.resume is not None:
        resume_value, restart_value = args.resume, None
    elif args.restart is not None:
        resume_value, restart_value = None, args.restart
    else:
        resume_value = config.get("resume_from")
        restart_value = config.get("restart_from")
        if resume_value and restart_value:
            raise ValueError("Set only one of resume_from and restart_from")
    resume_path = (
        resolve_config_path(resume_value, config_directory)
        if resume_value
        else None
    )
    restart_path = (
        resolve_config_path(restart_value, config_directory)
        if restart_value
        else None
    )
    checkpoint_path = resume_path or restart_path
    checkpoint_preview = (
        load_checkpoint(checkpoint_path, map_location="cpu")
        if checkpoint_path is not None
        else None
    )
    resume_preview = checkpoint_preview if resume_path is not None else None
    restart_preview = checkpoint_preview if restart_path is not None else None
    if resume_preview is not None and int(
        resume_preview.get("training_objective_version", 0)
    ) != 2:
        raise ValueError(
            "This checkpoint predates the audited Mandel-stress objective and cannot "
            "be resumed without mixing two different losses. Start a fresh training run."
        )

    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device requests CUDA, but torch.cuda.is_available() is false")
    dtype_setting = config.get("dtype")
    if dtype_setting is None and checkpoint_preview is not None:
        dtype_setting = checkpoint_preview.get("model_dtype", "float32")
    model_dtype = parse_training_dtype(dtype_setting)
    deterministic = bool(config.get("deterministic_algorithms", False))
    determinism = configure_determinism(deterministic, device)
    if device.type == "cuda":
        allow_tf32 = bool(config.get("allow_tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(
            "high" if bool(config.get("allow_tf32", False)) else "highest"
        )
    if config.get("torch_num_threads") is not None:
        thread_count = int(config["torch_num_threads"])
        if thread_count < 1:
            raise ValueError("torch_num_threads must be positive")
        torch.set_num_threads(thread_count)

    epochs = int(config.get("epochs", 100))
    stop_after_epoch = int(config.get("stop_after_epoch", epochs))
    batch_size = int(config.get("batch_size", 1))
    patience = int(config.get("early_stopping_patience", 20))
    num_workers = int(config.get("num_workers", 0))
    max_batch_atoms = config.get("max_batch_atoms") or None
    clip_grad_norm = float(config.get("clip_grad_norm", 10.0))
    if epochs < 1 or batch_size < 1 or stop_after_epoch < 1:
        raise ValueError("epochs, stop_after_epoch, and batch_size must be positive")
    stop_after_epoch = min(stop_after_epoch, epochs)
    if stop_after_epoch < epochs and not save_last:
        raise ValueError("stop_after_epoch requires save_last_checkpoint: true")
    if patience < 0 or num_workers < 0:
        raise ValueError("early_stopping_patience and num_workers must be nonnegative")
    if not math.isfinite(clip_grad_norm) or clip_grad_norm <= 0.0:
        raise ValueError("clip_grad_norm must be positive and finite")

    train_file = resolve_config_path(config["train_file"], config_directory)
    if not train_file.is_file():
        raise FileNotFoundError(f"Training trajectory does not exist: {train_file}")
    valid_file = (
        resolve_config_path(config["valid_file"], config_directory)
        if config.get("valid_file")
        else None
    )
    if valid_file is not None and not valid_file.is_file():
        raise FileNotFoundError(f"Validation trajectory does not exist: {valid_file}")
    if valid_file == train_file:
        raise ValueError("Training and validation trajectories must be different files")

    frames = read(train_file, index=":")
    if valid_file is not None:
        training_frames, validation_frames = frames, read(valid_file, index=":")
        excluded_indices = []
    else:
        training_frames, validation_frames, excluded_indices = split_frames(
            frames,
            validation_fraction=float(config.get("validation_fraction", 0.1)),
            seed=seed,
            mode=config.get("split_mode", "blocked"),
            block_size=int(config.get("split_block_size", 25)),
            gap=int(config.get("split_gap", 0)),
        )

    architecture = str(config.get("architecture", "mtace_v2"))
    if architecture == "mtace_v2":
        model_class = MambaACEV2
        excluded = {"remove_pair_self_contractions"}
    elif architecture == "mtace_canonical":
        model_class = CanonicalMambaACE
        excluded = {
            "radial_mlp_hidden",
            "radial_mlp_layers",
            "avg_num_neighbors",
            "shell_r_min",
            "shell_boundary_mode",
            "continuum_mode",
            "mamba_rotary_layout",
            "mamba_scan_mode",
            "ffn_type",
            "invariant_pair_channels",
            "attention_dropout",
            "mixer_type",
            "attention_heads",
            "tokenizer_type",
            "num_shells",
            "shell_coupling_mode",
            # CanonicalMambaACE has no schedule and no routed experts.
            "mixer_schedule",
            "gate_norm",
            "num_experts",
            "expert_hidden",
            "expert_latent_dim",
            "router_tau",
            "router_switch",
            "router_threshold_init",
            "router_balance_rate",
            "router_balance_target",
            "routing_backend",
        }
    else:
        raise ValueError("architecture must be 'mtace_v2' or 'mtace_canonical'")
    requested_model_config = {
        key: config[key] for key in MODEL_KEYS if key in config and key not in excluded
    }
    # Density normalization.  Without it the ACE density scales with coordination
    # and the order-nu correlation as (n_neigh)^(nu-1), which makes activations,
    # and therefore accuracy, depend on the local density of the training set.
    if (
        architecture == "mtace_v2"
        and checkpoint_preview is None
        and config.get("avg_num_neighbors") in {None, "auto"}
    ):
        estimated = average_num_neighbors(
            training_frames, float(config.get("r_max", 5.0))
        )
        requested_model_config["avg_num_neighbors"] = estimated
        print(
            f"avg_num_neighbors=auto -> {estimated:.4f} "
            "(estimated from the training split)",
            flush=True,
        )
    if checkpoint_preview is not None:
        continuation_name = "Resume" if resume_preview is not None else "Restart"
        if checkpoint_preview.get("architecture") != architecture:
            raise ValueError(
                f"{continuation_name} checkpoint architecture does not match the configuration"
            )
        conflicts = conflicting_model_settings(
            checkpoint_preview, requested_model_config
        )
        if conflicts and resume_preview is not None:
            warm_restart_parent = (
                resume_preview.get("run_signature", {}).get("restart_parent")
            )
            if warm_restart_parent is None:
                raise ValueError(
                    "Model settings cannot change during an exact resume; "
                    f"conflicting keys: {', '.join(conflicts)}"
                )
        model, loaded_checkpoint = restore_model(checkpoint_path, device=device)
        if next(model.parameters()).dtype != model_dtype:
            raise ValueError(
                f"{continuation_name} checkpoint dtype does not match the configured dtype"
            )
        model_config = migrated_model_config(loaded_checkpoint)
        resume_checkpoint = loaded_checkpoint if resume_preview is not None else None
        restart_checkpoint = loaded_checkpoint if restart_preview is not None else None
        if restart_checkpoint is not None:
            print(
                "restart_model_contract=checkpoint "
                f"architecture_version={restart_checkpoint.get('architecture_version', 1)}",
                flush=True,
            )
    else:
        resume_checkpoint = None
        restart_checkpoint = None
        model_config = requested_model_config
        model = model_class(**model_config).to(device=device, dtype=model_dtype)

    precompute_neighbors = bool(config.get("precompute_neighbors", True))
    training_dataset = AtomisticDataset(
        training_frames,
        model.r_max,
        precompute_neighbors,
        dtype=model_dtype,
    )
    validation_dataset = AtomisticDataset(
        validation_frames,
        model.r_max,
        True,
        dtype=model_dtype,
    )
    weights = {
        "energy": finite_nonnegative("energy_weight", config.get("energy_weight", 1.0)),
        "forces": finite_nonnegative("forces_weight", config.get("forces_weight", 1.0)),
        "stress": finite_nonnegative("stress_weight", config.get("stress_weight", 0.0)),
    }
    if not any(value > 0.0 for value in weights.values()):
        raise ValueError("At least one loss weight must be positive")
    # Raw loss weights are dimensionful, so the balance between the three terms
    # depends on the units and on the system.  With the values shipped in earlier
    # examples the energy term sat about four orders of magnitude below the force
    # term, which means energies were effectively unconstrained.  Dividing each
    # squared-error term by the dataset variance of its target makes the weights
    # dimensionless, comparable, and transferable.
    normalize_loss = bool(config.get("normalize_loss_weights", False))
    loss_scales = {"energy": 1.0, "forces": 1.0, "stress": 1.0}
    if normalize_loss:
        statistics = target_statistics(training_frames)
        loss_scales = {
            "energy": statistics["energy_ev_per_atom"],
            "forces": statistics["forces_ev_per_angstrom"],
            "stress": statistics["stress_ev_per_angstrom3"],
        }
        for key, scale in loss_scales.items():
            if scale > 0.0:
                weights[key] = weights[key] / (scale * scale)
        print(
            "loss_normalization=on  sigma_E={:.6g} eV/atom  sigma_F={:.6g} eV/A  "
            "sigma_S={:.6g} eV/A^3".format(
                loss_scales["energy"], loss_scales["forces"], loss_scales["stress"]
            ),
            flush=True,
        )
        print(
            "  effective weights: energy={:.6g} forces={:.6g} stress={:.6g}".format(
                weights["energy"], weights["forces"], weights["stress"]
            ),
            flush=True,
        )
    if weights["stress"] > 0.0 and training_dataset.has_stress_count == 0:
        raise ValueError("stress_weight is positive but the training set has no stress labels")
    if weights["stress"] > 0.0 and validation_dataset.has_stress_count == 0:
        raise ValueError(
            "stress_weight is positive but validation has no stress labels for model selection"
        )

    # Pairs closer than the inner shell radius all fold onto shell zero, so the
    # mixer loses every bit of radial resolution below it -- precisely where the
    # potential is steepest.  The direct ACE path still resolves them, so the
    # degradation is silent; refuse it rather than discover it during molecular
    # dynamics.
    shell_r_min = float(getattr(model.ace, "shell_r_min", 0.0) or 0.0)
    if shell_r_min > 0.0:
        shortest = minimum_edge_distance(training_frames, model.r_max)
        margin = float(config.get("shell_r_min_margin", 0.25))
        if shell_r_min > shortest - margin:
            raise ValueError(
                f"shell_r_min={shell_r_min:.3f} A is not safely below the shortest "
                f"training distance {shortest:.3f} A (margin {margin:.3f} A). Every "
                "closer pair folds onto shell 0 and the mixer loses all radial "
                "resolution there. Lower shell_r_min or reduce shell_r_min_margin "
                "deliberately."
            )
        print(
            f"shell_r_min={shell_r_min:.3f} A  shortest_training_distance="
            f"{shortest:.3f} A  margin={shortest - shell_r_min:.3f} A",
            flush=True,
        )

    training_atomic_numbers = validate_training_species(
        training_frames, validation_frames
    )
    configured_atomic_energies = parse_atomic_energies(config.get("atomic_energies"))
    inherited_restart_checkpoint = restart_checkpoint
    if (
        inherited_restart_checkpoint is None
        and resume_checkpoint is not None
        and resume_checkpoint.get("run_signature", {}).get("restart_parent") is not None
    ):
        inherited_restart_checkpoint = resume_checkpoint
    if inherited_restart_checkpoint is not None:
        checkpoint_atomic_energies = {
            int(key): float(value)
            for key, value in inherited_restart_checkpoint.get(
                "atomic_energies", {}
            ).items()
        }
        if (
            config.get("atomic_energies") is not None
            and configured_atomic_energies != checkpoint_atomic_energies
        ):
            raise ValueError(
                "Warm restart must preserve the checkpoint atomic reference energies; "
                "changing them would shift the physical energy without correcting the readout"
            )
        atomic_energies = checkpoint_atomic_energies
        checkpoint_atomic_numbers = sorted(
            {
                int(number)
                for number in inherited_restart_checkpoint.get("atomic_numbers", [])
            }
            or set(atomic_energies)
        )
        if checkpoint_atomic_numbers:
            unsupported = sorted(
                set(training_atomic_numbers) - set(checkpoint_atomic_numbers)
            )
            if unsupported:
                raise ValueError(
                    "Warm-restart data contain species absent from the parent checkpoint: "
                    f"Z={unsupported}"
                )
            model_atomic_numbers = checkpoint_atomic_numbers
        else:
            model_atomic_numbers = training_atomic_numbers
    else:
        atomic_energies = configured_atomic_energies
        if not atomic_energies and config.get("solve_atomic_energies", True):
            atomic_energies = solve_atomic_energies(training_frames)
        if resume_checkpoint is not None:
            model_atomic_numbers = sorted(
                int(number)
                for number in resume_checkpoint.get(
                    "atomic_numbers", training_atomic_numbers
                )
            )
            unsupported = sorted(
                set(training_atomic_numbers) - set(model_atomic_numbers)
            )
            if unsupported:
                raise ValueError(
                    "Resume data contain species absent from the checkpoint: "
                    f"Z={unsupported}"
                )
        else:
            model_atomic_numbers = training_atomic_numbers
    reference_values = atomic_reference_tensor(
        atomic_energies,
        set(model_atomic_numbers),
        device,
        dtype=model_dtype,
    )
    report_stress_metrics = bool(config.get("report_stress_metrics", True))
    minimum_learning_rate = finite_nonnegative(
        "minimum_learning_rate", config.get("minimum_learning_rate", 1.0e-5)
    )

    run_signature = {
        "resume_contract_version": 2,
        "training_source_sha256": file_sha256(train_file),
        "validation_source_sha256": (
            file_sha256(valid_file) if valid_file is not None else None
        ),
        "training_frames": len(training_frames),
        "validation_frames": len(validation_frames),
        "excluded_frames": len(excluded_indices),
        "split_mode": "external" if valid_file is not None else config.get("split_mode", "blocked"),
        "split_seed": seed,
        "split_block_size": int(config.get("split_block_size", 25)),
        "split_gap": int(config.get("split_gap", 0)),
        "validation_fraction": float(config.get("validation_fraction", 0.1)),
        "dtype": str(model_dtype).split(".", 1)[-1],
        "batch_size": batch_size,
        "num_workers": num_workers,
        "precompute_neighbors": precompute_neighbors,
        "energy_weight": weights["energy"],
        "forces_weight": weights["forces"],
        "stress_weight": weights["stress"],
        "clip_grad_norm": clip_grad_norm,
        "optimizer": str(config.get("optimizer", "adamw")).lower(),
        "optimizer_hyperparameters": {
            key: config.get(key) for key in OPTIMIZER_CONTRACT_KEYS if key in config
        },
        "minimum_learning_rate": minimum_learning_rate,
        "early_stopping_patience": patience,
        "report_stress_metrics": report_stress_metrics,
        "deterministic_algorithms": deterministic,
        "allow_tf32": bool(config.get("allow_tf32", False)),
        "torch_num_threads": torch.get_num_threads(),
        "device": str(device),
        "visible_cuda_devices": torch.cuda.device_count() if device.type == "cuda" else 0,
        "software": {
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
        },
    }
    if restart_checkpoint is not None:
        run_signature["restart_parent"] = {
            "sha256": file_sha256(restart_path),
            "epoch": int(restart_checkpoint.get("epoch", 0)),
            "architecture_version": int(
                restart_checkpoint.get("architecture_version", 1)
            ),
        }
    if resume_checkpoint is not None:
        checkpoint_signature = resume_checkpoint.get("run_signature") or {}
        if not checkpoint_signature:
            raise ValueError(
                "Resume checkpoint lacks the numerical run contract required for "
                "an exact continuation"
            )
        if "restart_parent" in checkpoint_signature:
            run_signature["restart_parent"] = checkpoint_signature["restart_parent"]
        if checkpoint_signature != run_signature:
            raise ValueError(
                "Training data, numerical settings, or optimizer configuration differ "
                "from the resume checkpoint"
            )
        checkpoint_references = {
            int(key): float(value)
            for key, value in resume_checkpoint.get("atomic_energies", {}).items()
        }
        if checkpoint_references != atomic_energies:
            raise ValueError("Atomic reference energies differ from the resume checkpoint")
        if sorted(resume_checkpoint.get("atomic_numbers", [])) != model_atomic_numbers:
            raise ValueError("Training species differ from the resume checkpoint")

    optimizer = build_optimizer(model, config)
    # Averaging the optimizer trajectory is a large, cheap accuracy win for
    # force-and-energy fitting, where small batches keep the raw iterate rattling
    # around the minimum.  Validation, model selection and the deployed "best"
    # checkpoint use the averaged weights; the "last" checkpoint keeps the raw
    # iterate so an exact resume is still possible.
    ema_decay = config.get("ema_decay")
    ema = (
        ExponentialMovingAverage(model, float(ema_decay))
        if ema_decay is not None
        else None
    )
    if ema is not None:
        print(f"ema_decay={float(ema_decay):g}", flush=True)
    print(
        f"optimizer={str(config.get('optimizer', 'adamw')).lower()}; "
        f"{optimizer_group_summary(optimizer)}"
    )
    if any(
        minimum_learning_rate > float(group["lr"])
        for group in optimizer.param_groups
    ):
        raise ValueError(
            "minimum_learning_rate cannot exceed an optimizer group's initial learning rate"
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=minimum_learning_rate,
    )
    data_generator = torch.Generator().manual_seed(seed + 1)
    pin_memory = device.type == "cuda"
    loader_options = {
        "batch_size": batch_size,
        "collate_fn": AtomisticDataset.collate,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = bool(
            config.get("persistent_workers", False)
        )
        loader_options["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    configure_worker_sharing(num_workers)
    training_loader = DataLoader(
        training_dataset,
        shuffle=True,
        generator=data_generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )
    print(
        f"data: train={len(training_dataset)} (stress={training_dataset.has_stress_count}) "
        f"valid={len(validation_dataset)} (stress={validation_dataset.has_stress_count}) "
        f"excluded_by_gap={len(excluded_indices)} dtype={str(model_dtype).split('.', 1)[-1]} "
        f"determinism={determinism}",
        flush=True,
    )
    print(f"best_checkpoint={output}", flush=True)
    if save_last:
        print(f"last_checkpoint={last_output}", flush=True)

    gate_every = int(config.get("gate_diagnostic_every", 10))
    gate_probe = None
    if gate_every > 0 and architecture == "mtace_v2":
        probe = training_dataset[0]
        gate_probe = {
            "z": probe["z"].to(device),
            "pos": probe["pos"].to(device),
            "cell": probe["cell"].to(device),
            "edge_index": probe["edge_index"].to(device),
            "edge_shift": probe["edge_shift"].to(device),
        }

    start_epoch = 1
    best = float("inf")
    stale = 0
    if resume_checkpoint is not None:
        if resume_checkpoint.get("optimizer_state_dict") is None:
            raise ValueError("Resume checkpoint does not contain optimizer state")
        if resume_checkpoint.get("scheduler_state_dict") is None:
            raise ValueError("Resume checkpoint does not contain scheduler state")
        original_epochs = int(
            resume_checkpoint.get("training_config", {}).get("epochs", epochs)
        )
        if original_epochs != epochs:
            raise ValueError(
                "epochs cannot change during exact resume because it defines the cosine schedule"
            )
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_value = resume_checkpoint.get("best_validation_loss")
        best = float(best_value) if best_value is not None else float("inf")
        stale = int(resume_checkpoint.get("stale_epochs", 0))
        rng_state = resume_checkpoint.get("rng_state") or {}
        if not rng_state:
            raise ValueError(
                "Resume checkpoint lacks RNG state and cannot provide an exact continuation"
            )
        restore_rng_state(rng_state, data_generator)
        saved_ema = resume_checkpoint.get("ema_state_dict")
        if (ema is None) != (saved_ema is None):
            raise ValueError(
                "ema_decay must match the resume checkpoint: the averaged weights "
                "are part of the training state"
            )
        if ema is not None and saved_ema is not None:
            ema.load_state_dict(saved_ema)
        print(f"resumed_from={resume_path} next_epoch={start_epoch}", flush=True)
    elif restart_checkpoint is not None:
        print(
            f"restarted_from={restart_path} parent_epoch="
            f"{int(restart_checkpoint.get('epoch', 0))} next_epoch=1 "
            "optimizer=reset scheduler=reset early_stopping=reset",
            flush=True,
        )
    if start_epoch > epochs:
        raise ValueError("Resume checkpoint has already reached the configured epoch count")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_totals = new_metric_accumulator(device)
        learning_rates = ",".join(
            f"{group['lr']:.3e}" for group in optimizer.param_groups
        )
        for batch_index, batch in enumerate(training_loader, start=1):
            batch_loss = torch.zeros((), device=device)
            # One forward over the whole batch when every structure agrees on
            # whether stress is included.  The model mixes atoms only through
            # edge_index, so this is identical to the per-structure loop below --
            # asserted in tests/test_batching.py, including the accumulated
            # parameter gradients.  It exists because a single 40-atom structure
            # leaves a GPU almost idle; batching is what actually feeds it.
            grouped = _batched_forward_groups(
                batch, weights, report_stress_metrics, max_batch_atoms
            )
            for raws, include_stress in grouped:
                items = [move(raw, device, non_blocking=pin_memory) for raw in raws]
                if len(items) > 1:
                    merged = collate_structures(items, device=device)
                    energies, forces, stresses, _ = model(
                        merged, training=True, compute_stress=include_stress
                    )
                    per_structure = _split_batched_outputs(
                        items, merged, energies, forces, stresses
                    )
                else:
                    energy, force, stress, _ = model(
                        items[0], training=True, compute_stress=include_stress
                    )
                    per_structure = [(items[0], energy, force, stress)]
                chunk_loss = torch.zeros((), device=device)
                for item, energy, forces, stress in per_structure:
                    loss, energy_error, force_error, stress_error, terms = supervised_loss(
                        energy,
                        forces,
                        stress,
                        item,
                        reference_values,
                        weights,
                        include_stress,
                    )
                    update_metric_accumulator(
                        training_totals,
                        loss,
                        energy_error,
                        force_error,
                        stress_error,
                        terms,
                    )
                    chunk_loss = chunk_loss + loss / len(batch)
                if not bool(torch.isfinite(chunk_loss.detach())):
                    raise FloatingPointError(
                        f"Nonfinite training loss at epoch {epoch}, batch {batch_index}"
                    )
                # Backward per chunk rather than once for the whole batch.  The
                # gradient is identical, because the batch loss is a sum over
                # chunks and gradients add, but each chunk's graph is released
                # as soon as it is used.  Accumulating the graph across chunks
                # instead would make max_batch_atoms pointless: peak memory
                # would still scale with the whole batch, which is exactly how
                # a 768-atom water batch exhausted a 16 GiB card.
                chunk_loss.backward()
                batch_loss = batch_loss + chunk_loss.detach()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        training_metrics = finalize_metrics(training_totals)
        if gate_probe is not None and (epoch == start_epoch or epoch % gate_every == 0):
            report = model.gate_shell_dependence(**gate_probe)
            summary = " ".join(
                f"L{int(entry['layer'])}:res={entry['residual_fraction']:.4f}"
                f",|g|={entry['gate_abs_mean']:.3f}"
                f",sd_shell={entry['gate_std_over_shells']:.4f}"
                for entry in report
            )
            # residual_fraction is the share of the equivariant update that a
            # shell-constant gate cannot reproduce.  Because sum_k T_ik = A_i, a
            # constant gate is exactly the direct ACE path, so a value near zero
            # means the sequence mixer is contributing nothing.
            print(f"  gate_shell_dependence {summary}", flush=True)
        if ema is not None:
            ema.store(model)
        try:
            metrics = evaluate(
                model,
                validation_loader,
                device,
                reference_values,
                weights,
                report_stress_metrics,
                max_batch_atoms,
            )
        finally:
            if ema is not None:
                ema.restore(model)
        require_finite_metrics(training_metrics, "training")
        require_finite_metrics(metrics, "validation")
        scheduler.step()
        print(f"epoch={epoch:04d} lr_used=[{learning_rates}]", flush=True)
        print(format_metric_line("train", training_metrics), flush=True)
        print(format_metric_line("valid", metrics), flush=True)
        improved = metrics["loss"] < best
        if improved:
            best, stale = metrics["loss"], 0
        else:
            stale += 1
        rng_state = capture_rng_state(
            data_generator, include_cuda=device.type == "cuda"
        )
        if improved:
            if ema is not None:
                ema.store(model)
            save_checkpoint(
                output,
                model,
                model_config,
                config,
                optimizer,
                epoch,
                atomic_energies,
                model_atomic_numbers,
                scheduler=scheduler,
                metrics=metrics,
                training_metrics=training_metrics,
                best_validation_loss=best,
                checkpoint_role="best",
                stale_epochs=stale,
                rng_state=rng_state,
                run_signature=run_signature,
                ema_state=ema.state_dict() if ema is not None else None,
            )
            if ema is not None:
                ema.restore(model)
        if save_last:
            save_checkpoint(
                last_output,
                model,
                model_config,
                config,
                optimizer,
                epoch,
                atomic_energies,
                model_atomic_numbers,
                scheduler=scheduler,
                metrics=metrics,
                training_metrics=training_metrics,
                best_validation_loss=best,
                checkpoint_role="last",
                stale_epochs=stale,
                rng_state=rng_state,
                run_signature=run_signature,
                ema_state=ema.state_dict() if ema is not None else None,
            )
        if not improved and patience > 0 and stale >= patience:
            print(
                f"early_stopping_epoch={epoch} stale_epochs={stale}",
                flush=True,
            )
            break
        if epoch >= stop_after_epoch:
            if epoch < epochs:
                print(
                    f"stopped_cleanly_after_epoch={epoch}; resume from {last_output}",
                    flush=True,
                )
            break


if __name__ == "__main__":
    main()
