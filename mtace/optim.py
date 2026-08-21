"""Muon matrix optimization with auxiliary AdamW for MTACE."""

from __future__ import annotations

import math
from fnmatch import fnmatch
from typing import Iterable, MutableMapping, Sequence

import torch


DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
DEFAULT_NS_STEPS = 5
DEFAULT_NS_EPS = 1.0e-7
DEFAULT_NS_PRECISION = "auto"
DEFAULT_ADJUST_LR_FN = "match_rms_adamw"
_ADJUST_LR_MODES = {"original", "match_rms_adamw", "spectral_unclamped"}
_NS_PRECISIONS = {"auto", "float32", "bfloat16", "parameter"}


def _validate_ns_arguments(
    coefficients: Sequence[float],
    steps: int,
    eps: float,
    precision: str,
) -> tuple[float, float, float]:
    if len(coefficients) != 3:
        raise ValueError("Newton-Schulz coefficients must contain exactly three values")
    if not 0 <= int(steps) < 100:
        raise ValueError("Newton-Schulz steps must lie in [0, 100)")
    if not math.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("Newton-Schulz epsilon must be positive")
    if precision not in _NS_PRECISIONS:
        options = ", ".join(sorted(_NS_PRECISIONS))
        raise ValueError(f"ns_precision must be one of: {options}")
    parsed = tuple(float(value) for value in coefficients)
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError("Newton-Schulz coefficients must be finite")
    return parsed


def _orthogonalization_dtype(matrix: torch.Tensor, precision: str) -> torch.dtype:
    if precision == "bfloat16":
        return torch.bfloat16
    if precision == "parameter":
        return matrix.dtype
    if precision == "auto" and matrix.device.type == "cuda":
        return torch.float64 if matrix.dtype == torch.float64 else torch.bfloat16
    return torch.float64 if matrix.dtype == torch.float64 else torch.float32


def zeropower_via_newton_schulz5(
    matrix: torch.Tensor,
    steps: int = DEFAULT_NS_STEPS,
    coefficients: Sequence[float] = DEFAULT_NS_COEFFICIENTS,
    eps: float = DEFAULT_NS_EPS,
    precision: str = DEFAULT_NS_PRECISION,
) -> torch.Tensor:
    """Approximate the matrix zeroth power with the Muon quintic iteration.

    If ``matrix = U S V^T``, its exact zeroth power is ``U V^T``. The aggressive
    quintic coefficients used by Muon deliberately favor rapid conditioning over
    exact convergence of every transformed singular value to one.
    """

    if matrix.ndim != 2:
        raise ValueError("Muon orthogonalization requires a two-dimensional matrix")
    if matrix.is_complex():
        raise ValueError("Muon orthogonalization does not support complex matrices")
    a, b, c = _validate_ns_arguments(coefficients, steps, eps, precision)
    if not matrix.is_floating_point():
        raise ValueError("Muon orthogonalization requires a floating-point matrix")
    update = matrix.to(_orthogonalization_dtype(matrix, precision))
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.mT

    update = update / update.norm().clamp_min(float(eps))
    for _ in range(int(steps)):
        gram = update @ update.mT
        polynomial = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        update = torch.addmm(update, polynomial, update, beta=a)

    return update.mT if transposed else update


# Canonical Muon implementations commonly expose this unseparated spelling.
zeropower_via_newtonschulz5 = zeropower_via_newton_schulz5


def adjusted_muon_learning_rate(
    learning_rate: float,
    matrix_shape: Sequence[int],
    mode: str | None = DEFAULT_ADJUST_LR_FN,
) -> float:
    """Scale a Muon step consistently across rectangular hidden matrices."""

    if len(matrix_shape) != 2:
        raise ValueError("Muon learning-rate adjustment requires a matrix shape")
    if not math.isfinite(float(learning_rate)) or learning_rate < 0.0:
        raise ValueError("Muon learning rate must be finite and nonnegative")
    mode = "original" if mode is None else str(mode)
    if mode not in _ADJUST_LR_MODES:
        options = ", ".join(sorted(_ADJUST_LR_MODES))
        raise ValueError(f"adjust_lr_fn must be one of: {options}")
    rows, columns = int(matrix_shape[0]), int(matrix_shape[1])
    if rows < 1 or columns < 1:
        raise ValueError("Muon matrix dimensions must be positive")
    if mode == "original":
        factor = math.sqrt(max(1.0, rows / columns))
    elif mode == "match_rms_adamw":
        factor = 0.2 * math.sqrt(max(rows, columns))
    else:  # spectral_unclamped
        factor = math.sqrt(rows / columns)
    return float(learning_rate) * factor


def muon_update(
    gradient: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum: float = 0.95,
    ns_steps: int = DEFAULT_NS_STEPS,
    nesterov: bool = True,
    ns_coefficients: Sequence[float] = DEFAULT_NS_COEFFICIENTS,
    eps: float = DEFAULT_NS_EPS,
    ns_precision: str = DEFAULT_NS_PRECISION,
) -> torch.Tensor:
    """Update momentum in place and return its orthogonalized Muon direction."""

    if gradient.ndim != 2 or momentum_buffer.shape != gradient.shape:
        raise ValueError("Muon gradient and momentum must be equal-shaped matrices")
    if not 0.0 <= float(momentum) < 1.0:
        raise ValueError("Muon momentum must lie in [0, 1)")
    if gradient.is_sparse:
        raise RuntimeError("Muon does not support sparse gradients")
    if gradient.is_complex():
        raise RuntimeError("Muon does not support complex gradients")
    if not gradient.is_floating_point():
        raise RuntimeError("Muon requires floating-point gradients")

    momentum_buffer.lerp_(gradient, 1.0 - float(momentum))
    direction = (
        gradient.lerp(momentum_buffer, float(momentum))
        if nesterov
        else momentum_buffer
    )
    return zeropower_via_newton_schulz5(
        direction,
        steps=ns_steps,
        coefficients=ns_coefficients,
        eps=eps,
        precision=ns_precision,
    )


def adamw_update(
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    """Update Adam moments in place and return the bias-corrected direction."""

    beta1, beta2 = betas
    exp_avg.lerp_(gradient, 1.0 - beta1)
    exp_avg_sq.lerp_(gradient.square(), 1.0 - beta2)
    first = exp_avg / (1.0 - beta1**step)
    second = exp_avg_sq / (1.0 - beta2**step)
    return first / (second.sqrt() + eps)


# Short public alias retained for callers of the canonical hybrid API.
adam_update = adamw_update


def annotate_equivariant_blocks(model: torch.nn.Module) -> int:
    """Record the matrix structure hidden inside flattened e3nn weights.

    ``o3.Linear`` and ``o3.FullyConnectedTensorProduct`` store all of their path
    weights in one flat vector, so a parameter-shape test sends them to AdamW even
    though each path *is* a matrix: ``(mul_in, mul_out)`` for a linear map and
    ``(mul_in1, mul_in2, mul_out)`` for a tensor product, which matricizes as
    ``(mul_in1 * mul_in2, mul_out)``.

    Muon's spectral normalization is defined per matrix, so the physically correct
    generalization for an equivariant layer is to orthogonalize each path block
    separately rather than to skip the tensor.  Blocks belonging to different
    irreps are independent maps between different representation spaces; mixing
    them into one matrix would be meaningless, and ignoring them leaves most of an
    equivariant model outside Muon.

    Returns the number of annotated parameters.
    """

    from e3nn import o3

    annotated = 0
    for module in model.modules():
        if not isinstance(module, (o3.Linear, o3.FullyConnectedTensorProduct)):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.nn.Parameter) or weight.ndim != 1:
            continue
        blocks: list[tuple[int, int, int]] = []
        offset = 0
        for view in module.weight_views():
            shape = tuple(int(size) for size in view.shape)
            numel = 1
            for size in shape:
                numel *= size
            # Matricize a tensor-product path as (in1 * in2, out); a linear path is
            # already a matrix.  Blocks with a unit dimension have no nontrivial
            # spectrum and are left to AdamW.
            rows = 1
            for size in shape[:-1]:
                rows *= size
            columns = shape[-1] if shape else 1
            if min(rows, columns) > 1:
                blocks.append((offset, rows, columns))
            offset += numel
        if blocks and offset == weight.numel():
            weight._irrep_blocks = tuple(blocks)
            annotated += 1
    return annotated


def muon_update_blockwise(
    gradient: torch.Tensor,
    momentum_buffer: torch.Tensor,
    blocks: Sequence[tuple[int, int, int]],
    momentum: float = 0.95,
    ns_steps: int = DEFAULT_NS_STEPS,
    nesterov: bool = True,
    ns_coefficients: Sequence[float] = DEFAULT_NS_COEFFICIENTS,
    eps: float = DEFAULT_NS_EPS,
    ns_precision: str = DEFAULT_NS_PRECISION,
    adjust_lr_fn: str = DEFAULT_ADJUST_LR_FN,
    learning_rate: float = 1.0,
) -> torch.Tensor:
    """Per-irrep-block Muon direction for one flattened equivariant weight.

    The returned tensor already carries each block's shape-dependent learning-rate
    factor, because blocks of one parameter generally have different shapes and
    therefore different Muon scalings.  The caller applies a single ``-1`` step.
    """

    if gradient.ndim != 1 or momentum_buffer.shape != gradient.shape:
        raise ValueError("blockwise Muon expects a flat gradient and momentum")
    if not 0.0 <= float(momentum) < 1.0:
        raise ValueError("Muon momentum must lie in [0, 1)")

    momentum_buffer.lerp_(gradient, 1.0 - float(momentum))
    direction = (
        gradient.lerp(momentum_buffer, float(momentum)) if nesterov else momentum_buffer
    )
    update = torch.zeros_like(gradient)
    for offset, rows, columns in blocks:
        block = direction[offset : offset + rows * columns].view(rows, columns)
        orthogonal = zeropower_via_newton_schulz5(
            block,
            steps=ns_steps,
            coefficients=ns_coefficients,
            eps=eps,
            precision=ns_precision,
        ).to(update.dtype)
        scale = adjusted_muon_learning_rate(
            learning_rate, (rows, columns), mode=adjust_lr_fn
        )
        update[offset : offset + rows * columns] = orthogonal.reshape(-1) * scale
    return update


def _validate_group(group: MutableMapping) -> None:
    learning_rate = float(group["lr"])
    weight_decay = float(group["weight_decay"])
    if not math.isfinite(learning_rate) or not math.isfinite(weight_decay):
        raise ValueError("learning rates and weight decay must be finite")
    if learning_rate < 0.0 or weight_decay < 0.0:
        raise ValueError("learning rates and weight decay must be nonnegative")
    if group["use_muon"]:
        momentum = float(group["momentum"])
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Muon momentum must lie in [0, 1)")
        _validate_ns_arguments(
            group["ns_coefficients"],
            int(group["ns_steps"]),
            float(group["eps"]),
            str(group["ns_precision"]),
        )
        if group["adjust_lr_fn"] not in _ADJUST_LR_MODES:
            raise ValueError("unsupported Muon learning-rate adjustment")
        for parameter in group["params"]:
            if parameter.ndim == 2:
                continue
            if parameter.ndim == 1 and getattr(parameter, "_irrep_blocks", None):
                continue
            raise ValueError(
                "Muon parameter groups may contain only 2D tensors or flattened "
                "e3nn weights annotated with irrep block structure"
            )
    else:
        if len(group["betas"]) != 2:
            raise ValueError("AdamW betas must contain exactly two values")
        beta1, beta2 = (float(value) for value in group["betas"])
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("AdamW beta values must lie in [0, 1)")
        if not math.isfinite(float(group["eps"])) or float(group["eps"]) <= 0.0:
            raise ValueError("AdamW epsilon must be positive")


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """Single-device Muon for hidden matrices and AdamW for all other tensors."""

    def __init__(self, param_groups: Iterable[MutableMapping]):
        groups = list(param_groups)
        if not groups:
            raise ValueError("MuonWithAuxAdamW requires at least one parameter group")
        for group in groups:
            group.setdefault("use_muon", False)
            if group["use_muon"]:
                group.setdefault("lr", 1.0e-3)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("nesterov", True)
                group.setdefault("ns_coefficients", DEFAULT_NS_COEFFICIENTS)
                group.setdefault("eps", DEFAULT_NS_EPS)
                group.setdefault("ns_steps", DEFAULT_NS_STEPS)
                group.setdefault("ns_precision", DEFAULT_NS_PRECISION)
                group.setdefault("adjust_lr_fn", DEFAULT_ADJUST_LR_FN)
                if group["adjust_lr_fn"] is None:
                    group["adjust_lr_fn"] = "original"
            else:
                group.setdefault("lr", 3.0e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1.0e-10)
                group.setdefault("weight_decay", 0.0)
            group["params"] = list(group["params"])
            _validate_group(group)
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group: MutableMapping) -> None:
        base_lr = float(group["lr"])
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            if gradient.is_complex():
                raise RuntimeError("Muon does not support complex gradients")
            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    gradient, memory_format=torch.preserve_format
                )
            blocks = getattr(parameter, "_irrep_blocks", None)
            if parameter.ndim == 1 and blocks:
                # Each irrep path is its own matrix, with its own Muon scaling, so
                # the per-block learning rate is folded into the direction here.
                update = muon_update_blockwise(
                    gradient,
                    state["momentum_buffer"],
                    blocks,
                    momentum=float(group["momentum"]),
                    ns_steps=int(group["ns_steps"]),
                    nesterov=bool(group["nesterov"]),
                    ns_coefficients=group["ns_coefficients"],
                    eps=float(group["eps"]),
                    ns_precision=str(group["ns_precision"]),
                    adjust_lr_fn=str(group["adjust_lr_fn"]),
                    learning_rate=base_lr,
                )
                weight_decay = float(group["weight_decay"])
                if weight_decay:
                    parameter.mul_(1.0 - base_lr * weight_decay)
                parameter.add_(update, alpha=-1.0)
                continue
            update = muon_update(
                gradient,
                state["momentum_buffer"],
                momentum=float(group["momentum"]),
                ns_steps=int(group["ns_steps"]),
                nesterov=bool(group["nesterov"]),
                ns_coefficients=group["ns_coefficients"],
                eps=float(group["eps"]),
                ns_precision=str(group["ns_precision"]),
            )
            weight_decay = float(group["weight_decay"])
            if weight_decay:
                parameter.mul_(1.0 - base_lr * weight_decay)
            step_lr = adjusted_muon_learning_rate(
                base_lr,
                parameter.shape,
                mode=str(group["adjust_lr_fn"]),
            )
            parameter.add_(update, alpha=-step_lr)

    def _step_adamw_group(self, group: MutableMapping) -> None:
        learning_rate = float(group["lr"])
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("auxiliary AdamW does not support sparse gradients")
            if gradient.is_complex():
                raise RuntimeError("auxiliary AdamW does not support complex gradients")
            state = self.state[parameter]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(
                    gradient, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    gradient, memory_format=torch.preserve_format
                )
            state["step"] += 1
            update = adamw_update(
                gradient,
                state["exp_avg"],
                state["exp_avg_sq"],
                int(state["step"]),
                tuple(group["betas"]),
                float(group["eps"]),
            )
            weight_decay = float(group["weight_decay"])
            if weight_decay:
                parameter.mul_(1.0 - learning_rate * weight_decay)
            parameter.add_(update, alpha=-learning_rate)


SingleDeviceMuonWithAuxAdam = MuonWithAuxAdamW


class ExponentialMovingAverage:
    """Shadow copy of the parameters, averaged over training steps.

    Force-and-energy fitting is a small-batch, high-curvature problem and the raw
    iterate rattles around the minimum.  Averaging the trajectory is a standard
    and unusually effective variance reduction for interatomic potentials, and it
    costs one extra copy of the weights.

    The decay is warmed up as ``min(decay, (1 + step) / (10 + step))`` so the
    average is not dominated by the random initialization during the first steps.
    ``store``/``restore`` swap the averaged weights in for evaluation and
    checkpointing and put the raw iterate back afterwards, so training continues
    from the un-averaged trajectory.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        decay = float(decay)
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie strictly between zero and one")
        self.decay = decay
        self.step = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self._saved: dict[str, torch.Tensor] = {}

    def current_decay(self) -> float:
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        decay = self.current_decay()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            shadow = self.shadow.get(name)
            if shadow is None:
                self.shadow[name] = parameter.detach().clone()
                continue
            shadow.lerp_(parameter.detach(), 1.0 - decay)

    @torch.no_grad()
    def store(self, model: torch.nn.Module) -> None:
        """Swap the averaged weights into the model, keeping the raw iterate."""

        self._saved = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name not in self.shadow:
                continue
            self._saved[name] = parameter.detach().clone()
            parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self._saved:
                parameter.copy_(self._saved[name])
        self._saved = {}

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "step": self.step,
            "shadow": {name: value.cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.step = int(state["step"])
        for name, value in state["shadow"].items():
            if name in self.shadow:
                self.shadow[name].copy_(value.to(self.shadow[name]))


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


def _is_no_decay_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    return bool(getattr(parameter, "_no_weight_decay", False)) or (
        lowered.endswith(".bias")
        or lowered.endswith("_bias")
        or "norm" in lowered
        or lowered.endswith("layer_scale")
        or lowered.endswith("ffn_scale")
    )


def _is_mamba_matrix(name: str, parameter: torch.nn.Parameter) -> bool:
    """Select Mamba and scalar-channel block matrices for ablations."""

    return (
        parameter.ndim == 2
        and min(parameter.shape) > 1
        and name.endswith(".weight")
        and _matches_any(
            name,
            ("layers.*.mixer.*weight", "layers.*.scalar_ffn*.weight"),
        )
    )


def _final_readout_parameter_ids(model: torch.nn.Module) -> set[int]:
    """Find the final scalar prediction map, which canonical Muon excludes."""

    readout = getattr(model, "readout", None)
    if not isinstance(readout, torch.nn.Module):
        return set()
    linear_modules = [
        module for module in readout.modules() if isinstance(module, torch.nn.Linear)
    ]
    if not linear_modules:
        return set()
    return {
        id(parameter)
        for parameter in linear_modules[-1].parameters(recurse=False)
    }


def _is_hidden_matrix(
    name: str,
    parameter: torch.nn.Parameter,
    owner: torch.nn.Module | None,
    final_readout_ids: set[int],
) -> bool:
    """Classify an internal matrix according to canonical Muon parameter roles."""

    if (
        parameter.ndim != 2
        or min(parameter.shape) <= 1
        or not name.endswith(".weight")
        or _is_no_decay_parameter(name, parameter)
        or id(parameter) in final_readout_ids
    ):
        return False
    return not isinstance(owner, torch.nn.Embedding)


def _normalize_patterns(patterns: Sequence[str] | str | None) -> tuple[str, ...]:
    if patterns is None:
        return ()
    if isinstance(patterns, str):
        return (patterns,)
    return tuple(str(pattern) for pattern in patterns)


def get_muon_param_groups(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    muon_learning_rate: float | None = None,
    aux_learning_rate: float | None = None,
    momentum: float = 0.95,
    ns_steps: int = DEFAULT_NS_STEPS,
    nesterov: bool = True,
    ns_coefficients: Sequence[float] = DEFAULT_NS_COEFFICIENTS,
    muon_eps: float = DEFAULT_NS_EPS,
    ns_precision: str = DEFAULT_NS_PRECISION,
    adjust_lr_fn: str = DEFAULT_ADJUST_LR_FN,
    aux_betas: tuple[float, float] = (0.9, 0.95),
    aux_eps: float = 1.0e-10,
    parameter_mode: str = "hidden",
    include_patterns: Sequence[str] | str | None = None,
    exclude_patterns: Sequence[str] | str | None = None,
) -> list[dict]:
    """Partition every trainable tensor exactly once between Muon and AdamW."""

    if parameter_mode not in {"hidden", "mamba_only", "all_matrices", "equivariant"}:
        raise ValueError(
            "muon_parameter_mode must be 'hidden', 'mamba_only', 'all_matrices', "
            "or 'equivariant'"
        )
    if parameter_mode == "equivariant":
        # Expose the per-path matrices hidden inside flattened e3nn weights so the
        # equivariant half of the model is not silently excluded from Muon.
        annotate_equivariant_blocks(model)
    include_patterns = _normalize_patterns(include_patterns)
    exclude_patterns = _normalize_patterns(exclude_patterns)
    modules = dict(model.named_modules())
    final_readout_ids = _final_readout_parameter_ids(model)
    muon_parameters = []
    auxiliary_decay = []
    auxiliary_no_decay = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = _is_no_decay_parameter(name, parameter)
        owner_name = name.rpartition(".")[0]
        owner = modules.get(owner_name)
        if parameter_mode == "hidden":
            use_muon = _is_hidden_matrix(
                name, parameter, owner, final_readout_ids
            )
        elif parameter_mode == "mamba_only":
            use_muon = _is_mamba_matrix(name, parameter)
        elif parameter_mode == "equivariant":
            use_muon = _is_hidden_matrix(
                name, parameter, owner, final_readout_ids
            ) or bool(getattr(parameter, "_irrep_blocks", None))
        else:
            use_muon = parameter.ndim == 2 and not no_decay
        if _matches_any(name, include_patterns):
            if parameter.ndim != 2:
                raise ValueError(f"Muon include pattern selected non-matrix parameter {name}")
            use_muon = True
        if _matches_any(name, exclude_patterns):
            use_muon = False

        if use_muon:
            muon_parameters.append(parameter)
        elif no_decay:
            auxiliary_no_decay.append(parameter)
        else:
            auxiliary_decay.append(parameter)

    grouped = muon_parameters + auxiliary_decay + auxiliary_no_decay
    grouped_ids = [id(parameter) for parameter in grouped]
    expected_ids = [
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(expected_ids):
        raise RuntimeError("Muon parameter grouping must cover every trainable tensor once")

    muon_lr = learning_rate if muon_learning_rate is None else muon_learning_rate
    auxiliary_lr = learning_rate if aux_learning_rate is None else aux_learning_rate
    groups = []
    if muon_parameters:
        groups.append(
            {
                "name": f"muon_{parameter_mode}_matrices",
                "params": sorted(muon_parameters, key=lambda item: item.numel(), reverse=True),
                "use_muon": True,
                "lr": float(muon_lr),
                "momentum": float(momentum),
                "weight_decay": float(weight_decay),
                "nesterov": bool(nesterov),
                "ns_coefficients": tuple(float(value) for value in ns_coefficients),
                "eps": float(muon_eps),
                "ns_steps": int(ns_steps),
                "ns_precision": str(ns_precision),
                "adjust_lr_fn": str(adjust_lr_fn),
            }
        )
    if auxiliary_decay:
        groups.append(
            {
                "name": "adamw_aux_decay",
                "params": auxiliary_decay,
                "use_muon": False,
                "lr": float(auxiliary_lr),
                "betas": tuple(float(value) for value in aux_betas),
                "eps": float(aux_eps),
                "weight_decay": float(weight_decay),
            }
        )
    if auxiliary_no_decay:
        groups.append(
            {
                "name": "adamw_aux_no_decay",
                "params": auxiliary_no_decay,
                "use_muon": False,
                "lr": float(auxiliary_lr),
                "betas": tuple(float(value) for value in aux_betas),
                "eps": float(aux_eps),
                "weight_decay": 0.0,
            }
        )
    return groups


def optimizer_group_summary(optimizer: torch.optim.Optimizer) -> str:
    summaries = []
    for group in optimizer.param_groups:
        values = sum(parameter.numel() for parameter in group["params"])
        algorithm = "Muon" if group.get("use_muon", False) else "AdamW"
        summaries.append(
            f"{group.get('name', algorithm.lower())}: {algorithm}, "
            f"tensors={len(group['params'])}, parameters={values}, lr={group['lr']:g}"
        )
    return "; ".join(summaries)


def adamw_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict]:
    """Preserve the original MTACE AdamW decay partition."""

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        target = no_decay if _is_no_decay_parameter(name, parameter) else decay
        target.append(parameter)
    groups = [{"params": decay, "weight_decay": float(weight_decay)}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _config_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"invalid boolean setting: {value}")
    return bool(value)


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Construct AdamW or the configured Muon plus AdamW hybrid optimizer."""

    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    learning_rate = float(config.get("learning_rate", 1.0e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    if optimizer_name == "muon":
        auxiliary_betas = tuple(
            float(value) for value in config.get("muon_aux_betas", (0.9, 0.95))
        )
        ns_coefficients = tuple(
            float(value)
            for value in config.get("muon_ns_coefficients", DEFAULT_NS_COEFFICIENTS)
        )
        ns_steps = config.get(
            "muon_ns_steps",
            config.get("muon_newton_schulz_steps", 5),
        )
        groups = get_muon_param_groups(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            muon_learning_rate=_optional_float(config.get("muon_learning_rate")),
            aux_learning_rate=_optional_float(config.get("muon_aux_learning_rate")),
            momentum=float(config.get("muon_momentum", 0.95)),
            ns_steps=int(ns_steps),
            nesterov=_config_bool(config.get("muon_nesterov", True)),
            ns_coefficients=ns_coefficients,
            muon_eps=float(config.get("muon_eps", 1.0e-7)),
            ns_precision=str(
                config.get("muon_ns_precision", DEFAULT_NS_PRECISION)
            ).lower(),
            adjust_lr_fn=str(
                config.get("muon_adjust_lr_fn", DEFAULT_ADJUST_LR_FN)
            ).lower(),
            aux_betas=auxiliary_betas,
            aux_eps=float(config.get("muon_aux_eps", 1.0e-10)),
            parameter_mode=str(config.get("muon_parameter_mode", "hidden")),
            include_patterns=config.get("muon_include"),
            exclude_patterns=config.get("muon_exclude"),
        )
        return MuonWithAuxAdamW(groups)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            adamw_parameter_groups(model, weight_decay),
            lr=learning_rate,
        )
    raise ValueError("optimizer must be 'adamw' or 'muon'")
