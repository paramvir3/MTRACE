"""Portable Mamba-3 SISO/MIMO recurrences for differentiable atomistic models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import (
        mamba3_siso_combined as _mamba3_siso_fused,
    )
except (ImportError, OSError):
    _mamba3_siso_fused = None

try:
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import (
        mamba3_mimo as _mamba3_mimo_fused,
    )
except (ImportError, OSError):
    _mamba3_mimo_fused = None


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def _fused_siso_accepts_rotary_divisor() -> bool:
    """Report whether the installed fused SISO kernel takes the rotary divisor."""

    if _mamba3_siso_fused is None:
        return False
    try:
        import inspect

        return "rotary_dim_divisor" in inspect.signature(_mamba3_siso_fused).parameters
    except (TypeError, ValueError):
        return False


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """Positive data-dependent decay magnitude used by official Mamba-3."""
    negative = x.clamp_max(0.0)
    positive = x.clamp_min(0.0)
    return positive + torch.reciprocal(1.0 - negative)


def mamba3_angle_increments(
    raw_angles: torch.Tensor,
    mode: str = "official",
) -> torch.Tensor:
    """Return the per-step phase increments used by both Mamba-3 backends.

    Official Mamba-3 feeds the learned angle projection directly to the
    cumulative ``angle * dt`` rotation.  ``legacy_bounded`` preserves the
    bounded phase used by architecture-v4 MTACE checkpoints.
    """

    if mode == "official":
        return raw_angles
    if mode == "legacy_bounded":
        return math.pi * torch.tanh(raw_angles)
    raise ValueError("angle_mode must be 'official' or 'legacy_bounded'")


class Mamba3RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1.0e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        accumulation_dtype = _accumulation_dtype(x.dtype)
        x_acc = x.to(accumulation_dtype)
        inverse_rms = torch.rsqrt(
            x_acc.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return x * inverse_rms.to(x.dtype) * self.weight.to(x.dtype)


def rotate_state_pairs(
    vectors: torch.Tensor,
    cumulative_angles: torch.Tensor,
) -> torch.Tensor:
    """Apply Mamba-3's adjacent-pair data-dependent rotary transformation."""
    if cumulative_angles.shape[:-1] != vectors.shape[:-1]:
        raise ValueError(
            "angle and vector batch, length, and head dimensions must agree"
        )
    rotated_dimension = 2 * cumulative_angles.shape[-1]
    if rotated_dimension > vectors.shape[-1]:
        raise ValueError("rotary dimension exceeds state dimension")
    paired = vectors[..., :rotated_dimension].reshape(
        *vectors.shape[:-1], cumulative_angles.shape[-1], 2
    )
    cosine = torch.cos(cumulative_angles)
    sine = torch.sin(cumulative_angles)
    first = paired[..., 0] * cosine - paired[..., 1] * sine
    second = paired[..., 0] * sine + paired[..., 1] * cosine
    rotated = torch.stack((first, second), dim=-1).flatten(start_dim=-2)
    return torch.cat((rotated, vectors[..., rotated_dimension:]), dim=-1)


def rotate_state_halves(
    vectors: torch.Tensor,
    cumulative_angles: torch.Tensor,
) -> torch.Tensor:
    """Apply the blockwise pair layout used by the official MIMO kernel."""

    if cumulative_angles.shape[:-1] != vectors.shape[:-1]:
        raise ValueError("angle and vector leading dimensions must agree")
    if vectors.shape[-1] % 2 != 0:
        raise ValueError("blockwise rotary layout requires an even state dimension")
    half_dimension = vectors.shape[-1] // 2
    rotated_pairs = cumulative_angles.shape[-1]
    if rotated_pairs > half_dimension:
        raise ValueError("rotary dimension exceeds state dimension")
    first = vectors[..., :half_dimension]
    second = vectors[..., half_dimension:]
    cosine = torch.cos(cumulative_angles)
    sine = torch.sin(cumulative_angles)
    first_rotated = (
        first[..., :rotated_pairs] * cosine
        - second[..., :rotated_pairs] * sine
    )
    second_rotated = (
        first[..., :rotated_pairs] * sine
        + second[..., :rotated_pairs] * cosine
    )
    return torch.cat(
        (
            first_rotated,
            first[..., rotated_pairs:],
            second_rotated,
            second[..., rotated_pairs:],
        ),
        dim=-1,
    )


def rotate_state(
    vectors: torch.Tensor,
    cumulative_angles: torch.Tensor,
    layout: str,
) -> torch.Tensor:
    """Dispatch the requested rotary state layout.

    ``halves`` is the layout of the official MIMO kernel and the architecture-v9
    default for both SISO and MIMO, so a single portable code path serves every
    rank.  ``pairs`` is the adjacent-pair layout used by the architecture-v8 and
    earlier SISO path and is retained for exact checkpoint reproduction.  The two
    differ only by a fixed permutation of state coordinates, but that permutation
    is *not* a no-op for a given parameter vector, so mixing them silently
    changes the energy of a trained model.
    """

    if layout == "halves":
        return rotate_state_halves(vectors, cumulative_angles)
    if layout == "pairs":
        return rotate_state_pairs(vectors, cumulative_angles)
    raise ValueError("rotary_layout must be 'halves' or 'pairs'")


def _blocked_affine_scan(
    state: torch.Tensor,
    transition: torch.Tensor,
    chunk_size: int | None,
) -> torch.Tensor:
    """Inclusive scan of ``S_t = a_t S_{t-1} + d_t`` along dimension one.

    ``state`` holds the drives ``d_t`` and ``transition`` the scalars ``a_t``
    broadcast against the trailing state axes.

    With ``chunk_size=None`` this is a single Hillis-Steele pass: ``O(L log L)``
    work and ``O(log L)`` dependency depth.  With a finite ``chunk_size`` the
    pass is applied inside blocks and the block boundary state is carried
    forward, trading ``L / chunk_size`` sequential steps for shallower blocks;
    ``chunk_size=1`` is exactly the serial three-term recurrence and is useful as
    an in-model reference.  Both forms evaluate the same affine recurrence and
    both support the double backward required by force and stress training.

    Measured memory note.  Blocking was introduced expecting a large reduction in
    the tensors retained for reverse mode, on the assumption that the single pass
    keeps ``log2(L)`` copies of the ``(B, L, ..., P, N)`` drive tensor.  Direct
    peak-RSS measurement (``benchmarks/benchmark_scan_memory.py``) does not
    support that assumption: at ``B=64, L=128, R=4, d_inner=128, d_state=16`` the
    single pass peaks at 791 MiB and every blocked variant is equal or slightly
    worse, because concatenating the blocks reintroduces a full copy.  The
    default is therefore the single pass, and the honest statement for the
    manuscript is that the portable scan memory is set by the state tensor and is
    *not* reduced by blocking.  A genuine reduction needs a custom autograd
    function that recomputes the forward states in the backward pass.
    """

    length = state.shape[1]
    if chunk_size is None or chunk_size >= length:
        return _hillis_steele_scan(state, transition)
    chunk_size = max(1, int(chunk_size))
    blocks = []
    carry = None
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        block_state = state[:, start:stop]
        block_transition = transition[:, start:stop]
        if carry is not None:
            # Injecting a_start * S_{start-1} into the first drive of the block
            # makes the in-block scan reproduce the global recurrence exactly.
            leading = block_state[:, :1] + block_transition[:, :1] * carry
            block_state = torch.cat((leading, block_state[:, 1:]), dim=1)
        block_state = _hillis_steele_scan(block_state, block_transition)
        blocks.append(block_state)
        carry = block_state[:, -1:]
    return torch.cat(blocks, dim=1)


def affine_shell_scan(
    drive: torch.Tensor, transition: torch.Tensor
) -> torch.Tensor:
    """Inclusive scan of ``S_k = a_k S_{k-1} + d_k`` over dimension one.

    ``drive`` is ``(atoms, shells, features)`` and ``transition`` broadcasts
    against it, typically ``(atoms, shells, 1)``.  This is the same audited
    associative pass the state-space mixer uses, exposed for the equivariant
    shell recurrences in :mod:`mtace.ssm`, which need a scan over an
    equivariant feature axis rather than over head/state axes.
    """

    if drive.ndim != 3:
        raise ValueError("drive must have shape (atoms, shells, features)")
    if transition.shape[:2] != drive.shape[:2]:
        raise ValueError("transition must share the atom and shell axes")
    return _hillis_steele_scan(drive, transition.expand_as(drive))


def _hillis_steele_scan(
    state: torch.Tensor, transition: torch.Tensor
) -> torch.Tensor:
    offset = 1
    length = state.shape[1]
    while offset < length:
        state = torch.cat(
            (
                state[:, :offset],
                state[:, offset:] + transition[:, offset:] * state[:, :-offset],
            ),
            dim=1,
        )
        transition = torch.cat(
            (
                transition[:, :offset],
                transition[:, offset:] * transition[:, :-offset],
            ),
            dim=1,
        )
        offset *= 2
    return state


def _validate_scan_shapes(
    x: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> None:
    if x.ndim != 4 or k.ndim != 4 or q.shape != k.shape:
        raise ValueError("x must be (B,L,H,P), and k and q must be (B,L,H,N)")
    if x.shape[1] < 1:
        raise ValueError("the sequence length must be positive")
    if x.shape[:3] != k.shape[:3]:
        raise ValueError("x, k, and q batch, length, and head dimensions must agree")
    expected = x.shape[:3]
    if alpha.shape != expected or beta.shape != expected or gamma.shape != expected:
        raise ValueError("alpha, beta, and gamma must have shape (B,L,H)")


def _validate_skip_and_gate(
    x: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
) -> None:
    if D is not None and D.shape != (x.shape[2],):
        raise ValueError("D must have shape (H,)")
    if z is not None and z.shape != x.shape:
        raise ValueError("z must have the same shape as x")


def _mamba3_drive(
    x: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    current = x.unsqueeze(-1) * k.unsqueeze(-2)
    previous = torch.cat((torch.zeros_like(current[:, :1]), current[:, :-1]), dim=1)
    return gamma[..., None, None] * current + beta[..., None, None] * previous


def mamba3_scan_reference(
    x: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Serial exponential-trapezoidal SISO recurrence from Mamba-3."""
    _validate_scan_shapes(x, k, q, alpha, beta, gamma)
    _validate_skip_and_gate(x, D, z)
    input_dtype = x.dtype
    accumulation_dtype = _accumulation_dtype(input_dtype)
    x_acc = x.to(accumulation_dtype)
    k_acc = k.to(accumulation_dtype)
    q_acc = q.to(accumulation_dtype)
    alpha_acc = alpha.to(accumulation_dtype)
    beta_acc = beta.to(accumulation_dtype)
    gamma_acc = gamma.to(accumulation_dtype)
    drive = _mamba3_drive(x_acc, k_acc, beta_acc, gamma_acc)
    state = x_acc.new_zeros(
        (x.shape[0], x.shape[2], x.shape[3], k.shape[3])
    )
    outputs = []
    for step in range(x.shape[1]):
        state = alpha_acc[:, step, :, None, None] * state + drive[:, step]
        outputs.append(torch.einsum("bhpn,bhn->bhp", state, q_acc[:, step]))
    y = torch.stack(outputs, dim=1)
    if D is not None:
        y = y + D.to(accumulation_dtype)[None, None, :, None] * x_acc
    if z is not None:
        y = y * F.silu(z.to(accumulation_dtype))
    return y.to(input_dtype)


def mamba3_scan_parallel(
    x: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Associative scan equivalent to the Mamba-3 serial recurrence."""
    _validate_scan_shapes(x, k, q, alpha, beta, gamma)
    _validate_skip_and_gate(x, D, z)
    input_dtype = x.dtype
    accumulation_dtype = _accumulation_dtype(input_dtype)
    x_acc = x.to(accumulation_dtype)
    k_acc = k.to(accumulation_dtype)
    q_acc = q.to(accumulation_dtype)
    transition = alpha.to(accumulation_dtype)[..., None, None]
    state = _mamba3_drive(
        x_acc,
        k_acc,
        beta.to(accumulation_dtype),
        gamma.to(accumulation_dtype),
    )
    state = _blocked_affine_scan(state, transition, chunk_size)

    y = torch.einsum("blhpn,blhn->blhp", state, q_acc)
    if D is not None:
        y = y + D.to(accumulation_dtype)[None, None, :, None] * x_acc
    if z is not None:
        y = y * F.silu(z.to(accumulation_dtype))
    return y.to(input_dtype)


def _validate_mimo_scan_shapes(
    x: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> None:
    if x.ndim != 5 or k.ndim != 5 or q.shape != k.shape:
        raise ValueError(
            "x must be (B,L,R,H,P), and k and q must be (B,L,R,H,N)"
        )
    if x.shape[1] < 1:
        raise ValueError("the sequence length must be positive")
    if x.shape[:4] != k.shape[:4]:
        raise ValueError("x, k, and q batch, length, rank, and head axes must agree")
    expected = (x.shape[0], x.shape[1], x.shape[3])
    if alpha.shape != expected or beta.shape != expected or gamma.shape != expected:
        raise ValueError("alpha, beta, and gamma must have shape (B,L,H)")


def _mamba3_mimo_drive(
    x: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    current = torch.einsum("blrhp,blrhn->blhpn", x, k)
    previous = torch.cat((torch.zeros_like(current[:, :1]), current[:, :-1]), dim=1)
    return gamma[..., None, None] * current + beta[..., None, None] * previous


def _validate_mimo_skip_and_gate(
    x: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
) -> None:
    if D is not None and D.shape != (x.shape[3],):
        raise ValueError("D must have shape (H,)")
    if z is not None and z.shape != x.shape:
        raise ValueError("z must have the same shape as x")


def mamba3_mimo_scan_reference(
    x: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Serial rank-R MIMO recurrence with output shape ``(B,L,R,H,P)``."""

    _validate_mimo_scan_shapes(x, k, q, alpha, beta, gamma)
    _validate_mimo_skip_and_gate(x, D, z)
    input_dtype = x.dtype
    accumulation_dtype = _accumulation_dtype(input_dtype)
    x_acc = x.to(accumulation_dtype)
    q_acc = q.to(accumulation_dtype)
    drive = _mamba3_mimo_drive(
        x_acc,
        k.to(accumulation_dtype),
        beta.to(accumulation_dtype),
        gamma.to(accumulation_dtype),
    )
    state = x_acc.new_zeros((x.shape[0], x.shape[3], x.shape[4], k.shape[4]))
    alpha_acc = alpha.to(accumulation_dtype)
    outputs = []
    for step in range(x.shape[1]):
        state = alpha_acc[:, step, :, None, None] * state + drive[:, step]
        outputs.append(torch.einsum("bhpn,brhn->brhp", state, q_acc[:, step]))
    y = torch.stack(outputs, dim=1)
    if D is not None:
        y = y + D.to(accumulation_dtype)[None, None, None, :, None] * x_acc
    if z is not None:
        y = y * F.silu(z.to(accumulation_dtype))
    return y.to(input_dtype)


def mamba3_mimo_scan_parallel(
    x: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Associative scan equivalent to ``mamba3_mimo_scan_reference``."""

    _validate_mimo_scan_shapes(x, k, q, alpha, beta, gamma)
    _validate_mimo_skip_and_gate(x, D, z)
    input_dtype = x.dtype
    accumulation_dtype = _accumulation_dtype(input_dtype)
    x_acc = x.to(accumulation_dtype)
    transition = alpha.to(accumulation_dtype)[..., None, None]
    state = _mamba3_mimo_drive(
        x_acc,
        k.to(accumulation_dtype),
        beta.to(accumulation_dtype),
        gamma.to(accumulation_dtype),
    )
    state = _blocked_affine_scan(state, transition, chunk_size)
    y = torch.einsum("blhpn,blrhn->blrhp", state, q.to(accumulation_dtype))
    if D is not None:
        y = y + D.to(accumulation_dtype)[None, None, None, :, None] * x_acc
    if z is not None:
        y = y * F.silu(z.to(accumulation_dtype))
    return y.to(input_dtype)


class Mamba3Direction(nn.Module):
    """One causal Mamba-3 SISO or MIMO direction."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        headdim: int = 16,
        rope_fraction: float = 0.5,
        a_floor: float = 1.0e-4,
        chunk_size: int = 64,
        angle_mode: str = "official",
        mimo_rank: int = 1,
        rotary_layout: str = "halves",
        scan_mode: str = "auto",
        backend: str = "auto",
    ):
        super().__init__()
        if d_model < 1 or d_state < 2 or expand < 1 or headdim < 1:
            raise ValueError("Mamba-3 dimensions must be positive and d_state at least two")
        if d_state % 2 != 0:
            raise ValueError("Mamba-3 d_state must be even for adjacent complex pairs")
        if rope_fraction not in {0.5, 1.0}:
            raise ValueError("rope_fraction must be 0.5 or 1.0")
        if a_floor <= 0.0 or chunk_size < 1 or mimo_rank < 1:
            raise ValueError("a_floor, chunk_size, and mimo_rank must be positive")
        if backend not in {"auto", "torch", "cuda"}:
            raise ValueError("backend must be 'auto', 'torch', or 'cuda'")
        if angle_mode not in {"official", "legacy_bounded"}:
            raise ValueError("angle_mode must be 'official' or 'legacy_bounded'")
        rotary_layout = str(rotary_layout).lower()
        if rotary_layout not in {"halves", "pairs"}:
            raise ValueError("rotary_layout must be 'halves' or 'pairs'")
        scan_mode = str(scan_mode).lower()
        if scan_mode not in {"auto", "parallel", "chunked"}:
            raise ValueError("scan_mode must be 'auto', 'parallel', or 'chunked'")
        self.rotary_layout = rotary_layout
        self.scan_mode = scan_mode
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(expand * d_model)
        self.headdim = int(headdim)
        if self.d_inner % self.headdim != 0:
            raise ValueError("expand * d_model must be divisible by headdim")
        self.nheads = self.d_inner // self.headdim
        self.mimo_rank = int(mimo_rank)
        split_dimension = int(self.d_state * rope_fraction)
        split_dimension -= split_dimension % 2
        if split_dimension < 2:
            raise ValueError("rope_fraction leaves no complex state pair")
        self.num_rope_angles = split_dimension // 2
        self.rotary_dim_divisor = int(2 / rope_fraction)
        self.a_floor = float(a_floor)
        self.chunk_size = int(chunk_size)
        self.angle_mode = angle_mode
        self.backend = backend

        projection_dimension = (
            2 * self.d_inner
            + 2 * self.d_state * self.mimo_rank
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.d_model, projection_dimension, bias=False)
        self.B_norm = Mamba3RMSNorm(self.d_state, eps=1.0e-5)
        self.C_norm = Mamba3RMSNorm(self.d_state, eps=1.0e-5)
        bias_shape = (
            (self.nheads, self.d_state)
            if self.mimo_rank == 1
            else (self.nheads, self.mimo_rank, self.d_state)
        )
        self.B_bias = nn.Parameter(torch.ones(bias_shape))
        self.C_bias = nn.Parameter(torch.ones(bias_shape))
        if self.mimo_rank > 1:
            self.mimo_x = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim)
                / float(self.mimo_rank)
            )
            self.mimo_z = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim)
            )
            self.mimo_o = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim)
                / float(self.mimo_rank)
            )
        self.dt_bias = nn.Parameter(self._initial_dt_bias(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.dt_bias._no_weight_decay = True
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    @staticmethod
    def _initial_dt_bias(nheads: int) -> torch.Tensor:
        dt_min, dt_max = 1.0e-3, 1.0e-1
        dt = torch.exp(
            torch.rand(nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp_min(1.0e-4)
        return dt + torch.log(-torch.expm1(-dt))

    @property
    def accelerated_backend_available(self) -> bool:
        if self.mimo_rank > 1:
            return _mamba3_mimo_fused is not None
        return _mamba3_siso_fused is not None

    @property
    def fused_configuration_error(self) -> str | None:
        if (self.headdim & (self.headdim - 1)) != 0:
            return "headdim must be a power of two"
        if (self.d_state & (self.d_state - 1)) != 0:
            return "d_state must be a power of two"
        if self.mimo_rank > 1 and self.chunk_size < 8:
            return "the official fused MIMO kernel requires chunk_size >= 8"
        if self.rotary_layout != "halves":
            # The fused kernels use the blockwise ("halves") rotary layout.  The
            # adjacent-pair layout differs by a permutation of state coordinates,
            # which changes the energy predicted by a *given* parameter vector.
            # Refusing to dispatch is the only safe behavior; see
            # tests/test_mamba3.py::test_portable_and_fused_agree.
            return (
                "the fused kernels assume rotary_layout='halves'; the "
                "adjacent-pair layout would permute the learned state coordinates"
            )
        return None

    def _project(self, hidden: torch.Tensor):
        projected = self.in_proj(hidden)
        return torch.split(
            projected,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.mimo_rank,
                self.d_state * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )

    def _resolved_scan_chunk(self, length: int) -> int | None:
        """Select the portable-scan schedule.

        ``auto`` resolves to the single Hillis-Steele pass at every length.
        Blocking measurably does not reduce peak memory (see
        ``_blocked_affine_scan``) and costs latency, so it is opt-in only.
        """

        if self.scan_mode == "chunked":
            return max(1, min(int(self.chunk_size), length))
        return None

    def forward(
        self,
        hidden: torch.Tensor,
        require_higher_order: bool = False,
        step_scale: float | torch.Tensor = 1.0,
        screening: torch.Tensor | None = None,
    ) -> torch.Tensor:
        configuration_error = self.fused_configuration_error
        fused_candidate = (
            hidden.is_cuda
            and hidden.dtype in {torch.float32, torch.bfloat16}
            and self.accelerated_backend_available
            and configuration_error is None
            and self.backend != "torch"
            and not require_higher_order
            # The fused kernels take ADT and DT separately and assume the
            # standard free-decay parameterisation; a screened ADT would be
            # silently inconsistent with what they recompute internally.
            and screening is None
        )
        if self.backend == "cuda" and not fused_candidate:
            if require_higher_order:
                raise RuntimeError(
                    "Mamba-3 fused CUDA does not provide the guaranteed double backward "
                    "required for force/stress training"
                )
            if configuration_error is not None:
                raise RuntimeError(
                    f"Unsupported fused Mamba-3 configuration: {configuration_error}"
                )
            if not hidden.is_cuda:
                raise RuntimeError("Mamba-3 backend='cuda' requires CUDA input tensors")
            if hidden.dtype not in {torch.float32, torch.bfloat16}:
                raise RuntimeError(
                    "Mamba-3 backend='cuda' requires FP32 or BF16 model activations"
                )
            raise RuntimeError(
                "Mamba-3 backend='cuda' requires the official fused SISO/MIMO kernels"
            )

        if fused_candidate:
            # Keep ACE/readout parameters in their trained precision. Autocast only
            # the Mamba projections required by the official BF16 fused kernels.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                projected = self._project(hidden)
        else:
            projected = self._project(hidden)
        z, x, B, C, raw_dt, raw_A, trap, raw_angles = projected
        batch, length = hidden.shape[:2]
        x = x.reshape(batch, length, self.nheads, self.headdim)
        z = z.reshape(batch, length, self.nheads, self.headdim)
        if self.mimo_rank > 1:
            B = B.reshape(batch, length, self.mimo_rank, self.d_state)
            C = C.reshape(batch, length, self.mimo_rank, self.d_state)
        B = self.B_norm(B)
        C = self.C_norm(C)

        accumulation_dtype = _accumulation_dtype(hidden.dtype)
        dt = F.softplus(
            raw_dt.to(accumulation_dtype) + self.dt_bias.to(accumulation_dtype)
        )
        # Radial metric.  The learned softplus output is an intensive rate; the
        # discrete step of the recurrence is that rate times the physical shell
        # increment.  ``step_scale`` equals one on the mesh the model was built
        # with, so this is a no-op there, and it makes the recurrence approximate
        # the same radial transfer operator when the mesh is refined.
        if not (isinstance(step_scale, float) and step_scale == 1.0):
            scale = (
                step_scale
                if torch.is_tensor(step_scale)
                else torch.as_tensor(step_scale, dtype=accumulation_dtype)
            )
            dt = dt * scale.to(device=dt.device, dtype=accumulation_dtype)
        if screening is None:
            decay = -heavy_tail_activation(raw_A.to(accumulation_dtype)).clamp_min(
                self.a_floor
            )
            adt = decay * dt
        else:
            # Physically constrained decay.  Instead of a free per-head rate, the
            # memory is a screened (Yukawa) radial kernel with a single per-atom
            # length:
            #
            #     alpha_k = exp(-dr / lambda_i),   dr = shell spacing in Angstrom
            #
            # so unrolling gives K(k,k') ~ exp(-|r_k - r_k'| / lambda_i), exactly
            # the Thomas-Fermi/Yukawa form.  lambda_i is then a predicted physical
            # observable in Angstrom rather than an opaque parameter, and it is
            # directly comparable with a screening length or the first minimum of
            # the radial distribution function.
            #
            # Only the *memory* is constrained.  dt still carries the learned
            # input strength and still drives the rotary phase, so the positional
            # mechanism is untouched; alpha and dt are separate physics.
            adt = -screening.to(accumulation_dtype)
            adt = adt.expand(dt.shape[0], dt.shape[1], dt.shape[2])
        angles = mamba3_angle_increments(
            raw_angles.to(accumulation_dtype), self.angle_mode
        )
        angles = angles[:, :, None, :].expand(-1, -1, self.nheads, -1)

        use_fused = fused_candidate
        if use_fused and x.dtype != torch.bfloat16:
            raise RuntimeError("CUDA autocast did not produce BF16 Mamba-3 projections")

        if use_fused and self.mimo_rank > 1:
            y = _mamba3_mimo_fused(
                Q=C[:, :, :, None, :],
                K=B[:, :, :, None, :],
                V=x,
                ADT=adt.transpose(1, 2).contiguous(),
                DT=dt.transpose(1, 2).contiguous(),
                Trap=trap.transpose(1, 2).contiguous(),
                Q_bias=self.C_bias,
                K_bias=self.B_bias,
                MIMO_V=self.mimo_x,
                MIMO_Z=self.mimo_z,
                MIMO_Out=self.mimo_o,
                Angles=angles.to(torch.float32),
                D=self.D,
                Z=z,
                chunk_size=self.chunk_size,
                rotary_dim_divisor=self.rotary_dim_divisor,
                dtype=x.dtype,
            )
        elif use_fused:
            siso_arguments = dict(
                Q=C[:, :, None, :],
                K=B[:, :, None, :],
                V=x,
                ADT=adt.transpose(1, 2).contiguous(),
                DT=dt.transpose(1, 2).contiguous(),
                Trap=trap.transpose(1, 2).contiguous(),
                Q_bias=self.C_bias,
                K_bias=self.B_bias,
                Angles=angles.to(torch.float32),
                D=self.D,
                Z=z,
                chunk_size=self.chunk_size,
            )
            # The rotary layout of the fused kernel must match the portable path.
            # Pass the divisor explicitly whenever the installed kernel accepts
            # it instead of relying on its default.
            if _fused_siso_accepts_rotary_divisor():
                siso_arguments["rotary_dim_divisor"] = self.rotary_dim_divisor
            y = _mamba3_siso_fused(**siso_arguments)
        else:
            cumulative_angles = torch.cumsum(angles * dt[..., None], dim=1)
            alpha = torch.exp(adt)
            mixing = torch.sigmoid(trap.to(accumulation_dtype))
            beta = (1.0 - mixing) * dt * alpha
            gamma = mixing * dt
            scan_chunk = self._resolved_scan_chunk(length)
            if self.mimo_rank > 1:
                B_heads = B[:, :, :, None, :].expand(
                    -1, -1, -1, self.nheads, -1
                )
                C_heads = C[:, :, :, None, :].expand(
                    -1, -1, -1, self.nheads, -1
                )
                bias_order = (1, 0, 2)
                B_heads = B_heads + self.B_bias.permute(bias_order)[None, None]
                C_heads = C_heads + self.C_bias.permute(bias_order)[None, None]
                rank_angles = cumulative_angles[:, :, None, :, :].expand(
                    -1, -1, self.mimo_rank, -1, -1
                )
                k = rotate_state(
                    B_heads.to(accumulation_dtype), rank_angles, self.rotary_layout
                )
                q = rotate_state(
                    C_heads.to(accumulation_dtype), rank_angles, self.rotary_layout
                )
                projection_order = (1, 0, 2)
                x_rank = x[:, :, None, :, :] * self.mimo_x.permute(
                    projection_order
                )[None, None]
                z_rank = z[:, :, None, :, :] * self.mimo_z.permute(
                    projection_order
                )[None, None]
                y_rank = mamba3_mimo_scan_parallel(
                    x_rank,
                    k,
                    q,
                    alpha,
                    beta,
                    gamma,
                    self.D,
                    z_rank,
                    chunk_size=scan_chunk,
                )
                y = torch.einsum("blrhp,hrp->blhp", y_rank, self.mimo_o)
            else:
                B_heads = B[:, :, None, :] + self.B_bias[None, None, :, :]
                C_heads = C[:, :, None, :] + self.C_bias[None, None, :, :]
                k = rotate_state(
                    B_heads.to(accumulation_dtype),
                    cumulative_angles,
                    self.rotary_layout,
                )
                q = rotate_state(
                    C_heads.to(accumulation_dtype),
                    cumulative_angles,
                    self.rotary_layout,
                )
                y = mamba3_scan_parallel(
                    x, k, q, alpha, beta, gamma, self.D, z, chunk_size=scan_chunk
                )

        return self.out_proj(y.reshape(batch, length, self.d_inner).to(hidden.dtype))


class Mamba3SequenceMixer(nn.Module):
    """Noncausal bidirectional Mamba-3 SISO/MIMO scientific mixer."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        headdim: int = 16,
        rope_fraction: float = 0.5,
        a_floor: float = 1.0e-4,
        chunk_size: int = 64,
        angle_mode: str = "official",
        mimo_rank: int = 1,
        rotary_layout: str = "halves",
        scan_mode: str = "auto",
        bidirectional_tied: bool = False,
        backend: str = "auto",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.bidirectional_tied = bool(bidirectional_tied)
        self.norm = Mamba3RMSNorm(d_model, eps=1.0e-6)
        direction_kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            rope_fraction=rope_fraction,
            a_floor=a_floor,
            chunk_size=chunk_size,
            angle_mode=angle_mode,
            mimo_rank=mimo_rank,
            rotary_layout=rotary_layout,
            scan_mode=scan_mode,
            backend=backend,
        )
        self.forward_direction = Mamba3Direction(**direction_kwargs)
        self.backward_direction = (
            None if self.bidirectional_tied else Mamba3Direction(**direction_kwargs)
        )

    @property
    def accelerated_backend_available(self) -> bool:
        return self.forward_direction.accelerated_backend_available

    def forward(
        self,
        hidden: torch.Tensor,
        require_higher_order: bool = False,
        step_scale: float | torch.Tensor = 1.0,
        screening: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError(f"hidden must have shape (batch, length, {self.d_model})")
        normalized = self.norm(hidden)
        forward = self.forward_direction(
            normalized, require_higher_order, step_scale, screening
        )
        if self.bidirectional_tied:
            reverse_direction = self.forward_direction
        else:
            if self.backward_direction is None:
                raise RuntimeError("untied bidirectional mixer has no backward direction")
            reverse_direction = self.backward_direction
        backward = torch.flip(
            reverse_direction(
                torch.flip(normalized, dims=(1,)), require_higher_order, step_scale
            ),
            dims=(1,),
        )
        scale = 0.5 if self.bidirectional_tied else math.sqrt(0.5)
        return hidden + scale * (forward + backward)
