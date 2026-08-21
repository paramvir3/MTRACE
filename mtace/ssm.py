"""Portable Mamba selective scan and symmetry-preserving ACE coupling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

from .mamba3 import Mamba3SequenceMixer, affine_shell_scan
from .mixers import (
    AttentionSequenceMixer,
    DeepSetsSequenceMixer,
    DenseRadialSequenceMixer,
    IdentitySequenceMixer,
)
from .routing import RoutedScalarFFN

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_cuda as _selective_scan_cuda,
        selective_scan_fn as _selective_scan_cuda_fn,
    )
except (ImportError, OSError):
    _selective_scan_cuda = None
    _selective_scan_cuda_fn = None


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulate low precision scans in FP32 without demoting FP64 models."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """Differentiable PyTorch reference for the real-valued Mamba-1 scan.

    Shapes follow the official implementation: u and delta are (B,D,L), A is
    (D,N), and input-dependent B and C are (B,N,L).
    """
    if u.ndim != 3 or delta.shape != u.shape:
        raise ValueError("u and delta must both have shape (batch, channels, length)")
    if A.ndim != 2 or A.shape[0] != u.shape[1]:
        raise ValueError("A must have shape (channels, state_dim)")
    expected_bc = (u.shape[0], A.shape[1], u.shape[2])
    if B.shape != expected_bc or C.shape != expected_bc:
        raise ValueError(f"B and C must have shape {expected_bc}")

    input_dtype = u.dtype
    accumulation_dtype = _accumulation_dtype(input_dtype)
    u_acc = u.to(accumulation_dtype)
    delta_acc = delta.to(accumulation_dtype)
    if delta_bias is not None:
        delta_acc = delta_acc + delta_bias.to(accumulation_dtype)[None, :, None]
    if delta_softplus:
        delta_acc = F.softplus(delta_acc)
    A_acc = A.to(accumulation_dtype)
    B_acc = B.to(accumulation_dtype)
    C_acc = C.to(accumulation_dtype)
    state = u_acc.new_zeros((u.shape[0], u.shape[1], A.shape[1]))
    outputs = []
    for step in range(u.shape[2]):
        dt = delta_acc[:, :, step]
        transition = torch.exp(dt[:, :, None] * A_acc[None, :, :])
        drive = dt[:, :, None] * B_acc[:, None, :, step] * u_acc[:, :, step, None]
        state = transition * state + drive
        output = torch.einsum("bdn,bn->bd", state, C_acc[:, :, step])
        outputs.append(output)
    y = torch.stack(outputs, dim=-1)
    if D is not None:
        y = y + D.to(accumulation_dtype)[None, :, None] * u_acc
    if z is not None:
        y = y * F.silu(z.to(accumulation_dtype))
    return y.to(input_dtype)


def selective_scan_parallel(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """Associative PyTorch scan equivalent to ``selective_scan_reference``."""
    if u.ndim != 3 or delta.shape != u.shape:
        raise ValueError("u and delta must both have shape (batch, channels, length)")
    expected_bc = (u.shape[0], A.shape[1], u.shape[2])
    if A.ndim != 2 or A.shape[0] != u.shape[1] or B.shape != expected_bc or C.shape != expected_bc:
        raise ValueError("Incompatible selective-scan tensor shapes")
    input_dtype = u.dtype
    accumulation_dtype = _accumulation_dtype(input_dtype)
    u_acc = u.to(accumulation_dtype)
    delta_acc = delta.to(accumulation_dtype)
    if delta_bias is not None:
        delta_acc = delta_acc + delta_bias.to(accumulation_dtype)[None, :, None]
    if delta_softplus:
        delta_acc = F.softplus(delta_acc)
    transition = torch.exp(
        torch.einsum("bdl,dn->bdln", delta_acc, A.to(accumulation_dtype))
    )
    state = torch.einsum(
        "bdl,bnl,bdl->bdln", delta_acc, B.to(accumulation_dtype), u_acc
    )

    offset = 1
    while offset < u.shape[2]:
        state = torch.cat(
            (
                state[:, :, :offset],
                state[:, :, offset:] + transition[:, :, offset:] * state[:, :, :-offset],
            ),
            dim=2,
        )
        transition = torch.cat(
            (
                transition[:, :, :offset],
                transition[:, :, offset:] * transition[:, :, :-offset],
            ),
            dim=2,
        )
        offset *= 2

    y = torch.einsum("bdln,bnl->bdl", state, C.to(accumulation_dtype))
    if D is not None:
        y = y + D.to(accumulation_dtype)[None, :, None] * u_acc
    if z is not None:
        y = y * F.silu(z.to(accumulation_dtype))
    return y.to(input_dtype)


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        accumulation_dtype = _accumulation_dtype(x.dtype)
        x_acc = x.to(accumulation_dtype)
        scale = torch.rsqrt(x_acc.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(x.dtype)) * self.weight.to(x.dtype)


class MambaSequenceMixer(nn.Module):
    """Bidirectional Mamba-1 block over a noncausal scientific token axis."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: int | None = None,
        bidirectional_tied: bool = False,
        backend: str = "auto",
    ):
        super().__init__()
        if d_model < 1 or d_state < 1 or d_conv < 1 or expand < 1:
            raise ValueError("Mamba dimensions must be positive")
        if backend not in {"auto", "torch", "cuda"}:
            raise ValueError("backend must be 'auto', 'torch', or 'cuda'")
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.d_inner = int(expand * d_model)
        if dt_rank is not None and dt_rank < 1:
            raise ValueError("dt_rank must be positive when specified")
        self.dt_rank = int(dt_rank if dt_rank is not None else math.ceil(d_model / 16))
        self.bidirectional_tied = bool(bidirectional_tied)
        self.backend = backend

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)[None, :]
        self.A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self._initialize_dt(self.dt_proj)

        if not self.bidirectional_tied:
            self.backward_in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
            self.backward_conv1d = nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                bias=True,
            )
            self.backward_x_proj = nn.Linear(
                self.d_inner, self.dt_rank + 2 * d_state, bias=False
            )
            self.backward_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            self.backward_out_proj = nn.Linear(self.d_inner, d_model, bias=False)
            self.backward_A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))
            self.backward_D = nn.Parameter(torch.ones(self.d_inner))
            self.backward_A_log._no_weight_decay = True
            self.backward_D._no_weight_decay = True
            self._initialize_dt(self.backward_dt_proj)

    def _initialize_dt(self, projection: nn.Linear) -> None:
        dt_min, dt_max = 1.0e-3, 1.0e-1
        bound = self.dt_rank**-0.5
        nn.init.uniform_(projection.weight, -bound, bound)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp_min(1.0e-4)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            projection.bias.copy_(inverse_softplus)
        projection.bias._no_reinit = True

    @property
    def accelerated_backend_available(self) -> bool:
        return _selective_scan_cuda_fn is not None and _selective_scan_cuda is not None

    def _scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        z: torch.Tensor,
        D: torch.Tensor,
        delta_bias: torch.Tensor,
        require_higher_order: bool,
    ) -> torch.Tensor:
        supported_cuda_dtype = u.dtype in {torch.float16, torch.bfloat16, torch.float32}
        use_cuda = (
            u.is_cuda
            and supported_cuda_dtype
            and self.accelerated_backend_available
            and self.backend != "torch"
            and not require_higher_order
        )
        if self.backend == "cuda" and not use_cuda:
            if require_higher_order:
                raise RuntimeError(
                    "backend='cuda' does not provide the guaranteed double backward required "
                    "for force/stress training; use backend='auto' or 'torch'"
                )
            raise RuntimeError(
                "backend='cuda' requires a FP16, BF16, or FP32 CUDA tensor and "
                "mamba-ssm built with selective_scan_cuda"
            )
        if use_cuda:
            return _selective_scan_cuda_fn(
                u,
                delta,
                A,
                B,
                C,
                D.float(),
                z=z,
                delta_bias=delta_bias.float(),
                delta_softplus=True,
            )
        return selective_scan_parallel(
            u,
            delta,
            A,
            B,
            C,
            D=D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=True,
        )

    def _direction(
        self,
        hidden: torch.Tensor,
        backward: bool = False,
        require_higher_order: bool = False,
    ) -> torch.Tensor:
        if backward and not self.bidirectional_tied:
            in_proj = self.backward_in_proj
            conv1d = self.backward_conv1d
            x_proj = self.backward_x_proj
            dt_proj = self.backward_dt_proj
            out_proj = self.backward_out_proj
            A_log = self.backward_A_log
            D = self.backward_D
        else:
            in_proj = self.in_proj
            conv1d = self.conv1d
            x_proj = self.x_proj
            dt_proj = self.dt_proj
            out_proj = self.out_proj
            A_log = self.A_log
            D = self.D
        sequence_length = hidden.shape[1]
        xz = in_proj(hidden).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        x = F.silu(conv1d(x)[..., :sequence_length])
        projected = x_proj(x.transpose(1, 2))
        dt_low_rank, B, C = torch.split(
            projected, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.linear(dt_low_rank, dt_proj.weight).transpose(1, 2)
        parameter_dtype = _accumulation_dtype(hidden.dtype)
        A = -torch.exp(A_log.to(parameter_dtype))
        y = self._scan(
            x,
            delta,
            A,
            B.transpose(1, 2).contiguous(),
            C.transpose(1, 2).contiguous(),
            z,
            D,
            dt_proj.bias,
            require_higher_order,
        )
        return out_proj(y.transpose(1, 2))

    def forward(
        self,
        hidden: torch.Tensor,
        require_higher_order: bool = False,
        step_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        del step_scale  # the Mamba-1 baseline has no radial-metric discretization
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError(f"hidden must have shape (batch, length, {self.d_model})")
        normalized = self.norm(hidden)
        forward = self._direction(normalized, require_higher_order=require_higher_order)
        backward = torch.flip(
            self._direction(
                torch.flip(normalized, dims=(1,)),
                backward=True,
                require_higher_order=require_higher_order,
            ),
            dims=(1,),
        )
        direction_scale = 0.5 if self.bidirectional_tied else math.sqrt(0.5)
        return hidden + direction_scale * (forward + backward)


def _irrep_blocks(irreps: o3.Irreps) -> tuple[tuple[int, int], ...]:
    return tuple((int(mul), int(ir.dim)) for mul, ir in irreps)


def _expand_irrep_scalars(
    scalars: torch.Tensor, blocks: tuple[tuple[int, int], ...]
) -> torch.Tensor:
    pieces = []
    offset = 0
    for multiplicity, irrep_dim in blocks:
        part = scalars[..., offset : offset + multiplicity]
        pieces.append(part.repeat_interleave(irrep_dim, dim=-1))
        offset += multiplicity
    return torch.cat(pieces, dim=-1)


class IrrepInvariantNorm(nn.Module):
    """Rotation-invariant magnitudes of every irrep copy.

    ``squared`` returns ``sum_m |x_{c l p m}|^2``, the architecture-v8 behavior.

    ``homogeneous`` returns the smoothed magnitude

        n = sqrt(sum_m |x|^2 + eps^2) - eps ,

    which is what architecture v10 uses by default.  The motivation is
    conditioning rather than expressivity.  The invariant vector consumed by the
    mixer concatenates even scalars, which are homogeneous of degree one in the
    ACE features, with irrep norms.  Squaring makes the second block degree two,
    so a global rescaling of the features -- a different ``avg_num_neighbors``, a
    retuned radial network, or fine-tuning on a new dataset -- moves the two
    blocks by different powers and silently changes the learned balance between
    scalar and angular information.  A single linear map cannot undo that.  The
    smoothed magnitude restores degree-one homogeneity for every block.

    The smoothing is required, not cosmetic: ``|x|`` has unbounded curvature at
    the origin and force training differentiates this expression twice.  With the
    form above the gradient is bounded by one and the curvature by ``1 / eps``,
    while ``n(0) = 0`` keeps an empty shell exactly invisible.  The relative
    departure from exact homogeneity is ``eps / |x|``.
    """

    def __init__(self, irreps, mode: str = "squared", eps: float = 1.0e-4):
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"squared", "homogeneous"}:
            raise ValueError("invariant_norm must be 'squared' or 'homogeneous'")
        eps = float(eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("invariant_norm_eps must be positive and finite")
        self.mode = mode
        self.eps = eps
        self.norm = o3.Norm(o3.Irreps(irreps), squared=True)
        self.irreps_out = self.norm.irreps_out

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        squared = self.norm(features)
        if self.mode == "squared":
            return squared
        return torch.sqrt(squared + self.eps * self.eps) - self.eps


class IrrepDropout(nn.Module):
    """Drop complete irrep copies while tying the mask over magnetic components."""

    def __init__(self, irreps, probability: float = 0.0):
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("dropout probability must lie in [0, 1)")
        self.blocks = _irrep_blocks(o3.Irreps(irreps))
        self.num_copies = sum(multiplicity for multiplicity, _ in self.blocks)
        self.probability = float(probability)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return features
        keep_probability = 1.0 - self.probability
        mask = features.new_empty((*features.shape[:-1], self.num_copies))
        mask.bernoulli_(keep_probability).div_(keep_probability)
        return features * _expand_irrep_scalars(mask, self.blocks)


def _reduced_irreps(irreps: o3.Irreps, channels: int) -> o3.Irreps:
    """Cap every multiplicity at ``channels``, keeping the angular structure."""

    blocks = [(min(int(mul), int(channels)), ir) for mul, ir in irreps if int(mul) > 0]
    return o3.Irreps([(mul, ir) for mul, ir in blocks if mul > 0])


def _reduced_non_scalar_irreps(irreps: o3.Irreps, channels: int) -> o3.Irreps:
    """Small equivariant target used to build learned cross-channel invariants."""

    blocks = [
        (min(int(mul), int(channels)), ir) for mul, ir in irreps if int(mul) > 0
    ]
    return o3.Irreps([(mul, ir) for mul, ir in blocks if mul > 0])


class EquivariantMambaACEBlock(nn.Module):
    """Use invariant mixer states to gate equivariant ACE token values."""

    def __init__(
        self,
        node_irreps,
        token_irreps,
        hidden_dim: int,
        num_token_kinds: int,
        sequence_length: int | None = None,
        token_reduction: str = "sqrt_length",
        mixer_type: str = "mamba",
        attention_heads: int = 4,
        mamba_dim: int = 64,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        bidirectional_tied: bool = False,
        mamba_variant: str = "mamba3",
        headdim: int | None = None,
        rope_fraction: float = 0.5,
        a_floor: float = 1.0e-4,
        chunk_size: int = 64,
        angle_mode: str = "official",
        mimo_rank: int = 1,
        rotary_layout: str = "halves",
        scan_mode: str = "auto",
        ffn_hidden: int | None = None,
        ffn_type: str | None = None,
        invariant_pair_channels: int = 0,
        invariant_norm: str = "squared",
        invariant_norm_eps: float = 1.0e-4,
        invariant_overlap_width: int = 0,
        shell_pair_channels: int = 0,
        shell_pair_width: int = 1,
        shell_pair_mode: str = "banded",
        shell_pair_state_clip: float = 4.0,
        decay_mode: str = "free",
        screening_min_angstrom: float = 0.15,
        shell_spacing_angstrom: float = 1.0,
        coupling_mode: str = "gate",
        coupling_channels: int = 8,
        num_experts: int = 0,
        expert_hidden: int | None = None,
        expert_latent_dim: int | None = None,
        router_tau: float = 1.0,
        router_switch: str = "c2",
        router_threshold_init: float | None = None,
        router_balance_rate: float = 0.0,
        router_balance_target: float | None = None,
        routing_backend: str = "dense",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_scale_init: float | None = 1.0e-2,
        backend: str = "auto",
    ):
        super().__init__()
        if not 0.0 <= float(attention_dropout) < 1.0:
            raise ValueError("attention_dropout must lie in [0, 1)")
        self.node_irreps = o3.Irreps(node_irreps)
        self.token_irreps = o3.Irreps(token_irreps)
        self.hidden_dim = int(hidden_dim)
        if not self.node_irreps or self.node_irreps[0].ir != o3.Irrep("0e"):
            raise ValueError("node_irreps must begin with even scalar channels")
        if self.node_irreps[0].mul != self.hidden_dim:
            raise ValueError("the leading node scalar multiplicity must equal hidden_dim")
        if not self.token_irreps or self.token_irreps[0].ir != o3.Irrep("0e"):
            raise ValueError("token_irreps must begin with even scalar channels")
        self.node_blocks = _irrep_blocks(self.node_irreps)
        self.num_node_copies = sum(mul for mul, _ in self.node_irreps)
        self.token_scalar_dim = int(self.token_irreps[0].mul)
        token_reduction = token_reduction.lower()
        if token_reduction not in {"sum", "sqrt_length"}:
            raise ValueError("token_reduction must be 'sum' or 'sqrt_length'")
        self.token_reduction = token_reduction

        self.node_non_scalar_irreps = o3.Irreps(self.node_irreps[1:])
        self.token_non_scalar_irreps = o3.Irreps(self.token_irreps[1:])
        self.invariant_norm = str(invariant_norm).lower()
        self.invariant_norm_eps = float(invariant_norm_eps)
        _norm = lambda irreps: IrrepInvariantNorm(
            irreps, mode=self.invariant_norm, eps=self.invariant_norm_eps
        )
        self.node_norms = _norm(self.node_non_scalar_irreps)
        self.token_norms = _norm(self.token_non_scalar_irreps)

        # Per-copy squared norms discard every cross-channel overlap
        # <x_{c1,l}, x_{c2,l}>.  A small equivariant linear map followed by a
        # squared norm restores a learned, rank-limited set of those overlaps,
        # because |a x1 + b x2|^2 = a^2|x1|^2 + 2ab<x1,x2> + b^2|x2|^2.  Setting
        # invariant_pair_channels=0 recovers the architecture-v8 invariant set.
        invariant_pair_channels = int(invariant_pair_channels)
        if invariant_pair_channels < 0:
            raise ValueError("invariant_pair_channels must be nonnegative")
        self.invariant_pair_channels = invariant_pair_channels
        self.node_pair_projection = None
        self.token_pair_projection = None
        self.node_pair_norms = None
        self.token_pair_norms = None
        node_pair_dim = 0
        token_pair_dim = 0
        if invariant_pair_channels > 0 and self.node_non_scalar_irreps.dim > 0:
            target = _reduced_non_scalar_irreps(
                self.node_non_scalar_irreps, invariant_pair_channels
            )
            if target.dim > 0:
                self.node_pair_projection = o3.Linear(
                    self.node_non_scalar_irreps, target
                )
                self.node_pair_norms = _norm(target)
                node_pair_dim = self.node_pair_norms.irreps_out.dim
        if invariant_pair_channels > 0 and self.token_non_scalar_irreps.dim > 0:
            target = _reduced_non_scalar_irreps(
                self.token_non_scalar_irreps, invariant_pair_channels
            )
            if target.dim > 0:
                self.token_pair_projection = o3.Linear(
                    self.token_non_scalar_irreps, target
                )
                self.token_pair_norms = _norm(target)
                token_pair_dim = self.token_pair_norms.irreps_out.dim

        node_invariant_dim = hidden_dim + self.node_norms.irreps_out.dim + node_pair_dim
        # ---- shell-alignment invariants ------------------------------------
        # The invariant map currently supplies the mixer with the *diagonal* of
        # the shell Gram matrix,
        #
        #     G^{c,l}_{k,k'} = sum_m T_{i k c l m} T_{i k' c l m},
        #
        # since ||T_ik||^2 = G_kk.  The off-diagonal band is a genuinely
        # independent degree of freedom: measured on the five CsPbI3 polymorphs
        # the adjacent-shell alignment cos(theta_{k,k+1}) spans [-0.16, 1.00]
        # with standard deviation 0.43, so the Cauchy-Schwarz bound
        # |G_{k,k+1}| <= sqrt(G_kk G_{k+1,k+1}) is nowhere near saturated and the
        # shell norms do not determine it.
        #
        # The stored quantity is the cosine rather than the raw overlap:
        #
        #     c^{(d)}_{k} = G_{k,k+d} / ( sqrt(G_kk G_{k+d,k+d}) + eps ) ,
        #
        # which is bounded in [-1, 1], smooth because the denominator is bounded
        # below by eps, and homogeneous of degree zero.  A magnitude (degree one)
        # and an angle (degree zero) are different kinds of quantity, so mixing
        # them is a polar decomposition rather than the degree-1/degree-2 clash
        # that motivated ``invariant_norm='homogeneous'``.
        #
        # Honest scope: on the CsPbI3 phases this band separates polymorphs only
        # about as well as the diagonal (median ratio 1.07), so it is cheap and
        # sound but not transformative.  It is off by default.
        invariant_overlap_width = int(invariant_overlap_width)
        if invariant_overlap_width < 0:
            raise ValueError("invariant_overlap_width must be nonnegative")
        self.invariant_overlap_width = invariant_overlap_width
        self.token_overlap_blocks = (
            _irrep_blocks(self.token_irreps) if invariant_overlap_width > 0 else ()
        )
        overlap_dim = invariant_overlap_width * sum(
            mul for mul, _ in self.token_overlap_blocks
        )

        token_invariant_dim = (
            self.token_scalar_dim
            + self.token_norms.irreps_out.dim
            + token_pair_dim
            + overlap_dim
        )
        self.node_invariant_dim = node_invariant_dim
        self.token_invariant_dim = token_invariant_dim

        self.node_context = nn.Linear(node_invariant_dim, mamba_dim, bias=False)
        self.token_input = nn.Linear(token_invariant_dim, mamba_dim, bias=False)
        self.kind_embedding = nn.Embedding(num_token_kinds, mamba_dim)
        self.coordinate_projection = nn.Linear(1, mamba_dim, bias=False)
        # Continuum quadrature factors, set by the owning model.  Both are one at
        # the resolution the model was constructed with.
        self.token_density_scale = 1.0
        self.step_scale = 1.0
        self.sequence_length = (
            int(sequence_length) if sequence_length is not None else None
        )
        mixer_type = mixer_type.lower()
        mamba_variant = mamba_variant.lower()
        if mamba_variant not in {"mamba1", "mamba3"}:
            raise ValueError("mamba_variant must be 'mamba1' or 'mamba3'")
        if ffn_type is None:
            # Historical coupling of the scalar residual block to the mixer,
            # retained so the canonical architecture and pre-v9 checkpoints keep
            # their exact structure.  New v2 models pass an explicit value.
            ffn_type = (
                "swiglu"
                if (mixer_type != "mamba" or mamba_variant == "mamba3")
                else "mlp"
            )
        ffn_type = str(ffn_type).lower()
        if ffn_type not in {"swiglu", "mlp"}:
            raise ValueError("ffn_type must be 'swiglu' or 'mlp'")
        self.mixer_type = mixer_type
        self.mamba_variant = mamba_variant
        self.ffn_type = ffn_type
        # The scalar residual block is now an independent choice.  Coupling it to
        # the mixer, as architecture v8 did, silently broke the "only the mixer
        # changes" contract of the controlled comparison.
        self.use_swiglu = ffn_type == "swiglu"
        if mixer_type == "mamba":
            if mamba_variant == "mamba1":
                self.mixer = MambaSequenceMixer(
                    d_model=mamba_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bidirectional_tied=bidirectional_tied,
                    backend=backend,
                )
            elif mamba_variant == "mamba3":
                resolved_headdim = int(headdim or math.gcd(16, mamba_dim * expand))
                if resolved_headdim < 2:
                    raise ValueError(
                        "automatic headdim resolution produced a degenerate value; "
                        "choose mamba_dim * mamba_expand divisible by a power of two "
                        "or set mamba_headdim explicitly"
                    )
                self.mixer = Mamba3SequenceMixer(
                    d_model=mamba_dim,
                    d_state=d_state,
                    expand=expand,
                    headdim=resolved_headdim,
                    rope_fraction=rope_fraction,
                    a_floor=a_floor,
                    chunk_size=chunk_size,
                    angle_mode=angle_mode,
                    mimo_rank=mimo_rank,
                    rotary_layout=rotary_layout,
                    scan_mode=scan_mode,
                    bidirectional_tied=bidirectional_tied,
                    backend=backend,
                )
        elif mixer_type == "attention":
            self.mixer = AttentionSequenceMixer(
                d_model=mamba_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
            )
        elif mixer_type == "dense":
            if sequence_length is None:
                raise ValueError("dense mixer requires a fixed sequence_length")
            self.mixer = DenseRadialSequenceMixer(mamba_dim, sequence_length)
        elif mixer_type in {"mlp", "deepsets"}:
            self.mixer = DeepSetsSequenceMixer(mamba_dim, expand=expand)
        elif mixer_type == "identity":
            self.mixer = IdentitySequenceMixer(mamba_dim)
        else:
            raise ValueError(
                "mixer_type must be 'mamba', 'attention', 'dense', 'mlp', or 'identity'"
            )
        # ---- physically constrained decay -----------------------------------
        # "free"      : the Mamba-3 default, alpha = exp(Delta A) with both
        #               factors learned per head and per shell.
        # "screening" : alpha_k = exp(-dr / lambda_i) for a single per-atom
        #               length lambda_i, so the unrolled kernel is
        #                   K(k,k') ~ exp(-|r_k - r_k'| / lambda_i),
        #               a screened (Yukawa / Thomas-Fermi) radial correlation.
        #
        # This *reduces* parameters -- one scalar per atom replaces a free rate
        # per head and shell -- and turns an opaque internal quantity into a
        # predicted observable in Angstrom that can be compared against a
        # Thomas-Fermi screening length or the first minimum of the radial
        # distribution function.  It is a constraint, so it can only cost
        # capacity; the question it answers is whether the physics it encodes is
        # worth that cost.
        decay_mode = str(decay_mode).lower()
        if decay_mode not in {"free", "screening"}:
            raise ValueError("decay_mode must be 'free' or 'screening'")
        self.decay_mode = decay_mode
        self.screening_min_angstrom = float(screening_min_angstrom)
        if self.screening_min_angstrom <= 0.0:
            raise ValueError("screening_min_angstrom must be positive")
        self.shell_spacing_angstrom = float(shell_spacing_angstrom)
        if self.shell_spacing_angstrom <= 0.0:
            raise ValueError("shell_spacing_angstrom must be positive")
        self.screening_projection = (
            nn.Linear(node_invariant_dim, 1)
            if decay_mode == "screening"
            else None
        )

        self.gate_projection = nn.Linear(mamba_dim, self.num_node_copies)
        self.value_projection = o3.Linear(self.token_irreps, self.node_irreps)
        self.output_projection = o3.Linear(self.node_irreps, self.node_irreps)

        # ---- mixer-emitted equivariant path weights -------------------------
        # In "gate" mode the value map W_V is fixed and the mixer only rescales
        # each irrep copy, so it can amplify an l = 2 channel but never change how
        # channels within an l combine.  In "path_weights" mode the mixer emits
        # the weights of the equivariant map itself,
        #
        #   dh_i = W_up sum_k Linear(D_down T_ik ; W_path(m_ik)) ,
        #
        # which is exactly the NequIP radial-MLP construction driven by the mixer
        # state instead of by r.  Equivariance is exact because the emitted
        # weights are O(3) invariants; a rotation commutes through the map.
        #
        # The scalar gate is the special case where the emitted weight is a
        # multiple of the identity on each path, so this mode is strictly more
        # expressive on the reduced subspace.
        #
        # Initialization scale, measured rather than assumed.  With bounded
        # emitted weights the update starts 1.50x the gate-mode update at
        # identical seed and geometry (0.4826 against 0.3222); unbounded weights
        # gave 2.53x.  The residual factor is real -- a full path map has more
        # active degrees of freedom than a diagonal gate -- and ``layer_scale``
        # absorbs it during training.  It does not affect the controlled mixer
        # comparison, where ``coupling_mode`` is held fixed across mixers, but a
        # gate-versus-path-weights ablation should report it.
        #
        # Memory is the reason for the down-projection.  Per-sample weights for
        # the *full* token -> node map cost N x L x weight_numel; at the
        # production irreps that is 2688 floats per shell per atom, i.e. 328 MiB
        # for 1000 atoms at L = 32.  Restricting the data-dependent map to
        # ``coupling_channels`` reduces that to 48 floats at 4 channels (5.9 MiB)
        # and 192 at 8 channels (23 MiB), with fixed maps on either side.
        coupling_mode = str(coupling_mode).lower()
        if coupling_mode not in {"gate", "path_weights"}:
            raise ValueError("coupling_mode must be 'gate' or 'path_weights'")
        self.coupling_mode = coupling_mode
        self.coupling_map = None
        if coupling_mode == "path_weights":
            coupling_channels = int(coupling_channels)
            if coupling_channels < 1:
                raise ValueError("coupling_channels must be positive")
            coupling_irreps = _reduced_irreps(self.token_irreps, coupling_channels)
            if coupling_irreps.dim == 0:
                raise ValueError("coupling_channels leaves no token irrep")
            self.coupling_irreps = coupling_irreps
            self.coupling_channels = coupling_channels
            self.coupling_down = o3.Linear(self.token_irreps, coupling_irreps)
            self.coupling_map = o3.Linear(
                coupling_irreps, coupling_irreps,
                internal_weights=False, shared_weights=False,
            )
            self.coupling_path = nn.Linear(
                mamba_dim, self.coupling_map.weight_numel
            )
            # e3nn normalizes assuming unit-variance weights.  Put that scale in
            # the bias so the layer starts as a fixed random equivariant map, and
            # let the data-dependent part grow from the default small weight init.
            nn.init.normal_(self.coupling_path.bias, mean=0.0, std=1.0)
            self.coupling_up = o3.Linear(coupling_irreps, self.node_irreps)

        # ---- banded shell-pair correlation ------------------------------------
        # The linear term above is the only place shell tokens reach the energy,
        # and it is linear in T_ik.  All many-body structure therefore has to come
        # through a scalar gate, which caps what the mixer can express: it can
        # reweight the shell-resolved density but never correlate two radii.
        #
        # This term adds
        #
        #   dh^(2)_i = W_up sum_k sum_{d=0}^{D} eta^(d)_ik  TP_d(D T_ik, D T_i,k+d)
        #
        # which is a *banded* piece of the double sum
        # TP(A_i, A_i) = sum_{k,k'} TP(T_ik, T_ik').  Only the full, untied double
        # sum reproduces the ACE object, so a narrow band carries information the
        # direct ACE path does not have whenever L exceeds the number of ACE radial
        # channels.  Physically it is angular correlation resolved between two
        # radii -- bond-angle information at a specified pair of distances.
        #
        # Cost is O(N L (D+1)) with D small, so the model stays linear in both the
        # atom count and the shell count.  Shifted tokens are zero-padded, which
        # keeps the C^(degree-1) smoothness of the tokenizer.
        # ``shell_pair_mode`` selects how the degree-2 (and higher) shell
        # correlation is generated.  All three are exactly O(3)-equivariant,
        # permutation invariant, and inherit the tokenizer's smoothness order.
        #
        # "banded"      sum_k sum_{d<=D} eta^(d)_k TP(T_k, T_{k+d}) -- a hard
        #               radial band of width D.  Degree 2.  O(N L D).
        #
        # "exponential" S_k = a_k S_{k-1} + b_k TP(T_{k-1}, T_k)
        #                                 + c_k TP(T_k, T_k)
        #               The band becomes soft and *learned*: unrolling gives
        #                   K(k, k') ~ exp(-sum_{s=k'+1}^{k} Delta_s a_s)
        #               so the radial correlation length is environment
        #               dependent instead of a fixed hyperparameter.  The
        #               recurrence is linear in the state, so the associative
        #               scan applies and the cost stays O(N L log L) parallel.
        #               Degree 2.
        #
        # "cg_ssm"      S_k = a_k S_{k-1} + b_k TP(S_{k-1}, T_k) + c_k W_1 T_k
        #               The state is equivariant and the update is a tensor
        #               product, so deg_T(S_k) <= k: after L shells the state
        #               carries shell correlations up to degree L, i.e. body
        #               order L+1, at O(N L) cost.  This is the radial analogue
        #               of the ACE correlation recursion
        #                   C^(nu+1) = TP(C^(nu), A)
        #               made selective and given a learned decay.  The price is
        #               that the recurrence is nonlinear in the state, so no
        #               associative scan exists and the L steps are sequential.
        #               A smooth saturating clip keeps the repeated products
        #               bounded.
        shell_pair_channels = int(shell_pair_channels)
        shell_pair_width = int(shell_pair_width)
        shell_pair_mode = str(shell_pair_mode).lower()
        if shell_pair_mode not in {"banded", "exponential", "cg_ssm"}:
            raise ValueError(
                "shell_pair_mode must be 'banded', 'exponential', or 'cg_ssm'"
            )
        self.shell_pair_mode = shell_pair_mode
        self.shell_pair_state_clip = float(shell_pair_state_clip)
        if self.shell_pair_state_clip <= 0.0:
            raise ValueError("shell_pair_state_clip must be positive")
        if shell_pair_channels < 0:
            raise ValueError("shell_pair_channels must be nonnegative")
        if shell_pair_width < 0:
            raise ValueError("shell_pair_width must be nonnegative")
        self.shell_pair_channels = shell_pair_channels
        self.shell_pair_width = shell_pair_width
        self.shell_pair_products = None
        if shell_pair_channels > 0:
            pair_irreps = _reduced_irreps(self.token_irreps, shell_pair_channels)
            if pair_irreps.dim == 0:
                raise ValueError("shell_pair_channels leaves no token irrep")
            self.shell_pair_irreps = pair_irreps
            self.shell_pair_down = o3.Linear(self.token_irreps, pair_irreps)
            self.shell_pair_products = nn.ModuleList(
                [
                    o3.FullyConnectedTensorProduct(
                        pair_irreps, pair_irreps, pair_irreps,
                        internal_weights=True, shared_weights=True,
                    )
                    for _ in range(shell_pair_width + 1 if shell_pair_mode == "banded" else 1)
                ]
            )
            self.num_shell_pair_copies = sum(int(mul) for mul, _ in pair_irreps)
            if shell_pair_mode != "banded":
                # alpha (decay), beta (product drive), gamma (linear drive)
                self.shell_pair_dynamics = nn.Linear(mamba_dim, 3)
                self.shell_pair_readout = nn.Linear(
                    mamba_dim, self.num_shell_pair_copies
                )
                if shell_pair_mode == "exponential":
                    self.shell_pair_self = o3.FullyConnectedTensorProduct(
                        pair_irreps, pair_irreps, pair_irreps,
                        internal_weights=True, shared_weights=True,
                    )
                else:
                    self.shell_pair_linear = o3.Linear(pair_irreps, pair_irreps)
            self.shell_pair_gate = nn.Linear(
                mamba_dim, (shell_pair_width + 1) * self.num_shell_pair_copies
            )
            self.shell_pair_blocks = _irrep_blocks(pair_irreps)
            self.shell_pair_up = o3.Linear(pair_irreps, self.node_irreps)
            self.shell_pair_scale = (
                nn.Parameter(
                    torch.full((self.num_node_copies,), float(layer_scale_init))
                )
                if layer_scale_init is not None
                else None
            )
        self.dropout = IrrepDropout(self.node_irreps, dropout)
        self.layer_scale = (
            nn.Parameter(torch.full((self.num_node_copies,), float(layer_scale_init)))
            if layer_scale_init is not None
            else None
        )

        ffn_hidden = int(ffn_hidden or 2 * hidden_dim)
        self.scalar_norm = nn.LayerNorm(node_invariant_dim)
        if self.use_swiglu:
            self.scalar_ffn_in = nn.Linear(node_invariant_dim, 2 * ffn_hidden)
            self.scalar_ffn_out = nn.Linear(ffn_hidden, hidden_dim)
            self.scalar_ffn_dropout = nn.Dropout(dropout)
        else:
            self.scalar_ffn = nn.Sequential(
                nn.Linear(node_invariant_dim, ffn_hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden, hidden_dim),
                nn.Dropout(dropout),
            )
        self.ffn_scale = (
            nn.Parameter(torch.full((hidden_dim,), float(layer_scale_init)))
            if layer_scale_init is not None
            else None
        )

        # ---- smooth compact-support expert routing ---------------------------
        # The scalar residual block is a per-atom map of the O(3) invariants, so
        # it is the one place in this architecture where a mixture of experts can
        # be inserted without touching equivariance at all: the router reads
        # invariants and emits invariant weights, and a rotation commutes through
        # untouched.  What it cannot be given is a top-k rule -- see the module
        # docstring of ``routing.py`` for why that makes the forces distributional
        # rather than merely inaccurate.
        #
        # ``num_experts = 0`` is the default and leaves the arithmetic of this
        # block bit-for-bit unchanged, so the term is strictly additive.
        num_experts = int(num_experts)
        if num_experts < 0:
            raise ValueError("num_experts must be nonnegative")
        if expert_hidden is not None and int(expert_hidden) < 1:
            # ``expert_hidden or ffn_hidden`` would read 0 as "unset" and
            # silently substitute the FFN width instead of rejecting it.
            raise ValueError("expert_hidden must be positive when given")
        self.num_experts = num_experts
        self.routed_ffn = None
        if num_experts > 0:
            self.routed_ffn = RoutedScalarFFN(
                context_dim=node_invariant_dim,
                out_dim=hidden_dim,
                num_experts=num_experts,
                expert_hidden=int(expert_hidden or ffn_hidden),
                swiglu=self.use_swiglu,
                tau=router_tau,
                contract=router_switch,
                threshold_init=router_threshold_init,
                expert_scale_init=layer_scale_init,
                backend=routing_backend,
                latent_dim=expert_latent_dim,
                balance_rate=router_balance_rate,
                balance_target=router_balance_target,
            )

    def set_resolution_scaling(
        self, token_density_scale: float, step_scale: float
    ) -> None:
        """Install the continuum quadrature factors for the current shell mesh."""

        self.token_density_scale = float(token_density_scale)
        self.step_scale = float(step_scale)

    def _node_invariants(self, x: torch.Tensor) -> torch.Tensor:
        non_scalar = x[:, self.hidden_dim :]
        parts = [x[:, : self.hidden_dim], self.node_norms(non_scalar)]
        if self.node_pair_projection is not None:
            parts.append(self.node_pair_norms(self.node_pair_projection(non_scalar)))
        return torch.cat(parts, dim=-1)

    def _shell_alignment(self, tokens: torch.Tensor) -> torch.Tensor:
        """Cosines between the angular patterns of shells k and k + d.

        Returns ``(atoms, shells, width * copies)``.  Shells beyond the last are
        zero-padded, which sets their cosine to zero -- the correct value, since
        an absent shell carries no angular pattern to align with.
        """

        atoms, length, _ = tokens.shape
        cosines = []
        for delta in range(1, self.invariant_overlap_width + 1):
            if delta >= length:
                cosines.append(
                    tokens.new_zeros(
                        (atoms, length, sum(m for m, _ in self.token_overlap_blocks))
                    )
                )
                continue
            shifted = torch.cat(
                (
                    tokens[:, delta:],
                    tokens.new_zeros((atoms, delta, tokens.shape[-1])),
                ),
                dim=1,
            )
            offset = 0
            per_delta = []
            for multiplicity, irrep_dim in self.token_overlap_blocks:
                width = multiplicity * irrep_dim
                left = tokens[:, :, offset : offset + width].reshape(
                    atoms, length, multiplicity, irrep_dim
                )
                right = shifted[:, :, offset : offset + width].reshape(
                    atoms, length, multiplicity, irrep_dim
                )
                # Contracting over m makes each entry an O(3) invariant.
                overlap = (left * right).sum(dim=-1)
                norm_left = left.square().sum(dim=-1)
                norm_right = right.square().sum(dim=-1)
                denominator = torch.sqrt(
                    norm_left * norm_right + self.invariant_norm_eps**4
                ) + self.invariant_norm_eps**2
                per_delta.append(overlap / denominator)
                offset += width
            cosines.append(torch.cat(per_delta, dim=-1))
        return torch.cat(cosines, dim=-1)

    def _token_invariants(self, tokens: torch.Tensor) -> torch.Tensor:
        scalar = tokens[..., : self.token_scalar_dim]
        non_scalar = tokens[..., self.token_scalar_dim :]
        flat = non_scalar.reshape(-1, non_scalar.shape[-1])
        norms = self.token_norms(flat).reshape(tokens.shape[0], tokens.shape[1], -1)
        parts = [scalar, norms]
        if self.token_pair_projection is not None:
            pair = self.token_pair_norms(self.token_pair_projection(flat))
            parts.append(pair.reshape(tokens.shape[0], tokens.shape[1], -1))
        if self.invariant_overlap_width > 0:
            parts.append(self._shell_alignment(tokens))
        return torch.cat(parts, dim=-1)

    def _reduce_token_updates(self, weighted_values: torch.Tensor) -> torch.Tensor:
        update = weighted_values.sum(dim=1)
        if self.token_reduction == "sqrt_length":
            update = update / math.sqrt(float(weighted_values.shape[1]))
        return update

    def _controls(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        token_kind: torch.Tensor,
        token_coordinate: torch.Tensor,
    ) -> torch.Tensor:
        # The control path consumes the shell *density* T_ik / dxi so that its
        # invariants converge as the mesh is refined.  The value path keeps the
        # measure T_ik, which already makes the shell sum a quadrature.
        control_tokens = (
            tokens if self.token_density_scale == 1.0 else tokens * self.token_density_scale
        )
        controls = self.token_input(self._token_invariants(control_tokens))
        controls = controls + self.node_context(self._node_invariants(x))[:, None, :]
        controls = controls + self.kind_embedding(token_kind)[None, :, :]
        controls = controls + self.coordinate_projection(token_coordinate[:, None])[
            None, :, :
        ]
        return controls

    def screening_length(self, node_invariants: torch.Tensor) -> torch.Tensor:
        """Per-atom screening length in Angstrom, bounded below by a floor.

        Returned with shape ``(atoms, 1)``.  The floor keeps ``dr / lambda``
        finite; without it a vanishing length would drive the decay exponent to
        minus infinity and the recurrence would lose every shell instantly.
        """

        if self.screening_projection is None:
            raise RuntimeError("screening_length requires decay_mode='screening'")
        raw = self.screening_projection(node_invariants)
        return self.screening_min_angstrom + F.softplus(raw)

    def _mixer_states(
        self,
        controls: torch.Tensor,
        require_higher_order: bool,
        node_invariants: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mixer_type == "mamba" and self.mamba_variant == "mamba3":
            screening = None
            if self.decay_mode == "screening":
                if node_invariants is None:
                    raise RuntimeError(
                        "decay_mode='screening' needs the node invariants"
                    )
                lengths = self.screening_length(node_invariants)
                # dr / lambda_i, broadcast over the shell and head axes.
                screening = (
                    self.shell_spacing_angstrom / lengths
                )[:, :, None]
            return self.mixer(
                controls,
                require_higher_order=require_higher_order,
                step_scale=self.step_scale,
                screening=screening,
            )
        return self.mixer(controls, require_higher_order=require_higher_order)

    def _shell_dynamics(self, states: torch.Tensor):
        """Invariant decay and drive coefficients for the shell recurrences."""

        raw = self.shell_pair_dynamics(states)
        # alpha in (0, 1): a decay, never a growth.  Written as exp(-softplus)
        # so the unrolled kernel is exp(-sum Delta a) with a > 0, i.e. a genuine
        # radial correlation length rather than an unbounded gain.
        alpha = torch.exp(-F.softplus(raw[..., 0:1]))
        beta = torch.tanh(raw[..., 1:2])
        gamma = torch.tanh(raw[..., 2:3])
        return alpha, beta, gamma

    def _shifted(self, reduced: torch.Tensor) -> torch.Tensor:
        """``T_{k-1}`` with a zero at k = 0: no shell exists below the first."""

        return torch.cat(
            (torch.zeros_like(reduced[:, :1]), reduced[:, :-1]), dim=1
        )

    def _exponential_shell_state(self, reduced: torch.Tensor, states: torch.Tensor):
        """Degree-2 correlation with a learned exponential radial band."""

        atoms, length, width = reduced.shape
        alpha, beta, gamma = self._shell_dynamics(states)
        flat = reduced.reshape(-1, width)
        previous = self._shifted(reduced).reshape(-1, width)
        cross = self.shell_pair_products[0](previous, flat).reshape(atoms, length, -1)
        self_term = self.shell_pair_self(flat, flat).reshape(atoms, length, -1)
        drive = beta * cross + gamma * self_term
        forward = affine_shell_scan(drive, alpha)
        # The radial axis has no causal direction, so scan both ways and average
        # with the variance-preserving factor already used by the mixer.
        backward = torch.flip(
            affine_shell_scan(
                torch.flip(drive, dims=(1,)), torch.flip(alpha, dims=(1,))
            ),
            dims=(1,),
        )
        return math.sqrt(0.5) * (forward + backward)

    def _cg_shell_state(self, reduced: torch.Tensor, states: torch.Tensor):
        """Nonlinear equivariant recursion; degree in T grows with the shell index."""

        atoms, length, width = reduced.shape
        alpha, beta, gamma = self._shell_dynamics(states)
        linear = self.shell_pair_linear(reduced.reshape(-1, width)).reshape(
            atoms, length, width
        )
        clip = self.shell_pair_state_clip
        outputs = []
        state = reduced.new_zeros((atoms, width))
        for step in range(length):
            product = self.shell_pair_products[0](state, reduced[:, step])
            state = (
                alpha[:, step] * state
                + beta[:, step] * product
                + gamma[:, step] * linear[:, step]
            )
            # Smooth saturating clip.  Repeated tensor products would otherwise
            # grow geometrically over L steps; this bounds ||S|| by ``clip``
            # while staying differentiable and exactly equivariant, because the
            # factor is an invariant scalar.
            scale = torch.rsqrt(
                1.0 + state.pow(2).sum(dim=-1, keepdim=True) / (clip * clip)
            )
            state = state * scale
            outputs.append(state)
        return torch.stack(outputs, dim=1)

    def _shell_pair_update(
        self, tokens: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor | None:
        """Banded equivariant correlation between shells ``k`` and ``k + d``."""

        if self.shell_pair_products is None:
            return None
        atoms, length, _ = tokens.shape
        reduced = self.shell_pair_down(tokens.reshape(-1, tokens.shape[-1])).reshape(
            atoms, length, -1
        )
        if self.shell_pair_mode != "banded":
            if self.shell_pair_mode == "exponential":
                state = self._exponential_shell_state(reduced, states)
            else:
                state = self._cg_shell_state(reduced, states)
            weights = _expand_irrep_scalars(
                torch.tanh(self.shell_pair_readout(states)), self.shell_pair_blocks
            )
            update = self.shell_pair_up(self._reduce_token_updates(weights * state))
            if self.shell_pair_scale is not None:
                update = update * _expand_irrep_scalars(
                    self.shell_pair_scale, self.node_blocks
                )
            return update
        gates = torch.tanh(self.shell_pair_gate(states))
        gates = gates.reshape(atoms, length, self.shell_pair_width + 1, -1)
        total = None
        for offset, product in enumerate(self.shell_pair_products):
            if offset == 0:
                shifted = reduced
            elif offset < length:
                # Zero padding, not wrap-around: shell L-1 has no partner beyond
                # the cutoff, and padding keeps the tokenizer's smoothness order.
                shifted = torch.cat(
                    (
                        reduced[:, offset:],
                        reduced.new_zeros((atoms, offset, reduced.shape[-1])),
                    ),
                    dim=1,
                )
            else:
                continue
            correlation = product(
                reduced.reshape(-1, reduced.shape[-1]),
                shifted.reshape(-1, shifted.shape[-1]),
            ).reshape(atoms, length, -1)
            weighted = correlation * _expand_irrep_scalars(
                gates[:, :, offset], self.shell_pair_blocks
            )
            total = weighted if total is None else total + weighted
        if total is None:
            return None
        update = self.shell_pair_up(self._reduce_token_updates(total))
        if self.shell_pair_scale is not None:
            update = update * _expand_irrep_scalars(
                self.shell_pair_scale, self.node_blocks
            )
        return update

    def _path_weight_update(
        self, tokens: torch.Tensor, states: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Equivariant map whose weights the mixer emits, summed over shells."""

        atoms, length, _ = tokens.shape
        reduced = self.coupling_down(tokens.reshape(-1, tokens.shape[-1]))
        # tanh is not decoration: it makes this mode the exact generalization of
        # the gate coupling, which applies tanh to one scalar per irrep copy,
        # from a per-copy scalar to a per-path weight.  Without it the emitted map
        # lacks the gate's damping and starts 2.5x larger, which would make the
        # controlled mixer comparison unfair and the two modes incomparable.  It
        # also bounds the map, which matters for molecular-dynamics stability.
        emitted = (
            torch.tanh(self.coupling_path(states)) if weights is None else weights
        )
        mapped = self.coupling_map(reduced, emitted.reshape(-1, emitted.shape[-1]))
        mapped = mapped.reshape(atoms, length, -1)
        return self.coupling_up(self._reduce_token_updates(mapped))

    def _token_values(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.value_projection(tokens.reshape(-1, tokens.shape[-1])).reshape(
            tokens.shape[0], tokens.shape[1], self.node_irreps.dim
        )

    @torch.no_grad()
    def gate_statistics(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        token_kind: torch.Tensor,
        token_coordinate: torch.Tensor,
    ) -> dict[str, float]:
        """Quantify how much of the update depends on the shell index.

        See :meth:`MambaACEV2.gate_shell_dependence` for the interpretation.
        """

        node_invariants = self._node_invariants(x)
        controls = self._controls(x, tokens, token_kind, token_coordinate)
        states = self._mixer_states(controls, False, node_invariants)
        if self.coupling_mode == "path_weights":
            # The shell-constant baseline replaces the emitted weights by their
            # mean over shells, which is exactly the map a mixer with no radial
            # selectivity would apply; by sum_k T_ik = A_i that collapses onto the
            # direct ACE path, so the interpretation of the residual is unchanged.
            emitted = torch.tanh(self.coupling_path(states))
            full = self._path_weight_update(tokens, states, emitted)
            constant = self._path_weight_update(
                tokens, states, emitted.mean(dim=1, keepdim=True).expand_as(emitted)
            )
            probe = emitted
        else:
            gates = torch.tanh(self.gate_projection(states))
            expanded = _expand_irrep_scalars(gates, self.node_blocks)
            values = self._token_values(tokens)
            full = self._reduce_token_updates(expanded * values)
            constant = self._reduce_token_updates(
                expanded.mean(dim=1, keepdim=True).expand_as(expanded) * values
            )
            probe = gates
        full_norm = float(full.norm())
        residual = float((full - constant).norm())
        return {
            "gate_abs_mean": float(probe.abs().mean()),
            "gate_std_over_shells": float(probe.std(dim=1).mean()),
            "gate_std_over_channels": float(probe.std(dim=-1).mean()),
            "update_norm": full_norm,
            "residual_fraction": residual / full_norm if full_norm > 0.0 else 0.0,
        }

    def _mixed_features(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        token_kind: torch.Tensor,
        token_coordinate: torch.Tensor,
        require_higher_order: bool = False,
    ) -> torch.Tensor:
        """The equivariant half of the block: ``x + dh``, before the scalar FFN."""

        controls = self._controls(x, tokens, token_kind, token_coordinate)
        states = self._mixer_states(
            controls, require_higher_order, self._node_invariants(x)
        )

        if self.coupling_mode == "path_weights":
            update = self._path_weight_update(tokens, states)
        else:
            gates = torch.tanh(self.gate_projection(states))
            gates = _expand_irrep_scalars(gates, self.node_blocks)
            values = self._token_values(tokens)
            update = self._reduce_token_updates(gates * values)
        update = self.output_projection(self.dropout(update))
        if self.layer_scale is not None:
            update = update * _expand_irrep_scalars(self.layer_scale, self.node_blocks)
        pair_update = self._shell_pair_update(tokens, states)
        if pair_update is not None:
            update = update + pair_update
        return x + update

    @torch.no_grad()
    def routing_statistics(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        token_kind: torch.Tensor,
        token_coordinate: torch.Tensor,
    ) -> dict[str, float] | None:
        """Expert occupancy for this layer, or ``None`` when routing is off.

        Reads the invariants through the same path ``forward`` uses -- after the
        equivariant update and after ``scalar_norm`` -- so the reported weights
        are the ones the experts are actually gated by, not an approximation
        taken from the layer input.
        """

        if self.routed_ffn is None:
            return None
        mixed = self._mixed_features(x, tokens, token_kind, token_coordinate)
        invariants = self.scalar_norm(self._node_invariants(mixed))
        return self.routed_ffn.routing_statistics(invariants)

    def forward(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        token_kind: torch.Tensor,
        token_coordinate: torch.Tensor,
        require_higher_order: bool = False,
    ) -> torch.Tensor:
        x = self._mixed_features(
            x, tokens, token_kind, token_coordinate, require_higher_order
        )

        invariants = self._node_invariants(x)
        normalized_invariants = self.scalar_norm(invariants)
        if self.use_swiglu:
            gate, value = self.scalar_ffn_in(normalized_invariants).chunk(2, dim=-1)
            scalar_update = self.scalar_ffn_out(F.silu(gate) * value)
            scalar_update = self.scalar_ffn_dropout(scalar_update)
        else:
            scalar_update = self.scalar_ffn(normalized_invariants)
        if self.ffn_scale is not None:
            scalar_update = scalar_update * self.ffn_scale
        if self.routed_ffn is not None:
            # Added after ``ffn_scale`` because the routed branch carries its own
            # ``expert_scale``; scaling it twice would make the two branches
            # start at different orders for no stated reason.  The shared branch
            # above is the always-on baseline that lets every expert switch off
            # smoothly without the energy collapsing.
            scalar_update = scalar_update + self.routed_ffn(normalized_invariants)
        return torch.cat(
            (x[:, : self.hidden_dim] + scalar_update, x[:, self.hidden_dim :]), dim=-1
        )
