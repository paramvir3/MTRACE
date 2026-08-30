"""Conservative MTACE energy, force, and stress model."""

from __future__ import annotations

import contextlib
import operator
from typing import Sequence

import torch
import torch.nn as nn
from e3nn import o3

from .physics import ACEV2MambaTokenizer, CanonicalACETokenizer
from .routing import resolve_switch_contract
from .schedule import resolve_mixer_schedule
from .ssm import EquivariantMambaACEBlock


def _positive_integer(name: str, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = operator.index(value)
    except TypeError as exception:
        raise ValueError(f"{name} must be a positive integer") from exception
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(parsed)


def _voigt_basis(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """The six symmetric strain generators, in the order the scalar path uses.

    ``basis[k]`` is the matrix that Voigt component ``k`` multiplies, so
    ``epsilon = einsum("bk,kij->bij", strain, basis)`` reproduces the explicit
    assignment in the single-structure branch of :meth:`forward` exactly:
    the three normal components on the diagonal, and each shear component
    written into both symmetric off-diagonal entries.
    """

    basis = torch.zeros((6, 3, 3), dtype=dtype, device=device)
    basis[0, 0, 0] = basis[1, 1, 1] = basis[2, 2, 2] = 1.0
    basis[3, 0, 1] = basis[3, 1, 0] = 1.0
    basis[4, 0, 2] = basis[4, 2, 0] = 1.0
    basis[5, 1, 2] = basis[5, 2, 1] = 1.0
    return basis


def _mamba_rank_and_chunk(mimo_rank: int, chunk_size: int | None) -> tuple[int, int]:
    rank = _positive_integer("mamba_mimo_rank", mimo_rank)
    if chunk_size is None:
        return rank, max(1, 64 // rank)
    return rank, _positive_integer("mamba_chunk_size", chunk_size)


class InvariantEnergyReadout(nn.Module):
    def __init__(self, irreps, hidden_dim: int, readout_hidden: int = 64):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.hidden_dim = int(hidden_dim)
        self.non_scalar_irreps = o3.Irreps(self.irreps[1:])
        self.non_scalar_norm = o3.Norm(self.non_scalar_irreps, squared=True)
        invariant_dim = hidden_dim + self.non_scalar_norm.irreps_out.dim
        self.network = nn.Sequential(
            nn.Linear(invariant_dim, readout_hidden),
            nn.SiLU(),
            nn.Linear(readout_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        invariants = torch.cat(
            (
                features[:, : self.hidden_dim],
                self.non_scalar_norm(features[:, self.hidden_dim :]),
            ),
            dim=-1,
        )
        return self.network(invariants).squeeze(-1)


class CanonicalMambaACE(nn.Module):
    """Experimental radial/body-order MTACE architecture."""

    architecture = "mtace_canonical"
    architecture_version = 3

    def __init__(
        self,
        r_max: float = 5.0,
        l_max: int = 2,
        num_radial: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        correlation_order: int = 4,
        correlation_channels: int = 16,
        radial_basis_type: str = "gaussian",
        radial_trainable: bool = False,
        gaussian_width: float = 0.7,
        remove_pair_self_contractions: bool = True,
        mamba_dim: int = 64,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 3,
        mamba_expand: int = 2,
        mamba_bidirectional_tied: bool = False,
        mamba_variant: str = "mamba3",
        mamba_headdim: int | None = None,
        mamba_rope_fraction: float = 0.5,
        mamba_a_floor: float = 1.0e-4,
        mamba_chunk_size: int | None = None,
        mamba_angle_mode: str = "official",
        mamba_mimo_rank: int = 1,
        mamba_backend: str = "auto",
        ffn_hidden: int | None = None,
        dropout: float = 0.0,
        layer_scale_init: float | None = 1.0e-2,
        readout_hidden: int = 64,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.r_max = float(r_max)
        self.hidden_dim = int(hidden_dim)
        mamba_mimo_rank, resolved_chunk_size = _mamba_rank_and_chunk(
            mamba_mimo_rank, mamba_chunk_size
        )
        self.species_embedding = nn.Embedding(119, hidden_dim, padding_idx=0)
        self.ace = CanonicalACETokenizer(
            r_max=r_max,
            l_max=l_max,
            num_radial=num_radial,
            hidden_dim=hidden_dim,
            correlation_order=correlation_order,
            correlation_channels=correlation_channels,
            radial_basis_type=radial_basis_type,
            radial_trainable=radial_trainable,
            gaussian_width=gaussian_width,
            remove_pair_self_contractions=remove_pair_self_contractions,
        )
        self.irreps = self.ace.irreps_out
        self.layers = nn.ModuleList(
            [
                EquivariantMambaACEBlock(
                    node_irreps=self.irreps,
                    token_irreps=self.ace.irreps_token,
                    hidden_dim=hidden_dim,
                    num_token_kinds=self.ace.num_body_tokens + 1,
                    mamba_dim=mamba_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    bidirectional_tied=mamba_bidirectional_tied,
                    mamba_variant=mamba_variant,
                    headdim=mamba_headdim,
                    rope_fraction=mamba_rope_fraction,
                    a_floor=mamba_a_floor,
                    chunk_size=resolved_chunk_size,
                    angle_mode=mamba_angle_mode,
                    mimo_rank=mamba_mimo_rank,
                    ffn_hidden=ffn_hidden,
                    dropout=dropout,
                    layer_scale_init=layer_scale_init,
                    backend=mamba_backend,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = InvariantEnergyReadout(self.irreps, hidden_dim, readout_hidden)

    def atomic_energies(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        cell: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shift: torch.Tensor,
        require_higher_order: bool = False,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.ndim != 1 or z.dtype != torch.long:
            raise ValueError("z must be a one-dimensional torch.long tensor")
        if pos.ndim != 2 or pos.shape != (z.numel(), 3) or not pos.is_floating_point():
            raise ValueError("pos must be a floating tensor with shape (num_atoms, 3)")
        if batch is None:
            if cell.shape != (3, 3) or not cell.is_floating_point():
                raise ValueError("cell must be a floating tensor with shape (3, 3)")
        else:
            # Batched: several structures concatenated into one disconnected
            # graph.  Nothing in this function mixes atoms except through
            # ``edge_index``, so as long as no edge crosses structures the
            # per-atom energies are identical to evaluating each structure on
            # its own.  ``tests/test_batching.py`` asserts that in FP64.
            if batch.dtype != torch.long or batch.shape != (z.numel(),):
                raise ValueError("batch must be torch.long with shape (num_atoms,)")
            if cell.ndim != 3 or cell.shape[1:] != (3, 3) or not cell.is_floating_point():
                raise ValueError("batched cell must be floating with shape (num_graphs, 3, 3)")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must be torch.long with shape (2, num_edges)")
        if edge_shift.shape != (edge_index.shape[1], 3) or not edge_shift.is_floating_point():
            raise ValueError("edge_shift must be floating with shape (num_edges, 3)")
        if z.numel() and (bool(torch.any(z < 1)) or bool(torch.any(z > 118))):
            raise ValueError("atomic numbers must lie in the physical range 1 <= Z <= 118")
        if not bool(torch.isfinite(pos).all()) or not bool(torch.isfinite(cell).all()):
            raise ValueError("positions and cell must be finite")
        if not bool(torch.isfinite(edge_shift).all()):
            raise ValueError("edge shifts must be finite")
        if edge_index.numel() and (
            bool(torch.any(edge_index < 0)) or bool(torch.any(edge_index >= z.numel()))
        ):
            raise ValueError("edge_index contains an atom index outside the structure")
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        if edge_shift.numel() > 0:
            shift = edge_shift.to(dtype=pos.dtype, device=pos.device)
            if batch is None:
                edge_vec = edge_vec + shift @ cell
            else:
                # Each edge takes the cell of the structure it belongs to.  An
                # edge never crosses structures, so the sender's assignment is
                # well defined and equals the receiver's.
                edge_vec = edge_vec + torch.einsum(
                    "ei,eij->ej", shift, cell.index_select(0, batch[edge_index[0]])
                )
        edge_len = torch.linalg.vector_norm(edge_vec, dim=-1)
        if not bool(torch.isfinite(edge_len).all()):
            raise ValueError("edge geometry produced a nonfinite distance")
        if edge_len.numel() and bool(torch.any(edge_len >= self.r_max)):
            raise ValueError("edge_index contains an edge outside r_max")
        features, tokens, token_kind, token_coordinate = self.ace(
            self.species_embedding(z), edge_index, edge_vec, edge_len
        )
        for layer in self.layers:
            features = layer(
                features,
                tokens,
                token_kind,
                token_coordinate,
                require_higher_order=require_higher_order,
            )
        return self.readout(features)

    def forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        detach_pos: bool = True,
        compute_stress: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        z = data["z"]
        pos = data["pos"]
        edge_index = data["edge_index"]
        cell = data.get("cell", torch.eye(3, device=pos.device, dtype=pos.dtype))
        edge_shift = data.get(
            "edge_shift", pos.new_zeros((edge_index.shape[1], 3))
        )
        volume = data.get("volume")
        # ``batch`` maps each atom to its structure.  Absent means a single
        # structure, and that path is left byte-for-byte as it was.
        batch = data.get("batch")
        if batch is not None:
            batch = batch.to(device=pos.device)
            num_graphs = int(batch.max()) + 1 if batch.numel() else 0
        if compute_stress is None:
            compute_stress = bool(training and volume is not None)
        if detach_pos:
            pos = pos.detach()
        pos = pos.requires_grad_(True)
        cell = cell.to(device=pos.device, dtype=pos.dtype)

        if compute_stress and batch is None:
            reference_volume = torch.det(cell).abs()
            if not bool(torch.isfinite(reference_volume)) or bool(reference_volume <= 1.0e-12):
                raise ValueError("stress requires a finite full-rank cell")
            strain = torch.zeros(6, device=pos.device, dtype=pos.dtype, requires_grad=True)
            epsilon = pos.new_zeros((3, 3))
            epsilon[0, 0], epsilon[1, 1], epsilon[2, 2] = strain[:3]
            epsilon[0, 1] = epsilon[1, 0] = strain[3]
            epsilon[0, 2] = epsilon[2, 0] = strain[4]
            epsilon[1, 2] = epsilon[2, 1] = strain[5]
            deformation = torch.eye(3, device=pos.device, dtype=pos.dtype) + epsilon
            deformed_pos = pos @ deformation
            deformed_cell = cell @ deformation
        elif compute_stress:
            reference_volume = torch.det(cell).abs()
            if not bool(torch.isfinite(reference_volume).all()) or bool(
                (reference_volume <= 1.0e-12).any()
            ):
                raise ValueError("stress requires a finite full-rank cell")
            strain = torch.zeros(
                (num_graphs, 6), device=pos.device, dtype=pos.dtype, requires_grad=True
            )
            basis = _voigt_basis(pos.dtype, pos.device)
            epsilon = torch.einsum("bk,kij->bij", strain, basis)
            deformation = torch.eye(3, device=pos.device, dtype=pos.dtype) + epsilon
            # ``pos @ deformation`` per structure, applied per atom.
            deformed_pos = torch.einsum(
                "ni,nij->nj", pos, deformation.index_select(0, batch)
            )
            deformed_cell = torch.bmm(cell, deformation)
        else:
            strain = None
            deformed_pos, deformed_cell = pos, cell

        atomic_energy = self.atomic_energies(
            z,
            deformed_pos,
            deformed_cell,
            edge_index,
            edge_shift,
            require_higher_order=training,
            batch=batch,
        )
        if batch is None:
            energy = atomic_energy.sum()
            total_energy = energy
        else:
            # Per-structure energy by segment sum; the scalar the gradients are
            # taken against is their total, which is what makes one backward
            # pass equivalent to summing per-structure backward passes.
            energy = atomic_energy.new_zeros(num_graphs).index_add(
                0, batch, atomic_energy
            )
            total_energy = energy.sum()
        position_gradient = torch.autograd.grad(
            total_energy,
            pos,
            create_graph=training,
            retain_graph=training or compute_stress,
            allow_unused=True,
        )[0]
        forces = -position_gradient if position_gradient is not None else torch.zeros_like(pos)

        stress = (
            pos.new_zeros((3, 3)) if batch is None
            else pos.new_zeros((num_graphs, 3, 3))
        )
        if compute_stress and strain is not None:
            strain_gradient = torch.autograd.grad(
                total_energy,
                strain,
                create_graph=training,
                retain_graph=training,
                allow_unused=True,
            )[0]
            if strain_gradient is not None and batch is None:
                stress[0, 0], stress[1, 1], stress[2, 2] = strain_gradient[:3]
                stress[0, 1] = stress[1, 0] = 0.5 * strain_gradient[3]
                stress[0, 2] = stress[2, 0] = 0.5 * strain_gradient[4]
                stress[1, 2] = stress[2, 1] = 0.5 * strain_gradient[5]
                stress = stress / reference_volume
            elif strain_gradient is not None:
                # The Voigt basis carries a 1 in both symmetric entries, so the
                # shear components take the same factor 1/2 as the scalar path.
                weights = strain_gradient.new_tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
                stress = torch.einsum(
                    "bk,kij->bij",
                    strain_gradient * weights,
                    _voigt_basis(pos.dtype, pos.device),
                ) / reference_volume[:, None, None]
        return energy, forces, stress, {"atomic_energy": atomic_energy}


class V2ScalarEnergyReadout(nn.Module):
    """TRACE-v2-compatible energy head using only the leading scalar channels."""

    def __init__(self, hidden_dim: int, readout_hidden: int = 64):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, readout_hidden),
            nn.SiLU(),
            nn.Linear(readout_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features[:, : self.hidden_dim]).squeeze(-1)


class MambaACEV2(CanonicalMambaACE):
    """TRACE-v2 ACE with physical shell tokens and interchangeable mixers."""

    architecture = "mtace_v2"
    # v11 adds the per-layer mixer schedule and smooth compact-support expert
    # routing.  Both are strictly additive -- ``mixer_schedule=None`` broadcasts
    # the scalar ``mixer_type`` exactly as before, and ``num_experts=0`` leaves
    # the block arithmetic untouched -- so a v10 checkpoint reproduces its
    # trained energy with no migration.  See ``migrated_model_config``.
    architecture_version = 11

    def __init__(
        self,
        r_max: float = 5.0,
        l_max: int = 2,
        num_radial: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        correlation_order: int = 4,
        correlation_channels: int = 16,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        gaussian_width: float = 0.5,
        radial_mlp_hidden: int = 32,
        radial_mlp_layers: int = 2,
        avg_num_neighbors: float = 1.0,
        tokenizer_type: str = "physical_shells",
        num_shells: int = 32,
        shell_coupling_mode: str | None = None,
        shell_r_min: float = 0.0,
        shell_boundary_mode: str = "fold",
        shell_degree: int = 3,
        shell_scales: int = 1,
        continuum_mode: bool = False,
        mixer_type: str = "mamba",
        mixer_schedule: Sequence[str] | str | None = None,
        attention_heads: int = 4,
        mamba_dim: int = 64,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 3,
        mamba_expand: int = 2,
        mamba_bidirectional_tied: bool = False,
        mamba_variant: str = "mamba3",
        mamba_headdim: int | None = None,
        mamba_rope_fraction: float = 0.5,
        mamba_a_floor: float = 1.0e-4,
        mamba_chunk_size: int | None = None,
        mamba_angle_mode: str = "official",
        mamba_mimo_rank: int = 4,
        mamba_rotary_layout: str = "halves",
        mamba_scan_mode: str = "auto",
        mamba_backend: str = "auto",
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
        coupling_mode: str = "gate",
        coupling_channels: int = 8,
        num_experts: int = 0,
        expert_hidden: int | None = None,
        expert_latent_dim: int | None = None,
        router_tau: float = 1.0,
        router_switch: str = "auto",
        router_threshold_init: float | None = None,
        router_balance_rate: float = 0.0,
        router_balance_target: float | None = None,
        routing_backend: str = "dense",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_scale_init: float | None = 1.0e-2,
        readout_hidden: int = 64,
    ):
        nn.Module.__init__(self)
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        schedule = resolve_mixer_schedule(mixer_type, mixer_schedule, num_layers)
        self.mixer_schedule = schedule
        # ``ffn_type=None`` historically resolved *inside* the block from that
        # block's own mixer.  With a heterogeneous schedule that would silently
        # give attention layers a SwiGLU residual and mamba1 layers a plain MLP,
        # so the stack would differ in two ways at once and the mixer comparison
        # would no longer be controlled.  Resolve it once, here.
        #
        # It must be resolved from the *schedule*, not from the ``mixer_type``
        # shorthand.  Resolving from the shorthand gave
        # ``mixer_schedule=["attention"]`` a different residual block from
        # ``mixer_type="attention"`` whenever ``mamba_variant="mamba1"``, because
        # the shorthand still held its default "mamba" -- two spellings of the
        # same stack producing different models.
        #
        # The block's own rule is ``"mlp"`` exactly when the mixer is Mamba *and*
        # the variant is mamba1, so restricting that to a schedule where every
        # entry is Mamba reproduces it for any homogeneous stack and leaves a
        # heterogeneous stack uniformly on SwiGLU.
        if ffn_type is None:
            ffn_type = (
                "mlp"
                if (
                    mamba_variant.lower() == "mamba1"
                    and all(entry == "mamba" for entry in schedule)
                )
                else "swiglu"
            )
        # A quintic shell tokenizer delivers a C^4 energy; routing it with the
        # C^2 switch would quietly downgrade the whole model to C^2.
        resolved_switch = resolve_switch_contract(router_switch, shell_degree)
        self.r_max = float(r_max)
        self.hidden_dim = int(hidden_dim)
        if shell_coupling_mode is None:
            shell_coupling_mode = (
                "legacy"
                if tokenizer_type.lower() == "legacy_basis"
                else "conservative"
            )
        shell_coupling_mode = shell_coupling_mode.lower()
        if shell_coupling_mode not in {"conservative", "legacy"}:
            raise ValueError(
                "shell_coupling_mode must be 'conservative' or 'legacy'"
            )
        self.shell_coupling_mode = shell_coupling_mode
        mamba_mimo_rank, resolved_chunk_size = _mamba_rank_and_chunk(
            mamba_mimo_rank, mamba_chunk_size
        )
        if mamba_variant.lower() == "mamba3" and int(mamba_d_conv) != 3:
            # Mamba-3 has no depthwise convolution; its width-two interaction is
            # inside the trapezoidal state equation.  Silently ignoring an
            # explicit setting used to hide a configuration error.
            raise ValueError(
                "mamba_d_conv only applies to mamba_variant='mamba1'; Mamba-3 "
                "replaces the depthwise convolution by its trapezoidal state "
                "injection.  Remove mamba_d_conv or select mamba_variant='mamba1'."
            )
        self.continuum_mode = bool(continuum_mode)
        self.species_embedding = nn.Embedding(119, hidden_dim, padding_idx=0)
        self.ace = ACEV2MambaTokenizer(
            r_max=r_max,
            l_max=l_max,
            num_radial=num_radial,
            hidden_dim=hidden_dim,
            correlation_order=correlation_order,
            correlation_channels=correlation_channels,
            radial_basis_type=radial_basis_type,
            radial_trainable=radial_trainable,
            gaussian_width=gaussian_width,
            radial_mlp_hidden=radial_mlp_hidden,
            radial_mlp_layers=radial_mlp_layers,
            avg_num_neighbors=avg_num_neighbors,
            tokenizer_type=tokenizer_type,
            num_shells=num_shells,
            shell_coupling_mode=shell_coupling_mode,
            shell_r_min=shell_r_min,
            shell_boundary_mode=shell_boundary_mode,
            shell_degree=shell_degree,
            shell_scales=shell_scales,
        )
        # Reference resolution for the continuum quadrature.  Every continuum
        # rescaling equals one at this resolution, so architecture v9 with
        # continuum_mode=True reproduces the v8 arithmetic exactly at the shell
        # count it was built with, and only differs when the mesh is refined.
        self.reference_shells = int(self.ace.sequence_length)
        self.irreps = self.ace.irreps_out
        self.layers = nn.ModuleList(
            [
                EquivariantMambaACEBlock(
                    node_irreps=self.irreps,
                    token_irreps=self.ace.irreps_correlation,
                    hidden_dim=hidden_dim,
                    num_token_kinds=self.ace.num_token_kinds,
                    sequence_length=self.ace.sequence_length,
                    token_reduction=(
                        "sum"
                        if shell_coupling_mode == "conservative"
                        else "sqrt_length"
                    ),
                    mixer_type=layer_mixer,
                    attention_heads=attention_heads,
                    mamba_dim=mamba_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    bidirectional_tied=mamba_bidirectional_tied,
                    mamba_variant=mamba_variant,
                    headdim=mamba_headdim,
                    rope_fraction=mamba_rope_fraction,
                    a_floor=mamba_a_floor,
                    chunk_size=resolved_chunk_size,
                    angle_mode=mamba_angle_mode,
                    mimo_rank=mamba_mimo_rank,
                    rotary_layout=mamba_rotary_layout,
                    scan_mode=mamba_scan_mode,
                    ffn_hidden=ffn_hidden,
                    ffn_type=ffn_type,
                    invariant_pair_channels=invariant_pair_channels,
                    invariant_norm=invariant_norm,
                    invariant_norm_eps=invariant_norm_eps,
                    invariant_overlap_width=invariant_overlap_width,
                    shell_pair_channels=shell_pair_channels,
                    shell_pair_width=shell_pair_width,
                    shell_pair_mode=shell_pair_mode,
                    shell_pair_state_clip=shell_pair_state_clip,
                    decay_mode=decay_mode,
                    screening_min_angstrom=screening_min_angstrom,
                    # Physical shell spacing in Angstrom.  The screened decay is
                    # exp(-dr / lambda), so this factor is what makes lambda an
                    # Angstrom-valued observable rather than a per-shell number.
                    shell_spacing_angstrom=(
                        (float(r_max) - float(shell_r_min)) / max(1, int(num_shells) - 1)
                    ),
                    coupling_mode=coupling_mode,
                    coupling_channels=coupling_channels,
                    num_experts=num_experts,
                    expert_hidden=expert_hidden,
                    expert_latent_dim=expert_latent_dim,
                    router_tau=router_tau,
                    router_switch=resolved_switch,
                    router_threshold_init=router_threshold_init,
                    router_balance_rate=router_balance_rate,
                    router_balance_target=router_balance_target,
                    routing_backend=routing_backend,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    layer_scale_init=layer_scale_init,
                    backend=mamba_backend,
                )
                for layer_mixer in schedule
            ]
        )
        self.readout = V2ScalarEnergyReadout(hidden_dim, readout_hidden)
        self._apply_resolution_scaling()

    def _apply_resolution_scaling(self) -> None:
        """Propagate the radial-metric factors for the current shell mesh.

        ``continuum_mode`` rescales the control path by ``1 / dxi`` and the
        state-space step by ``dxi``, which is the discretization a genuine radial
        transfer operator would require.  It is **off by default** because it does
        not achieve its intended purpose, for a reason worth recording: the
        neighbor density of a finite environment is atomic,
        ``rho_i(xi) = sum_j delta(xi - xi_ij) a_ij``, so the shell *density*
        ``T_ik / dxi`` has no continuum limit -- refining the mesh resolves
        individual neighbors and the density grows like ``1 / dxi`` at each
        neighbor distance.  Measured gate fields diverge accordingly
        (``docs/RESOLUTION_STUDY.md``).  A true continuum limit needs a pooling
        kernel whose physical width is decoupled from the shell spacing, which
        also forfeits the four-shell sparsity of the current tokenizer.

        The default measure form is neither exact nor convergent under
        refinement, and the manuscript says so: the reconstruction identity bounds
        the *constant-gate* part of the update but not the shell-dependent part,
        and the measured drift over a fifteenfold refinement is a few tenths of a
        meV per atom and grows.  The shell count is a hyperparameter to be tuned,
        not a mesh to be converged.
        """

        length = int(self.ace.sequence_length)
        if self.continuum_mode and length > 1 and self.reference_shells > 1:
            density_scale = float(length - 1) / float(self.reference_shells - 1)
            step_scale = float(self.reference_shells - 1) / float(length - 1)
        else:
            density_scale = 1.0
            step_scale = 1.0
        for layer in self.layers:
            layer.set_resolution_scaling(density_scale, step_scale)

    def set_num_shells(self, num_shells: int) -> None:
        """Evaluate a trained model on a different physical shell resolution.

        No learned tensor depends on the shell count for the ``mamba``,
        ``attention``, ``mlp`` and ``identity`` mixers, so the radial mesh can be
        refined or coarsened after training.  This is a **research tool, not a
        deployment feature**: the measured study in ``docs/RESOLUTION_STUDY.md``
        shows that the energy of a fixed parameter vector drifts by a small but
        *growing* amount under refinement rather than converging, so the shell
        count is a genuine hyperparameter.  Always revalidate a re-gridded model.
        """

        for layer in self.layers:
            if layer.mixer_type == "dense":
                raise ValueError(
                    "the dense L x L mixer has shell-count-dependent parameters "
                    "and cannot be re-gridded"
                )
        self.ace.set_num_shells(num_shells)
        for layer in self.layers:
            layer.sequence_length = int(self.ace.sequence_length)
        self._apply_resolution_scaling()

    @contextlib.contextmanager
    def _measuring(self):
        """Run a diagnostic without letting it change the model.

        Every diagnostic below advances the stack by calling each layer's
        ``forward``, and in training mode that fires the router's load-balancing
        update -- so measuring the model would move the quantity being measured.
        Eval mode also removes dropout, which a diagnostic should not be seeing
        either.  The previous mode is restored even if the body raises.
        """

        was_training = self.training
        self.eval()
        try:
            yield
        finally:
            self.train(was_training)

    def screening_lengths(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        cell: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shift: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Per-atom screening length in Angstrom, one tensor per layer.

        Only defined for ``decay_mode='screening'``.  This is the quantity the
        constrained decay makes physical: the recurrence memory is
        ``exp(-|r_k - r_k'| / lambda_i)``, so ``lambda_i`` is a screened
        correlation length and can be compared directly against a Thomas-Fermi
        length, the first minimum of the radial distribution function, or the
        coordination shell structure.  It is a per-atom scalar built from
        invariants, so it is itself invariant under translation, O(3) and atom
        relabeling.
        """

        report: list[torch.Tensor] = []
        with self._measuring(), torch.no_grad():
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
            if edge_shift.numel() > 0:
                edge_vec = edge_vec + edge_shift.to(pos) @ cell
            edge_len = torch.linalg.vector_norm(edge_vec, dim=-1)
            features, tokens, token_kind, token_coordinate = self.ace(
                self.species_embedding(z), edge_index, edge_vec, edge_len
            )
            for layer in self.layers:
                if layer.decay_mode != "screening":
                    raise RuntimeError(
                        "screening_lengths requires decay_mode='screening'"
                    )
                report.append(
                    layer.screening_length(layer._node_invariants(features)).squeeze(-1)
                )
                features = layer(features, tokens, token_kind, token_coordinate)
        return report

    def gate_shell_dependence(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        cell: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shift: torch.Tensor,
    ) -> list[dict[str, float]]:
        """Measure how much of the mixer update is *not* reproducible by ACE.

        Because ``sum_k T_ik = A_i``, a gate that is constant across shells makes
        the equivariant update equal to ``W_O W_V A_i``, which the direct ACE
        path of the manuscript already contains.  All of the additional
        expressivity of the mixer therefore lives in the shell dependence of the
        gate.  This diagnostic reports, per layer,

        ``residual_fraction = || dh - dh_const || / || dh ||``

        where ``dh_const`` replaces every gate by its mean over shells.  A value
        near zero means the mixer is decorative and the model has degenerated to
        plain ACE; the quantity should grow during training.
        """

        report: list[dict[str, float]] = []
        with self._measuring(), torch.no_grad():
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
            if edge_shift.numel() > 0:
                edge_vec = edge_vec + edge_shift.to(pos) @ cell
            edge_len = torch.linalg.vector_norm(edge_vec, dim=-1)
            features, tokens, token_kind, token_coordinate = self.ace(
                self.species_embedding(z), edge_index, edge_vec, edge_len
            )
            for index, layer in enumerate(self.layers):
                statistics = layer.gate_statistics(
                    features, tokens, token_kind, token_coordinate
                )
                statistics["layer"] = float(index)
                report.append(statistics)
                features = layer(features, tokens, token_kind, token_coordinate)
        return report

    def routing_occupancy(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        cell: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shift: torch.Tensor,
    ) -> list[dict[str, float]]:
        """Expert occupancy per routed layer; empty when routing is off.

        ``active_fraction`` is the fraction of (atom, expert) pairs whose switch
        weight is nonzero.  It is the quantity the sparse backend saves against,
        and the one that says whether the mixture is genuinely sparse: a run in
        which it stays at 1.0 is a dense model with extra parameters.  Because
        the switch has compact support these zeros are exact, so the number is a
        count and not a threshold on small values.

        Measured under :meth:`_measuring`, so it cannot move the load-balancing
        bias it is reporting on.
        """

        report: list[dict[str, float]] = []
        with self._measuring(), torch.no_grad():
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
            if edge_shift.numel() > 0:
                edge_vec = edge_vec + edge_shift.to(pos) @ cell
            edge_len = torch.linalg.vector_norm(edge_vec, dim=-1)
            features, tokens, token_kind, token_coordinate = self.ace(
                self.species_embedding(z), edge_index, edge_vec, edge_len
            )
            for index, layer in enumerate(self.layers):
                statistics = layer.routing_statistics(
                    features, tokens, token_kind, token_coordinate
                )
                if statistics is not None:
                    statistics["layer"] = float(index)
                    report.append(statistics)
                features = layer(features, tokens, token_kind, token_coordinate)
        return report


# The public default is deliberately the strict v2-first architecture. The
# canonical radial/body-order experiment remains available under its explicit
# class name for later research.
MambaACE = MambaACEV2
