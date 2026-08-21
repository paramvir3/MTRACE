"""Matched sequence-mixer baselines for physically tokenized ACE features."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentitySequenceMixer(nn.Module):
    """No-mixing control with the same input/output contract as learned mixers."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        require_higher_order: bool = False,
        step_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        del require_higher_order, step_scale
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError(f"hidden must have shape (batch, length, {self.d_model})")
        return hidden


class AttentionSequenceMixer(nn.Module):
    """Bidirectional self-attention baseline over the same ACE shell tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("attention dropout probability must lie in [0, 1)")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv_projection = nn.Linear(d_model, 3 * d_model, bias=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        hidden: torch.Tensor,
        require_higher_order: bool = False,
        step_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        del require_higher_order, step_scale
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError(f"hidden must have shape (batch, length, {self.d_model})")
        normalized = self.norm(hidden)
        batch, length = normalized.shape[:2]
        query, key, value = self.qkv_projection(normalized).chunk(3, dim=-1)
        query = query.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        weights = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim),
            dim=-1,
        )
        weights = self.attention_dropout(weights)
        mixed = torch.matmul(weights, value).transpose(1, 2).reshape(
            batch, length, self.d_model
        )
        return hidden + self.output_projection(mixed)


class DenseRadialSequenceMixer(nn.Module):
    """Unconstrained dense token-axis baseline with quadratic shell cost."""

    def __init__(self, d_model: int, sequence_length: int):
        super().__init__()
        if d_model < 1 or sequence_length < 1:
            raise ValueError("d_model and sequence_length must be positive")
        self.d_model = int(d_model)
        self.sequence_length = int(sequence_length)
        self.norm = nn.LayerNorm(d_model)
        self.channel_projection = nn.Linear(d_model, 2 * d_model, bias=False)
        self.token_projection = nn.Linear(sequence_length, sequence_length, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        require_higher_order: bool = False,
        step_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        del require_higher_order, step_scale
        expected = (self.sequence_length, self.d_model)
        if hidden.ndim != 3 or hidden.shape[1:] != expected:
            raise ValueError(
                "hidden must have shape "
                f"(batch, {self.sequence_length}, {self.d_model})"
            )
        gate, value = self.channel_projection(self.norm(hidden)).chunk(2, dim=-1)
        value = F.silu(gate) * value
        value = self.token_projection(value.transpose(1, 2)).transpose(1, 2)
        return hidden + self.output_projection(value)


class DeepSetsSequenceMixer(nn.Module):
    """Linear-cost permutation-equivariant MLP baseline with global context."""

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        if d_model < 1 or expand < 1:
            raise ValueError("d_model and expand must be positive")
        self.d_model = int(d_model)
        inner = int(d_model * expand)
        self.norm = nn.LayerNorm(d_model)
        self.local_projection = nn.Linear(d_model, 2 * inner, bias=False)
        self.context_projection = nn.Linear(d_model, inner, bias=False)
        self.output_projection = nn.Linear(inner, d_model, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        require_higher_order: bool = False,
        step_scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        del require_higher_order, step_scale
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError(f"hidden must have shape (batch, length, {self.d_model})")
        normalized = self.norm(hidden)
        gate, value = self.local_projection(normalized).chunk(2, dim=-1)
        context = self.context_projection(normalized.mean(dim=1, keepdim=True))
        update = F.silu(gate) * (value + context)
        return hidden + self.output_projection(update)
