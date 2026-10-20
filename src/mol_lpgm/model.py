from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from .attention import RotationlessIPAAttention
from .batch import MolecularBatch


@dataclass(frozen=True)
class ModelConfig:
    """Shared molecular network; dataset configurations select width and depth."""

    width: int = 512
    depth: int = 12
    heads: int = 8
    feedforward_width: int = 2048
    query_points: int = 4
    value_points: int = 8
    time_fourier_dim: int = 128
    flex_block_size: int = 128
    scalar_qk_norm: bool = True
    atom_class_count: int = 14
    max_fragment_length: int = 128
    normalize_time_embedding: bool = True
    coordinate_head: str = "fragment"


@dataclass
class Prediction:
    center: torch.Tensor
    rotation: torch.Tensor
    internal: torch.Tensor
    atom_logits: torch.Tensor


@dataclass
class CartesianPrediction:
    displacement: torch.Tensor
    atom_logits: torch.Tensor


class FourierTimeEmbedding(nn.Module):
    def __init__(self, fourier_dim: int, width: int):
        super().__init__()
        if fourier_dim % 2:
            raise ValueError("time Fourier dimension must be even")
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1_000.0), fourier_dim // 2)
        )
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * torch.pi * time[:, None] * self.frequencies[None]
        return self.projection(torch.cat((phase.sin(), phase.cos()), dim=-1))


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden_width: int, residual_scale: float):
        super().__init__()
        self.up = nn.Linear(width, hidden_width, bias=False)
        self.gate = nn.Linear(width, hidden_width, bias=False)
        self.down = nn.Linear(hidden_width, width, bias=False)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.down.weight)
        with torch.no_grad():
            self.down.weight.mul_(residual_scale)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down(self.up(hidden) * F.silu(self.gate(hidden)))


class AdaLNIPABlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        residual_scale = 1.0 / config.depth
        self.attention_norm = nn.LayerNorm(
            config.width, elementwise_affine=False, eps=1e-5
        )
        self.feedforward_norm = nn.LayerNorm(
            config.width, elementwise_affine=False, eps=1e-5
        )
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.width, 6 * config.width)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.attention = RotationlessIPAAttention(
            config.width,
            config.heads,
            query_points=config.query_points,
            value_points=config.value_points,
            block_size=config.flex_block_size,
            residual_scale=residual_scale,
            scalar_qk_norm=config.scalar_qk_norm,
        )
        self.feedforward = SwiGLU(
            config.width, config.feedforward_width, residual_scale
        )

    @staticmethod
    def _modulate(
        hidden: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return hidden * (1.0 + scale) + shift

    def forward(
        self,
        hidden: torch.Tensor,
        time_embedding: torch.Tensor,
        coords: torch.Tensor,
        document_id: torch.Tensor,
        block_mask: BlockMask | None,
        fragment_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            feedforward_shift,
            feedforward_scale,
            feedforward_gate,
        ) = self.modulation(time_embedding).chunk(6, dim=-1)
        attention_input = self._modulate(
            self.attention_norm(hidden), attention_shift, attention_scale
        )
        hidden = hidden + attention_gate * self.attention(
            attention_input,
            coords,
            document_id,
            block_mask,
            fragment_id=fragment_id,
        )
        feedforward_input = self._modulate(
            self.feedforward_norm(hidden), feedforward_shift, feedforward_scale
        )
        return hidden + feedforward_gate * self.feedforward(feedforward_input)


class MolecularModel(nn.Module):
    """Translation-equivariant, deliberately rotation-sensitive molecular IPA."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.atom_class_embedding = nn.Embedding(config.atom_class_count, config.width)
        self.fragment_length_embedding = nn.Embedding(
            config.max_fragment_length + 1, config.width
        )
        self.time_embedding = FourierTimeEmbedding(
            config.time_fourier_dim, config.width
        )
        self.time_embedding_norm = (
            nn.LayerNorm(config.width, elementwise_affine=False, eps=1e-5)
            if config.normalize_time_embedding
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(AdaLNIPABlock(config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.width)
        if config.coordinate_head == "cartesian":
            # Pool hidden features by fragment; the output is one Cartesian displacement.
            self.coordinate_head = nn.Linear(2 * config.width, 3)
            coordinate_heads = (self.coordinate_head,)
        elif config.coordinate_head == "fragment":
            self.internal_head = nn.Linear(config.width, 3)
            self.center_head = nn.Linear(config.width, 3)
            self.rotation_head = nn.Linear(config.width, 3)
            coordinate_heads = (
                self.internal_head,
                self.center_head,
                self.rotation_head,
            )
        else:
            raise ValueError("coordinate_head must be fragment or cartesian")
        self.atom_label_head = nn.Linear(config.width, config.atom_class_count)

        nn.init.normal_(self.atom_class_embedding.weight, std=0.02)
        nn.init.normal_(self.fragment_length_embedding.weight, std=0.02)
        for head in (*coordinate_heads, self.atom_label_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _token_input(self, batch: MolecularBatch) -> torch.Tensor:
        length = batch.fragment_length[batch.fragment_id].clamp(
            0, self.config.max_fragment_length
        )
        hidden = self.atom_class_embedding(
            batch.atom_class
        ) + self.fragment_length_embedding(length)
        return hidden * batch.valid_atom[:, None].to(hidden.dtype)

    def _fragment_pool(
        self, hidden: torch.Tensor, batch: MolecularBatch
    ) -> torch.Tensor:
        pooled = hidden.new_zeros((batch.fragment_capacity, hidden.shape[-1]))
        valid = batch.valid_atom[:, None].to(hidden.dtype)
        pooled.index_add_(0, batch.fragment_id, hidden * valid)
        denominator = batch.fragment_atom_count.clamp_min(1).to(hidden.dtype)[:, None]
        return pooled / denominator

    def forward(
        self,
        batch: MolecularBatch,
        block_mask: BlockMask | None = None,
    ) -> Prediction:
        hidden = self._token_input(batch)
        time_embedding = self.time_embedding_norm(self.time_embedding(batch.time_token))
        coords = batch.xt[None]
        for block in self.blocks:
            hidden = block(
                hidden[None],
                time_embedding[None],
                coords,
                batch.document_id,
                block_mask,
                fragment_id=batch.fragment_id,
            )[0]
        hidden = self.final_norm(hidden)
        fragment_hidden = self._fragment_pool(hidden, batch)
        if self.config.coordinate_head == "cartesian":
            return CartesianPrediction(
                displacement=self.coordinate_head(
                    torch.cat((hidden, fragment_hidden[batch.fragment_id]), dim=-1)
                ),
                atom_logits=self.atom_label_head(hidden),
            )
        return Prediction(
            center=self.center_head(fragment_hidden),
            rotation=self.rotation_head(fragment_hidden),
            internal=self.internal_head(hidden),
            atom_logits=self.atom_label_head(hidden),
        )
