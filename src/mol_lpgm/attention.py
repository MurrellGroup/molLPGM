from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

_compiled_create_block_mask = torch.compile(
    create_block_mask, fullgraph=True, dynamic=False
)


def make_document_block_mask(
    document_id: torch.Tensor,
    *,
    heads: int,
    block_size: int = 128,
    valid: torch.Tensor | None = None,
) -> BlockMask:
    """
    Construct a block-sparse, noncausal document mask.

    This is intentionally the only production masking representation: no
    token-by-token T×T document or padding mask is allocated.
    """
    if document_id.ndim != 1:
        raise ValueError("concatenation batching requires one 1-D document-id stream")
    docs = document_id.contiguous()
    length = docs.numel()

    if valid is None:

        def mask_mod(batch, head, query_index, key_index):
            del batch, head
            return docs[query_index] == docs[key_index]

    else:
        valid_tokens = valid.contiguous()
        if valid_tokens.shape != docs.shape:
            raise ValueError("valid mask must match document-id stream")

        def mask_mod(batch, head, query_index, key_index):
            del batch, head
            return (
                valid_tokens[query_index]
                & valid_tokens[key_index]
                & (docs[query_index] == docs[key_index])
            )

    return _compiled_create_block_mask(
        mask_mod,
        B=1,
        H=heads,
        Q_LEN=length,
        KV_LEN=length,
        device=docs.device,
        BLOCK_SIZE=block_size,
    )


def _split_heads(projected: torch.Tensor, heads: int) -> torch.Tensor:
    batch, length, width = projected.shape
    return projected.view(batch, length, heads, width // heads).permute(0, 2, 1, 3)


def _merge_heads(values: torch.Tensor) -> torch.Tensor:
    batch, heads, length, width = values.shape
    return values.permute(0, 2, 1, 3).contiguous().view(batch, length, heads * width)


def _point_heads(projected: torch.Tensor, heads: int, points: int) -> torch.Tensor:
    batch, length, _ = projected.shape
    return projected.view(batch, length, heads, points, 3).permute(0, 2, 1, 3, 4)


def _flatten_points(points: torch.Tensor) -> torch.Tensor:
    return points.flatten(start_dim=-2)


def _document_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    document_id: torch.Tensor,
    fragment_id: torch.Tensor | None = None,
    fragment_pair_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Slow CPU oracle which slices documents and never creates a dense document mask."""
    output = torch.empty(
        (*query.shape[:-1], value.shape[-1]),
        dtype=value.dtype,
        device=value.device,
    )
    boundaries = torch.cat(
        (
            document_id.new_tensor([0]),
            torch.nonzero(document_id[1:] != document_id[:-1]).flatten() + 1,
            document_id.new_tensor([document_id.numel()]),
        )
    )
    scale = 1.0 / math.sqrt(query.shape[-1])
    # The caller's augmented IPA query/key already carries its intended scale;
    # it passes reference_scale=1 by pre-scaling key. This local scale is
    # cancelled below for parity with flex_attention(scale=1).
    del scale
    for boundary_index in range(boundaries.numel() - 1):
        start = int(boundaries[boundary_index])
        stop = int(boundaries[boundary_index + 1])
        scores = query[:, :, start:stop] @ key[:, :, start:stop].transpose(-1, -2)
        if fragment_pair_bias is not None:
            ids = fragment_id[start:stop]
            relation = (ids[:, None] == ids[None, :]).long()
            scores = scores + fragment_pair_bias[:, relation][None]
        probabilities = torch.softmax(scores, dim=-1)
        output[:, :, start:stop] = probabilities @ value[:, :, start:stop]
    return output


class RotationlessIPAAttention(nn.Module):
    """
    IPA with translation-only frames; see docs/models-and-training.md.

    Global learned query/key/value points are anchored at each input atom.
    Distances enter attention logits, while attended value points are returned
    relative to the query atom. Translation cancels in exact arithmetic; the
    learned xyz projections do not enforce rotational equivariance.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        *,
        query_points: int,
        value_points: int,
        block_size: int,
        residual_scale: float,
        scalar_qk_norm: bool = False,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("model width must be divisible by head count")
        self.heads = heads
        self.head_width = width // heads
        self.query_points = query_points
        self.value_points = value_points
        self.block_size = block_size
        # Two relation categories, independent of atom order and fragment-ID
        # numbering. Zero initialization imposes no initial attention preference.
        self.fragment_pair_bias = nn.Parameter(torch.zeros(heads, 2))
        self.scalar_query_norm = (
            nn.RMSNorm(self.head_width, eps=1e-5) if scalar_qk_norm else nn.Identity()
        )
        self.scalar_key_norm = (
            nn.RMSNorm(self.head_width, eps=1e-5) if scalar_qk_norm else nn.Identity()
        )
        self.scalar_qkv = nn.Linear(width, 3 * width, bias=False)
        self.query_key_points = nn.Linear(
            width, 2 * 3 * heads * query_points, bias=False
        )
        self.value_point_projection = nn.Linear(
            width, 3 * heads * value_points, bias=False
        )
        output_width = heads * (self.head_width + 4 * value_points)
        self.output = nn.Linear(output_width, width, bias=False)
        self.head_weight = nn.Parameter(torch.zeros(heads))
        nn.init.xavier_uniform_(self.scalar_qkv.weight)
        nn.init.xavier_uniform_(self.query_key_points.weight)
        nn.init.xavier_uniform_(self.value_point_projection.weight)
        nn.init.xavier_uniform_(self.output.weight)
        with torch.no_grad():
            self.output.weight.mul_(residual_scale)

    def forward(
        self,
        hidden: torch.Tensor,
        coords: torch.Tensor,
        document_id: torch.Tensor,
        block_mask: BlockMask | None,
        fragment_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        if self.fragment_pair_bias is not None and fragment_id is None:
            raise ValueError("fragment pair conditioning requires fragment IDs")
        query_raw, key_raw, value_raw = self.scalar_qkv(hidden).chunk(3, dim=-1)
        query = _split_heads(query_raw, self.heads)
        key = _split_heads(key_raw, self.heads)
        value = _split_heads(value_raw, self.heads)
        query = self.scalar_query_norm(query)
        key = self.scalar_key_norm(key)

        query_offset, key_offset = self.query_key_points(hidden).chunk(2, dim=-1)
        query_points = (
            _point_heads(query_offset, self.heads, self.query_points)
            + coords[:, None, :, None, :]
        )
        key_points = (
            _point_heads(key_offset, self.heads, self.query_points)
            + coords[:, None, :, None, :]
        )
        value_points = (
            _point_heads(
                self.value_point_projection(hidden), self.heads, self.value_points
            )
            + coords[:, None, :, None, :]
        )

        scalar_scale = 1.0 / math.sqrt(3.0 * self.head_width)
        point_scale = F.softplus(self.head_weight).view(1, self.heads, 1, 1)
        point_scale = point_scale * math.sqrt(2.0 / (9.0 * self.query_points))
        query_squared = query_points.square().sum(dim=-1)
        key_squared = key_points.square().sum(dim=-1)
        ones = torch.ones_like(query_squared)
        augmented_query = torch.cat(
            (query, _flatten_points(query_points), query_squared, ones), dim=-1
        )
        augmented_key = torch.cat(
            (
                scalar_scale * key,
                point_scale * _flatten_points(key_points),
                -0.5 * point_scale * ones,
                -0.5 * point_scale * key_squared,
            ),
            dim=-1,
        )
        augmented_value = torch.cat((value, _flatten_points(value_points)), dim=-1)

        if hidden.is_cuda:
            if block_mask is None:
                raise ValueError("CUDA IPA requires a FlexAttention BlockMask")
            if not torch.compiler.is_compiling():
                raise RuntimeError(
                    "CUDA IPA must be invoked through torch.compile; eager "
                    "FlexAttention materializes dense scores"
                )
            score_mod = None
            if self.fragment_pair_bias is not None:
                pair_bias = self.fragment_pair_bias
                if torch.is_grad_enabled() and pair_bias.requires_grad:
                    # FlexAttention scatters captured-buffer gradients with
                    # atomics. A shared [head, relation] buffer concentrates all
                    # pairs onto two addresses/head. Give each query its own
                    # temporary storage, then let autograd sum back to the same
                    # learned parameter. Real storage (not a stride-zero view)
                    # is essential. This is O(H*N), never an N*N pair tensor.
                    pair_bias = (
                        pair_bias[:, None, :].expand(self.heads, length, 2).contiguous()
                    )

                    def score_mod(score, batch_index, head, query_index, key_index):
                        relation = (
                            fragment_id[query_index] == fragment_id[key_index]
                        ).long()
                        return score + pair_bias[head, query_index, relation]

                else:

                    def score_mod(score, batch_index, head, query_index, key_index):
                        relation = (
                            fragment_id[query_index] == fragment_id[key_index]
                        ).long()
                        return score + pair_bias[head, relation]

            attended = flex_attention(
                augmented_query,
                augmented_key,
                augmented_value,
                block_mask=block_mask,
                score_mod=score_mod,
                scale=1.0,
            )
        else:
            attended = _document_attention_reference(
                augmented_query,
                augmented_key,
                augmented_value,
                document_id,
                fragment_id,
                self.fragment_pair_bias,
            )

        scalar = attended[..., : self.head_width]
        point_flat = attended[..., self.head_width :]
        points = point_flat.view(batch, self.heads, length, self.value_points, 3)
        relative = points - coords[:, None, :, None, :]
        point_norm = torch.sqrt(relative.square().sum(dim=-1) + 1e-8)
        merged = _merge_heads(
            torch.cat((_flatten_points(relative), point_norm, scalar), dim=-1)
        )
        return self.output(merged)
