from __future__ import annotations

from dataclasses import dataclass

import torch

from .batch import MolecularBatch
from .weighting import TimeLossWeights


@dataclass
class StructuredPrediction:
    center: torch.Tensor
    rotation: torch.Tensor
    internal: torch.Tensor


@dataclass
class LossBreakdown:
    total: torch.Tensor
    center: torch.Tensor
    rotation: torch.Tensor
    internal: torch.Tensor
    raw_total: torch.Tensor
    raw_center: torch.Tensor
    raw_rotation: torch.Tensor
    raw_internal: torch.Tensor


@dataclass
class StructuredLossTerms:
    center: torch.Tensor
    rotation: torch.Tensor
    internal: torch.Tensor
    time: torch.Tensor
    valid: torch.Tensor


def _scatter_sum(
    values: torch.Tensor, index: torch.Tensor, output_size: int
) -> torch.Tensor:
    output = values.new_zeros((output_size, *values.shape[1:]))
    output.index_add_(0, index, values)
    return output


def segment_centers_and_residuals(
    coords: torch.Tensor, batch: MolecularBatch
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = batch.valid_atom[:, None].to(coords.dtype)
    centers = _scatter_sum(coords * valid, batch.segment_id, batch.segment_capacity)
    centers = centers / batch.segment_atom_count.clamp_min(1).to(coords.dtype)[:, None]
    residual = (coords - centers[batch.segment_id]) * valid
    return centers, residual


def project_document_centers(
    center: torch.Tensor, batch: MolecularBatch
) -> torch.Tensor:
    capacity = batch.segment_capacity
    valid = batch.valid_segment.to(center.dtype)
    mass = batch.segment_atom_count.to(center.dtype) * valid
    document_mass = _scatter_sum(
        mass[:, None], batch.segment_document, capacity
    ).squeeze(-1)
    weighted = center * mass[:, None]
    document_sum = _scatter_sum(weighted, batch.segment_document, capacity)
    shift = (
        document_sum[batch.segment_document]
        / document_mass[batch.segment_document].clamp_min(1.0)[:, None]
    )
    return (center - shift) * valid[:, None]


def inertia_tensor(residual: torch.Tensor, batch: MolecularBatch) -> torch.Tensor:
    identity = torch.eye(3, dtype=residual.dtype, device=residual.device)
    atom_inertia = (
        residual.square().sum(dim=-1)[:, None, None] * identity
        - residual[:, :, None] * residual[:, None, :]
    )
    atom_inertia = atom_inertia * batch.valid_atom[:, None, None].to(residual.dtype)
    inertia = _scatter_sum(atom_inertia, batch.segment_id, batch.segment_capacity)
    return (
        inertia
        / batch.segment_atom_count.clamp_min(1).to(residual.dtype)[:, None, None]
    )


def gauge_project(
    prediction: StructuredPrediction,
    batch: MolecularBatch,
    *,
    regularisation: float = 1e-5,
) -> tuple[StructuredPrediction, torch.Tensor, torch.Tensor]:
    # Geometry reductions and batched solves stay in fp32 even when the IPA
    # body runs under bf16 autocast.
    prediction = StructuredPrediction(
        center=prediction.center.float(),
        rotation=prediction.rotation.float(),
        internal=prediction.internal.float(),
    )
    center, residual = segment_centers_and_residuals(batch.xt, batch)
    del center
    valid_atom = batch.valid_atom[:, None].to(prediction.internal.dtype)
    internal_sum = _scatter_sum(
        prediction.internal * valid_atom,
        batch.segment_id,
        batch.segment_capacity,
    )
    internal_mean = (
        internal_sum
        / batch.segment_atom_count.clamp_min(1).to(prediction.internal.dtype)[:, None]
    )
    centered_internal = (
        prediction.internal - internal_mean[batch.segment_id]
    ) * valid_atom

    torque_atom = torch.linalg.cross(residual, centered_internal, dim=-1)
    torque = _scatter_sum(torque_atom, batch.segment_id, batch.segment_capacity)
    count = batch.segment_atom_count.clamp_min(1).to(residual.dtype)
    torque = torque / count[:, None]
    inertia = inertia_tensor(residual, batch)
    radius_squared = residual.square().sum(dim=-1)
    radius_squared = (
        _scatter_sum(
            radius_squared[:, None], batch.segment_id, batch.segment_capacity
        ).squeeze(-1)
        / count
    )
    identity = torch.eye(3, dtype=residual.dtype, device=residual.device)
    regularised = (
        inertia
        + (regularisation * radius_squared.clamp_min(1.0))[:, None, None] * identity
    )
    internal_rotation = torch.linalg.solve(regularised, torque.unsqueeze(-1)).squeeze(
        -1
    )
    projected_internal = centered_internal - torch.linalg.cross(
        internal_rotation[batch.segment_id], residual, dim=-1
    )
    projected = StructuredPrediction(
        center=project_document_centers(prediction.center, batch),
        rotation=prediction.rotation
        * batch.valid_segment[:, None].to(prediction.rotation.dtype),
        internal=projected_internal,
    )
    return projected, residual, inertia


def structured_loss_terms(
    prediction: StructuredPrediction,
    batch: MolecularBatch,
    *,
    regularisation: float = 1e-5,
) -> StructuredLossTerms:
    prediction, residual, inertia = gauge_project(
        prediction, batch, regularisation=regularisation
    )
    del residual
    segment_valid = batch.valid_segment.to(prediction.center.dtype)
    atom_valid = batch.valid_atom.to(prediction.internal.dtype)

    center_error = prediction.center - batch.target_center
    center_terms = center_error.square().sum(dim=-1)

    rotation_error = prediction.rotation - batch.target_rotation
    rotation_terms = torch.einsum(
        "si,sij,sj->s", rotation_error, inertia, rotation_error
    )

    internal_error = (prediction.internal - batch.target_internal).square().sum(
        dim=-1
    ) * atom_valid
    internal_terms = _scatter_sum(
        internal_error[:, None], batch.segment_id, batch.segment_capacity
    ).squeeze(-1)
    internal_terms = internal_terms / batch.segment_atom_count.clamp_min(1).to(
        internal_terms.dtype
    )
    return StructuredLossTerms(
        center=center_terms,
        rotation=rotation_terms,
        internal=internal_terms,
        time=batch.time_segment,
        valid=segment_valid,
    )


def reduce_structured_loss(
    terms: StructuredLossTerms,
    *,
    time_weights: TimeLossWeights | None = None,
    center_weight: float = 1.0,
    rotation_weight: float = 1.0,
    internal_weight: float = 1.0,
) -> LossBreakdown:
    denominator = terms.valid.sum().clamp_min(1.0)
    raw_center = (terms.center * terms.valid).sum() / denominator
    raw_rotation = (terms.rotation * terms.valid).sum() / denominator
    raw_internal = (terms.internal * terms.valid).sum() / denominator
    if time_weights is None:
        center_time = rotation_time = internal_time = torch.ones_like(terms.time)
    else:
        center_time, rotation_time, internal_time = time_weights.evaluate(terms.time)
    center_loss = (terms.center * terms.valid * center_time).sum() / denominator
    rotation_loss = (terms.rotation * terms.valid * rotation_time).sum() / denominator
    internal_loss = (terms.internal * terms.valid * internal_time).sum() / denominator
    total = (
        center_weight * center_loss
        + rotation_weight * rotation_loss
        + internal_weight * internal_loss
    )
    raw_total = (
        center_weight * raw_center
        + rotation_weight * raw_rotation
        + internal_weight * raw_internal
    )
    return LossBreakdown(
        total=total,
        center=center_loss,
        rotation=rotation_loss,
        internal=internal_loss,
        raw_total=raw_total,
        raw_center=raw_center,
        raw_rotation=raw_rotation,
        raw_internal=raw_internal,
    )


def structured_loss(
    prediction: StructuredPrediction,
    batch: MolecularBatch,
    *,
    time_weights: TimeLossWeights | None = None,
    center_weight: float = 1.0,
    rotation_weight: float = 1.0,
    internal_weight: float = 1.0,
    regularisation: float = 1e-5,
) -> LossBreakdown:
    terms = structured_loss_terms(prediction, batch, regularisation=regularisation)
    return reduce_structured_loss(
        terms,
        time_weights=time_weights,
        center_weight=center_weight,
        rotation_weight=rotation_weight,
        internal_weight=internal_weight,
    )
