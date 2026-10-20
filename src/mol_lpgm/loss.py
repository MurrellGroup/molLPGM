from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .batch import MolecularBatch
from .model import Prediction, CartesianPrediction
from .cartesian_loss import (
    CartesianLossTerms,
    cartesian_loss_terms,
    reduce_cartesian_loss,
    cartesian_binned_loss,
)
from .structured import (
    LossBreakdown,
    StructuredLossTerms,
    StructuredPrediction,
    reduce_structured_loss,
    structured_loss_terms,
)
from .weighting import TimeLossWeights


@dataclass
class LossTerms:
    continuous: StructuredLossTerms
    atom_label: torch.Tensor
    atom_time: torch.Tensor
    valid_atom: torch.Tensor
    logit_offset: torch.Tensor | None = None


@dataclass
class MolecularLoss:
    total: torch.Tensor
    center: torch.Tensor
    rotation: torch.Tensor
    internal: torch.Tensor
    atom_label: torch.Tensor
    raw_total: torch.Tensor
    raw_center: torch.Tensor
    raw_rotation: torch.Tensor
    raw_internal: torch.Tensor
    raw_atom_label: torch.Tensor
    logit_offset: torch.Tensor
    raw_logit_offset: torch.Tensor


def resolve_logit_offset_weight(
    requested: float | None, saved_state: dict | None
) -> float:
    saved = float((saved_state or {}).get("logit_offset_weight", 0.01))
    weight = saved if requested is None else float(requested)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("logit offset weight must be finite and nonnegative")
    if saved_state is not None and weight != saved:
        raise ValueError("logit offset weight conflicts with checkpoint")
    return weight


def shared_logit_offset(logits: torch.Tensor) -> torch.Tensor:
    """Per-token gauge penalty: its logit gradient is equal in every class."""
    return logits.float().mean(dim=-1).square()


def loss_terms(
    prediction: Prediction,
    batch: MolecularBatch,
    *,
    regularisation: float = 1e-5,
) -> LossTerms:
    if isinstance(prediction, CartesianPrediction):
        return cartesian_loss_terms(prediction, batch)
    continuous = structured_loss_terms(
        StructuredPrediction(
            center=prediction.center,
            rotation=prediction.rotation,
            internal=prediction.internal,
        ),
        batch,
        regularisation=regularisation,
    )
    label = F.cross_entropy(
        prediction.atom_logits.float(),
        batch.target_atom_class,
        reduction="none",
    )
    return LossTerms(
        continuous=continuous,
        atom_label=label,
        atom_time=batch.time_token,
        valid_atom=batch.valid_atom.to(label.dtype),
        logit_offset=shared_logit_offset(prediction.atom_logits),
    )


def reduce_loss(
    terms: LossTerms,
    *,
    time_weights: TimeLossWeights | None = None,
    atom_label_divisor: float = 3.0,
    atom_label_epsilon: float = 0.2,
    logit_offset_weight: float = 0.0,
) -> MolecularLoss:
    if isinstance(terms, CartesianLossTerms):
        return reduce_cartesian_loss(
            terms,
            time_weights,
            atom_label_divisor,
            atom_label_epsilon,
            logit_offset_weight,
        )
    continuous: LossBreakdown = reduce_structured_loss(
        terms.continuous, time_weights=time_weights
    )
    atom_denominator = terms.valid_atom.sum().clamp_min(1.0)
    raw_atom_label = (terms.atom_label * terms.valid_atom).sum() / atom_denominator
    # Flowfusion/QM9 endpoint-prediction weighting:
    # scalefloss(P, t, pow=1, eps=0.2) / 3.
    label_weight = (
        1.0 / (1.0 + atom_label_epsilon - terms.atom_time)
    ) / atom_label_divisor
    if time_weights is not None:
        label_weight = label_weight * time_weights.multiplier(
            "atom_label", terms.atom_time
        )
    atom_label = (
        terms.atom_label * terms.valid_atom * label_weight
    ).sum() / atom_denominator
    raw_offset = (
        terms.atom_label.new_zeros(())
        if terms.logit_offset is None
        else (terms.logit_offset * terms.valid_atom).sum() / atom_denominator
    )
    offset = logit_offset_weight * raw_offset
    return MolecularLoss(
        total=continuous.total + atom_label + offset,
        center=continuous.center,
        rotation=continuous.rotation,
        internal=continuous.internal,
        atom_label=atom_label,
        raw_total=continuous.raw_total
        + raw_atom_label
        + (raw_offset if logit_offset_weight else 0.0),
        raw_center=continuous.raw_center,
        raw_rotation=continuous.raw_rotation,
        raw_internal=continuous.raw_internal,
        raw_atom_label=raw_atom_label,
        logit_offset=offset,
        raw_logit_offset=raw_offset,
    )


def binned_loss(
    terms: LossTerms, *, time_weights: TimeLossWeights, bins: int = 10
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted/raw contributions and occupancy, each [4,bins].

    Full-batch denominators are retained: summing time bins exactly recovers
    each component objective. Dividing a contribution by occupancy gives its
    conditional mean within that bin. Padding never contributes.
    """
    if isinstance(terms, CartesianLossTerms):
        return cartesian_binned_loss(terms, time_weights, bins)
    continuous = terms.continuous
    cweights = time_weights.evaluate(continuous.time)
    aweight = time_weights.multiplier("atom_label", terms.atom_time) / (
        3 * (1.2 - terms.atom_time)
    )
    weighted, raw, occupancy = [], [], []
    for values, time, valid, weight in zip(
        (continuous.center, continuous.rotation, continuous.internal, terms.atom_label),
        (continuous.time,) * 3 + (terms.atom_time,),
        (continuous.valid,) * 3 + (terms.valid_atom,),
        (*cweights, aweight),
    ):
        index = (time.float() * bins).long().clamp(0, bins - 1)
        denominator = valid.sum().clamp_min(1.0)

        def pool(value):
            return (
                value.new_zeros(bins).scatter_add(0, index, value * valid) / denominator
            )

        weighted.append(pool(values * weight))
        raw.append(pool(values))
        occupancy.append(pool(torch.ones_like(values)))
    return torch.stack(weighted), torch.stack(raw), torch.stack(occupancy)
