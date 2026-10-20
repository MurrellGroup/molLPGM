"""One atom-wise Cartesian endpoint loss; no fragment geometric operations."""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .weighting import CartesianLossWeights


@dataclass
class CartesianLossTerms:
    coordinate: torch.Tensor
    atom_label: torch.Tensor
    atom_time: torch.Tensor
    valid_atom: torch.Tensor
    logit_offset: torch.Tensor


@dataclass
class CartesianLoss:
    total: torch.Tensor
    coordinate: torch.Tensor
    atom_label: torch.Tensor
    raw_total: torch.Tensor
    raw_coordinate: torch.Tensor
    raw_atom_label: torch.Tensor
    logit_offset: torch.Tensor
    raw_logit_offset: torch.Tensor


def cartesian_loss_terms(prediction, batch):
    return CartesianLossTerms(
        (prediction.displacement.float() - (batch.x1 - batch.xt).float())
        .square()
        .mean(-1),
        F.cross_entropy(
            prediction.atom_logits.float(), batch.target_atom_class, reduction="none"
        ),
        batch.time_token,
        batch.valid_atom.float(),
        prediction.atom_logits.float().mean(-1).square(),
    )


def reduce_cartesian_loss(
    terms,
    time_weights=None,
    atom_label_divisor=3.0,
    atom_label_epsilon=0.2,
    logit_offset_weight=0.0,
):
    weights = time_weights if time_weights is not None else CartesianLossWeights()
    if not isinstance(weights, CartesianLossWeights):
        raise ValueError("Cartesian training cannot use fragment loss weights")

    def mean(v):
        return (v * terms.valid_atom).sum() / terms.valid_atom.sum().clamp_min(1.0)

    coordinate = mean(terms.coordinate * weights.coordinate_weight(terms.atom_time))
    label = mean(
        terms.atom_label
        * weights.multiplier("atom_label", terms.atom_time)
        / (atom_label_divisor * (1 + atom_label_epsilon - terms.atom_time))
    )
    raw_coordinate, raw_label, raw_offset = map(
        mean, (terms.coordinate, terms.atom_label, terms.logit_offset)
    )
    offset = logit_offset_weight * raw_offset
    return CartesianLoss(
        coordinate + label + offset,
        coordinate,
        label,
        raw_coordinate + raw_label + (raw_offset if logit_offset_weight else 0.0),
        raw_coordinate,
        raw_label,
        offset,
        raw_offset,
    )


def cartesian_binned_loss(terms, time_weights, bins=10):
    if not isinstance(time_weights, CartesianLossWeights):
        raise ValueError("Cartesian training cannot use fragment loss weights")
    time, valid = terms.atom_time, terms.valid_atom
    index = (time.float() * bins).long().clamp(0, bins - 1)

    def pool(v):
        return v.new_zeros(bins).scatter_add(
            0, index, v * valid
        ) / valid.sum().clamp_min(1.0)

    weights = (
        time_weights.coordinate_weight(time),
        time_weights.multiplier("atom_label", time) / (3 * (1.2 - time)),
    )
    values = (terms.coordinate, terms.atom_label)
    return (
        torch.stack([pool(v * w) for v, w in zip(values, weights)]),
        torch.stack([pool(v) for v in values]),
        torch.stack([pool(torch.ones_like(v)) for v in values]),
    )
