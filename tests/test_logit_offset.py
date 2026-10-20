from dataclasses import replace

import pytest
import torch

from mol_lpgm.loss import (
    LossTerms,
    reduce_loss,
    resolve_logit_offset_weight,
    shared_logit_offset,
)
from mol_lpgm.structured import StructuredLossTerms


def test_offset_gradient_only_changes_common_logit_direction():
    logits = torch.tensor([[2.0, -1.0, -1.0], [5.0, 2.0, 2.0]], requires_grad=True)
    penalty = shared_logit_offset(logits)
    torch.testing.assert_close(penalty, torch.tensor([0.0, 9.0]))
    gradient = torch.autograd.grad(penalty.sum(), logits)[0]
    torch.testing.assert_close(
        gradient, torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    )
    stepped = logits - 0.1 * gradient
    torch.testing.assert_close(stepped.softmax(-1), logits.softmax(-1))
    assert shared_logit_offset(stepped).sum() < penalty.sum()
