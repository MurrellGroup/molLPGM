from types import SimpleNamespace
import torch
from mol_lpgm.model import CartesianPrediction
from mol_lpgm.cartesian_loss import cartesian_loss_terms, reduce_cartesian_loss


def test_cartesian_target_and_physical_time_weight():
    batch = SimpleNamespace(
        xt=torch.zeros(2, 3),
        x1=torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 0.0]]),
        target_atom_class=torch.tensor([0, 1]),
        time_token=torch.tensor([0.2, 0.8]),
        valid_atom=torch.ones(2),
        target_center=torch.full((1, 3), float("nan")),
        target_rotation=torch.full((1, 3), float("nan")),
        target_internal=torch.full((2, 3), float("nan")),
    )
    prediction = CartesianPrediction(torch.zeros(2, 3), torch.zeros(2, 14))
    terms = cartesian_loss_terms(prediction, batch)
    expected = batch.x1.square().mean(-1)
    torch.testing.assert_close(terms.coordinate, expected)
    loss = reduce_cartesian_loss(terms, logit_offset_weight=0.01)
    torch.testing.assert_close(
        loss.coordinate, (expected / (1.05 - batch.time_token).square()).mean()
    )
    assert torch.isfinite(loss.total)
