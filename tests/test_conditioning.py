from dataclasses import fields, replace

import pytest
import torch

from mol_lpgm.batch import MolecularBatch
from mol_lpgm.checkpoint import load_checkpoint, save_checkpoint
from mol_lpgm.model import ModelConfig, MolecularModel
from mol_lpgm.optimizer import MolecularMuon


def _batch(shift=0.0):
    atoms = 8
    fragments = 4
    return MolecularBatch(
        xt=torch.randn(atoms, 3) + shift,
        x1=torch.zeros(atoms, 3),
        target_internal=torch.zeros(atoms, 3),
        atom_class=torch.tensor([0, 2, 2, 0, 3, 0, 4, 0]),
        target_atom_class=torch.tensor([0, 2, 2, 0, 3, 0, 4, 0]),
        atom_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
        fragment_id=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        document_id=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        time_token=torch.full((atoms,), 0.4),
        valid_atom=torch.ones(atoms, dtype=torch.bool),
        target_center=torch.zeros(fragments, 3),
        target_rotation=torch.zeros(fragments, 3),
        fragment_length=torch.full((fragments,), 2),
        fragment_document=torch.tensor([0, 0, 1, 1]),
        fragment_atom_count=torch.full((fragments,), 2),
        internal_variance=torch.full((fragments,), 0.4),
        time_fragment=torch.full((fragments,), 0.4),
        valid_fragment=torch.ones(fragments, dtype=torch.bool),
        metadata={"n_documents": 2},
    )


@pytest.mark.parametrize("coordinate_head", ["fragment", "cartesian"])
def test_fragment_pair_conditioning_reaches_features_and_labels(
    coordinate_head, tmp_path
):
    torch.manual_seed(41004)
    config = ModelConfig(
        width=32,
        depth=2,
        heads=4,
        feedforward_width=64,
        time_fourier_dim=16,
        coordinate_head=coordinate_head,
    )
    model = MolecularModel(config).eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.modulation[-1].weight, std=0.1)
            torch.nn.init.normal_(block.modulation[-1].bias, std=0.1)
            block.attention.fragment_pair_bias[:, 0] = -0.4
            block.attention.fragment_pair_bias[:, 1] = 0.4
        for name, p in model.named_parameters():
            if "_head.weight" in name:
                torch.nn.init.normal_(p, std=0.1)
    batch = _batch()
    regrouped = replace(batch, fragment_id=torch.tensor([0, 1, 0, 1, 2, 3, 2, 3]))
    features = []
    hook = model.final_norm.register_forward_hook(
        lambda m, a, v: features.append(v.detach().clone())
    )
    original = model(batch)
    changed = model(regrouped)
    hook.remove()
    assert not torch.allclose(features[0], features[1], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(
        original.atom_logits, changed.atom_logits, atol=1e-6, rtol=1e-6
    )

    # Renaming fragment IDs must not convey new information.
    rename = torch.tensor([1, 0, 3, 2])
    renamed = model(replace(batch, fragment_id=rename[batch.fragment_id]))
    names = (
        ("center", "rotation", "internal", "atom_logits")
        if coordinate_head == "fragment"
        else ("displacement", "atom_logits")
    )
    for name in names:
        expected = getattr(original, name)
        if name in ("center", "rotation"):
            expected = expected[rename]
        torch.testing.assert_close(getattr(renamed, name), expected)

    # Atoms may be arbitrarily reordered within each molecule; no order leakage.
    permutation = torch.tensor([3, 0, 2, 1, 5, 7, 4, 6])
    permuted = replace(
        batch,
        **{
            field.name: getattr(batch, field.name)[permutation]
            for field in fields(batch)
            if isinstance(getattr(batch, field.name), torch.Tensor)
            and getattr(batch, field.name).shape[0] == batch.token_capacity
        },
    )
    permuted.atom_index = torch.tensor([73, -8, 44, 2, 9, 70, 15, -6])
    reordered = model(permuted)
    for name in names:
        expected = getattr(original, name)
        if name not in ("center", "rotation"):
            expected = expected[permutation]
        torch.testing.assert_close(
            getattr(reordered, name), expected, atol=2e-5, rtol=2e-5
        )

    original.atom_logits.square().mean().backward()
    for block in model.blocks:
        grad = block.attention.fragment_pair_bias.grad
        assert (
            grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 1e-8
        )
    optimizer = MolecularMuon(
        model.named_parameters(),
        learning_rate=0.02,
        adamw_lr_multiplier=0.1,
        fused_adamw=False,
    )
    pair_names = [
        n for n, _ in model.named_parameters() if n.endswith("fragment_pair_bias")
    ]
    assert len(pair_names) == config.depth
    assert all(n in optimizer.adamw_parameter_names for n in pair_names)
    path = tmp_path / "pair.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=0,
        loss_weights=None,
        training_state={
            "process": (
                {"rotation_noise_profile": "constant-v1", "rotation_variance": 6.0}
                if coordinate_head == "fragment"
                else {
                    "coordinate_process": "atom-brownian-kumaraswamy-v1",
                    "atom_bm_variance": 14.8,
                    "atom_bm_clock_a": 2.3,
                    "atom_bm_clock_b": 3.1,
                }
            )
        },
    )
    loaded, _ = load_checkpoint(path, device="cpu")
    for name in names:
        torch.testing.assert_close(
            getattr(loaded(batch), name), getattr(original, name)
        )
