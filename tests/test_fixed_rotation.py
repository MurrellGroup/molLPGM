from dataclasses import fields

import pytest

from mol_lpgm.batch import MolecularBatch
from mol_lpgm.process import check_rotation_variance


def test_rotation_variance_is_global_and_requires_matching_process_metadata():
    assert "rotation_variance" not in {field.name for field in fields(MolecularBatch)}
    fixed = dict(rotation_noise_profile="constant-v1", rotation_variance=6.0)
    check_rotation_variance(fixed, fixed)
    for wrong in (
        {},
        {**fixed, "rotation_variance": float("nan")},
        {**fixed, "rotation_variance": 0},
        {**fixed, "rotation_variance": 8},
    ):
        with pytest.raises(ValueError):
            check_rotation_variance(wrong, fixed)
        with pytest.raises(ValueError):
            check_rotation_variance(fixed, wrong)


def test_checkpoint_requires_process_metadata(tmp_path):
    import torch
    from mol_lpgm.checkpoint import load_checkpoint

    checkpoint = tmp_path / "incomplete.pt"
    torch.save(
        {
            "format_version": 1,
            "domain": "mol-lpgm",
            "training_domain": "geom-drugs",
            "model_config": {"coordinate_head": "fragment"},
            "training_state": {"process": {}},
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="fixed global rotation variance"):
        load_checkpoint(checkpoint, device="cpu")
