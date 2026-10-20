import torch
import pytest

from mol_lpgm.ema import ModelEMA
from mol_lpgm.model import ModelConfig, MolecularModel
from mol_lpgm.checkpoint import save_checkpoint, load_checkpoint


def test_constant_ema_and_resume_are_identical():
    model = torch.nn.Linear(2, 1, bias=False)
    model.register_buffer("counter", torch.tensor(0))
    with torch.no_grad():
        model.weight.fill_(2)
    ema = ModelEMA(model, 0.9999)
    with torch.no_grad():
        model.weight.fill_(10)
        model.counter.fill_(7)
    for _ in range(100):
        ema.update(model)
    assert torch.allclose(
        ema.shadow["weight"],
        torch.full_like(model.weight, 10 - 8 * 0.9999**100),
        atol=3e-6,
    )
    assert ema.shadow["counter"].item() == 7
    restored = ModelEMA(model, 0.9999)
    restored.load_state_dict(ema.state_dict())
    for i in range(30):
        with torch.no_grad():
            model.weight.fill_(i)
        ema.update(model)
        restored.update(model)
    assert restored.num_updates == ema.num_updates == 130
    assert torch.equal(restored.shadow["weight"], ema.shadow["weight"])
    with pytest.raises(ValueError, match="decay"):
        ModelEMA(model, 0.99).load_state_dict(ema.state_dict())


def test_checkpoint_loads_raw_or_ema_explicitly(tmp_path):
    model = MolecularModel(
        ModelConfig(
            width=32, depth=1, heads=4, feedforward_width=64, time_fourier_dim=16
        )
    )
    optimizer = torch.optim.AdamW(model.parameters())
    ema = ModelEMA(model, 0.9999)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1)
    ema.update(model)
    path = tmp_path / "checkpoint.pt"
    process = {
        "internal_clock_power": 16,
        "internal_mobility_sigma": 0.15,
        "internal_mobility_fade_start": 0.8,
    }
    process["source_sigma"] = 3.0
    process["rotation_noise_profile"] = "constant-v1"
    process["rotation_variance"] = 6.0
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        step=1,
        loss_weights=None,
        training_domain="geom-drugs",
        ema_state=ema.state_dict(),
        training_state={"process": process},
    )
    raw, checkpoint = load_checkpoint(path, device="cpu")
    averaged, _ = load_checkpoint(path, device="cpu", use_ema=True)
    assert checkpoint["ema"]["num_updates"] == 1
    assert raw.training_process == averaged.training_process
    assert raw.training_process["source_sigma"] == 3.0
    assert averaged.checkpoint_weights == "ema"
    for name, value in raw.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
        assert torch.equal(averaged.state_dict()[name], ema.shadow[name])
    save_checkpoint(path, model=model, optimizer=optimizer, step=1, loss_weights=None)
    with pytest.raises(ValueError, match="does not contain EMA"):
        load_checkpoint(path, device="cpu", use_ema=True)
