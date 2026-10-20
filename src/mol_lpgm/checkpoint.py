"""Atomic model snapshots shared by training, sampling and published weights."""

from dataclasses import asdict
from pathlib import Path
import re
import torch
from .model import ModelConfig, MolecularModel
from .process import check_rotation_variance
from .atom_bm import validate_profile


def prune_regular_checkpoints(directory: Path, keep: int) -> list[str]:
    """Retain the newest numbered snapshots; preserve named checkpoints."""
    if keep < 0:
        raise ValueError("checkpoint retention must be nonnegative")
    if not keep:
        return []
    paths = sorted(
        p
        for p in directory.iterdir()
        if re.fullmatch(r"checkpoint_\d{9}\.pt", p.name)
        and p.is_file()
        and not p.is_symlink()
    )
    removed = []
    for path in paths[:-keep]:
        path.unlink()
        removed.append(path.name)
    return removed


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    step,
    loss_weights,
    training_state=None,
    training_domain="qm9",
    ema_state=None,
):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        dict(
            format_version=1,
            domain="mol-lpgm",
            training_domain=training_domain,
            step=step,
            model_config=asdict(model.config),
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            optimizer_name="molecular-muon",
            loss_weights=loss_weights,
            training_state=training_state or {},
            ema=ema_state,
        ),
        temporary,
    )
    temporary.replace(destination)


def load_checkpoint(path, *, device, expected_training_domain=None, use_ema=False):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("format_version") != 1 or saved.get("domain") != "mol-lpgm":
        raise ValueError("unsupported checkpoint format")
    domain = saved["training_domain"]
    if domain not in ("qm9", "geom-drugs") or (
        expected_training_domain is not None and domain != expected_training_domain
    ):
        raise ValueError("checkpoint dataset does not match requested dataset")
    if use_ema and saved.get("ema") is None:
        raise ValueError("checkpoint does not contain EMA weights")
    if "process" not in saved.get("training_state", {}):
        raise ValueError("checkpoint does not specify its process")
    process = saved["training_state"]["process"]
    if saved["model_config"]["coordinate_head"] == "fragment":
        check_rotation_variance(process, process)
    else:
        validate_profile(process)
    model = MolecularModel(ModelConfig(**saved["model_config"])).to(device)
    model.load_state_dict(
        saved["ema"]["model"] if use_ema else saved["model"], strict=True
    )
    model.checkpoint_weights = "ema" if use_ema else "raw"
    model.training_process = dict(saved["training_state"]["process"])
    return model, saved
