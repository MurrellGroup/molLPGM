"""Measure objective gradients and calibrate fixed component multipliers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from mol_lpgm.batch import load_batch
from mol_lpgm.checkpoint import load_checkpoint
from mol_lpgm.loss import loss_terms, binned_loss
from mol_lpgm.train import _block_mask
from mol_lpgm.atom_bm import check_coordinate_process
from mol_lpgm.process import (
    check_internal_process,
    check_source_sigma,
    check_manifold_substep,
    check_rotation_variance,
    check_center_spread,
)
from mol_lpgm.weighting import TimeLossWeights, BinnedMultipliers, CartesianLossWeights

COMPONENTS = ("center", "rotation", "internal", "atom_label")
HEADS = tuple(name + "_head." for name in COMPONENTS)


def gram(gradients, length, device=None):
    # Chunked matrix product avoids a second 12 GB copy of all 40 gradients.
    device = gradients[0].device if device is None else device
    result = torch.zeros(
        (len(gradients), len(gradients)), device=device, dtype=torch.float64
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, length, 262144):
            block = torch.stack(
                [g[start : min(start + 262144, length)] for g in gradients]
            ).to(device)
            result += (block @ block.T).double()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return result.cpu().numpy()


def summarize(records, multipliers=None, components=COMPONENTS):
    weights = (
        np.ones(len(records[0]["losses"]))
        if multipliers is None
        else np.asarray(multipliers)
    )
    bins = len(weights) // len(components)
    losses = np.mean([r["losses"] for r in records], axis=0) * weights
    shared = np.mean([r["shared_gram"] for r in records], axis=0)
    full = np.mean([r["full_gram"] for r in records], axis=0)
    shared_norms = np.sqrt(np.maximum(np.diag(shared), 0)) * weights
    full_norms = np.sqrt(np.maximum(np.diag(full), 0)) * weights
    aggregate = np.zeros((len(components), len(weights)))
    for i in range(len(components)):
        aggregate[i, i * bins : (i + 1) * bins] = weights[i * bins : (i + 1) * bins]
    cells = lambda a: dict(zip(components, a.reshape(len(components), bins).tolist()))
    occupancy = np.mean([r["occupancy"] for r in records], axis=0)
    return {
        "global_batches": len(records),
        "losses": dict(
            zip(components, losses.reshape(len(components), bins).sum(1).tolist())
        ),
        "loss_by_time": cells(losses),
        "occupancy_by_time": cells(occupancy),
        "conditional_loss_by_time": cells(losses / occupancy),
        "shared_gradient_rms_norms": dict(
            zip(components, np.sqrt(np.diag(aggregate @ shared @ aggregate.T)).tolist())
        ),
        "full_gradient_rms_norms": dict(
            zip(components, np.sqrt(np.diag(aggregate @ full @ aggregate.T)).tolist())
        ),
        "shared_gradient_by_time": cells(shared_norms),
        "full_gradient_by_time": cells(full_norms),
        "combined_full_gradient_rms_norm": float(np.sqrt(weights @ full @ weights)),
        "combined_shared_gradient_rms_norm": float(np.sqrt(weights @ shared @ weights)),
        "loss_max_min_ratio": float(losses.max() / losses.min()),
        "shared_gradient_max_min_ratio": float(shared_norms.max() / shared_norms.min()),
        "full_gradient_max_min_ratio": float(full_norms.max() / full_norms.min()),
        "shared_gram": shared.tolist(),
        "full_gram": full.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--loss-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-batches", type=int, default=64)
    parser.add_argument("--holdout-batches", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--loss-fraction", type=float, default=0.25)
    parser.add_argument(
        "--gradient-storage",
        choices=("cpu", "cuda"),
        default="cuda",
        help="CPU stores full precision cell gradients off GPU for larger models",
    )
    parser.add_argument(
        "--reuse-measurements",
        type=Path,
        help="Recompute weights from an existing measurement directory, without GPU work",
    )
    args = parser.parse_args()
    for n in (args.calibration_batches, args.holdout_batches):
        if n <= 0 or n % 2:
            raise ValueError(
                "Batch counts must be positive and even: two local batches per global batch"
            )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if args.reuse_measurements:
        report = json.loads((args.reuse_measurements / "report.json").read_text())
        records = [
            json.loads(line)
            for line in (args.reuse_measurements / "batches.jsonl")
            .read_text()
            .splitlines()
        ]
        report["measurement_source"] = str(args.reuse_measurements.resolve())
        finalize(args, records, report)
        return
    paths = sorted(args.batch_root.glob("batch_*/metadata.toml"))
    total = args.calibration_batches + args.holdout_batches
    if len(paths) != total:
        raise ValueError(f"Expected exactly {total} batches, found {len(paths)}")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)
    model, saved = load_checkpoint(
        args.checkpoint, device=device, expected_training_domain=None
    )
    step = int(saved["step"])
    logit_offset_weight = float(
        saved.get("training_state", {}).get("logit_offset_weight", 0.0)
    )
    standard = model.config.coordinate_head == "cartesian"
    if not standard:
        assert saved["training_state"]["process"]["internal_clock_power"] == 16
    domain = saved["training_domain"]
    del saved
    model.train()
    forward = torch.compile(model, fullgraph=True, dynamic=False)
    weights = TimeLossWeights.load(args.loss_weights)
    components = weights.components
    heads = tuple(name + "_head." for name in components)
    existing = (
        weights.values
        if standard
        else (weights.time_multipliers.values if weights.time_multipliers else None)
    )
    if existing is not None and len(existing[0]) != args.bins:
        raise ValueError("Recalibration must retain the existing time-bin boundaries")
    named = list(model.named_parameters())
    parameters = [p for _, p in named]
    shared_count = sum(p.numel() for name, p in named if not name.startswith(heads))
    head_tensors = 2 * len(components)
    assert all(not n.startswith(heads) for n, _ in named[:-head_tensors])
    assert all(n.startswith(heads) for n, _ in named[-head_tensors:])
    records = []
    time_counts = np.zeros(args.bins, dtype=np.int64)
    document_count = 0
    started = time.monotonic()
    cells = [(name, b) for name in components for b in range(args.bins)]
    for pair_index in range(total // 2):
        accumulators = None
        losses = np.zeros(len(cells))
        raw_losses = np.zeros(len(cells))
        occupancy = np.zeros(len(cells))
        for path in paths[2 * pair_index : 2 * pair_index + 2]:
            batch = load_batch(path.parent, expected_domain=domain).to(device)
            if not standard:
                assert batch.metadata["process"]["internal_clock_power"] == 16
                check_internal_process(
                    batch.metadata["process"], model.training_process
                )
                check_manifold_substep(
                    batch.metadata["process"], model.training_process
                )
                check_center_spread(batch.metadata["process"], model.training_process)
                check_rotation_variance(batch.metadata["process"], model.training_process)
            check_source_sigma(batch.metadata["process"], model.training_process)
            check_coordinate_process(batch.metadata["process"], model.training_process)
            document_count += batch.metadata["n_documents"]
            times = (
                (
                    batch.time_token[batch.valid_atom]
                    if standard
                    else batch.time_fragment[batch.valid_fragment]
                )
                .cpu()
                .numpy()
            )
            time_counts += np.histogram(times, bins=np.linspace(0, 1, args.bins + 1))[0]
            mask = _block_mask(batch, model, device)
            if accumulators is None:
                accumulators = [None] * len(cells)
            for index, (name, bin_index) in enumerate(cells):
                # Fresh forward per VJP avoids retained compiled backward
                # buffers. This model has no dropout or other random layers.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = forward(batch, mask)
                weighted, raw, mass = binned_loss(
                    loss_terms(prediction, batch), time_weights=weights, bins=args.bins
                )
                value = weighted.reshape(-1)[index]
                gradients = torch.autograd.grad(value, parameters, allow_unused=True)
                flat = torch.cat(
                    [
                        (g if g is not None else torch.zeros_like(p))
                        .detach()
                        .reshape(-1)
                        for g, p in zip(gradients, parameters)
                    ]
                )
                if not torch.isfinite(flat).all() or not torch.isfinite(value):
                    raise FloatingPointError(f"Nonfinite {name} gradient or loss")
                if args.gradient_storage == "cpu":
                    flat = flat.cpu()
                if accumulators[index] is None:
                    accumulators[index] = flat.mul_(0.5)
                else:
                    accumulators[index].add_(flat, alpha=0.5)
                losses[index] += float(value.detach()) / 2
                raw_losses[index] += float(raw.reshape(-1)[index].detach()) / 2
                occupancy[index] += float(mass.reshape(-1)[index]) / 2
                del gradients, prediction, weighted, raw, mass, value, flat
            del batch, mask
        record = {
            "global_batch": pair_index + 1,
            "split": (
                "calibration"
                if 2 * pair_index < args.calibration_batches
                else "holdout"
            ),
            "losses": losses.tolist(),
            "raw_losses": raw_losses.tolist(),
            "occupancy": occupancy.tolist(),
            "shared_gram": gram(accumulators, shared_count, device=device).tolist(),
            "full_gram": gram(
                accumulators, sum(p.numel() for p in parameters), device=device
            ).tolist(),
        }
        records.append(record)
        del accumulators
        with (args.output / "batches.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            json.dumps(
                {
                    "global_batch": pair_index + 1,
                    "total": total // 2,
                    "losses": dict(
                        zip(
                            components,
                            losses.reshape(len(components), args.bins).sum(1).tolist(),
                        )
                    ),
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    with args.checkpoint.open("rb") as f:
        checkpoint_sha = hashlib.file_digest(f, "sha256").hexdigest()
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "checkpoint_sha256": checkpoint_sha,
        "component_order": components,
        "training_process": model.training_process,
        "checkpoint_weights": "raw",
        "bins": args.bins,
        "cell_order": cells,
        "batch_root": str(args.batch_root.resolve()),
        "local_batch_tokens": 4096,
        "local_batches_per_global_batch": 2,
        "documents": document_count,
        "time_bin_edges": np.linspace(0, 1, args.bins + 1).tolist(),
        (
            "atoms_per_time_bin" if standard else "fragments_per_time_bin"
        ): time_counts.tolist(),
        "gradient_parameter_count": sum(p.numel() for p in parameters),
        "shared_gradient_parameter_count": shared_count,
        "gradient_statistic": "RMS norm of per-objective gradients after averaging two 4096-token local batches, matching the effective training batch; before Muon/AdamW preconditioning",
        "base_weights": weights.to_dict(),
        "gradient_storage": args.gradient_storage,
        "fixed_auxiliary": {
            "logit_offset_weight": logit_offset_weight,
            "note": f"Mean-logit gauge penalty stays fixed; balance the {len(cells)} coordinate/label cells only",
        },
        "after_statistics": "Exact linear rescaling of measured loss gradients at this fixed checkpoint, including cross-component inner products",
        "seconds": time.monotonic() - started,
    }
    finalize(args, records, report)


def finalize(args, records, report):
    if not 0 <= args.loss_fraction <= 1:
        raise ValueError("loss fraction must be between zero and one")
    calibration = [r for r in records if r["split"] == "calibration"]
    holdout = [r for r in records if r["split"] == "holdout"]
    components = tuple(report["component_order"])
    before = summarize(calibration, components=components)
    loss = np.array(list(before["loss_by_time"].values())).reshape(-1)
    grad = np.array(list(before["full_gradient_by_time"].values())).reshape(-1)
    if not np.all(loss > 0) or not np.all(grad > 0):
        raise ValueError(
            "Every component/time cell needs nonzero measured loss and gradient"
        )
    relative = 1 / (loss**args.loss_fraction * grad ** (1 - args.loss_fraction))
    full = np.asarray(before["full_gram"])
    multipliers = relative * np.sqrt(full.sum() / (relative @ full @ relative))
    report.update(
        {
            "loss_fraction": args.loss_fraction,
            "gradient_fraction": 1 - args.loss_fraction,
            "gradient_basis": "full model (including task heads); shared-trunk norms also reported",
            "rule": "w_(component,bin) = c / (mean(L_(component,bin))^alpha * RMS(||grad_full L_(component,bin)||)^(1-alpha)); alpha=loss_fraction; full-batch denominators; c preserves combined full-model RMS gradient norm, including gradient cross-products",
            "relative_multipliers": dict(
                zip(
                    components,
                    multipliers.reshape(len(components), report["bins"]).tolist(),
                )
            ),
            "calibration_before": before,
            "calibration_after": summarize(calibration, multipliers, components),
            "holdout_before": summarize(holdout, components=components),
            "holdout_after": summarize(holdout, multipliers, components),
        }
    )
    config = compose_weights(
        report["base_weights"], multipliers.reshape(len(components), report["bins"])
    )
    report["absolute_multipliers"] = (
        config["components"]
        if config["kind"] == "cartesian_brownian_endpoint"
        else config["time_multipliers"]["components"]
    )
    config["calibration"] = {
        "report": str((args.output / "report.json").resolve()),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "rule": report["rule"],
        "loss_fraction": args.loss_fraction,
    }
    (args.output / "weights.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "calibrated",
                "output": str(args.output),
                "loss_fraction": args.loss_fraction,
                "holdout_losses_before": report["holdout_before"]["losses"],
                "holdout_losses_after": report["holdout_after"]["losses"],
                "holdout_gradients_before": report["holdout_before"][
                    "full_gradient_rms_norms"
                ],
                "holdout_gradients_after": report["holdout_after"][
                    "full_gradient_rms_norms"
                ],
            }
        ),
        flush=True,
    )


def compose_weights(base: dict, relative: np.ndarray) -> dict:
    """Apply measured factors to the objective actually used for measurement."""
    weights = TimeLossWeights.from_dict(base)
    if isinstance(weights, CartesianLossWeights):
        if np.asarray(weights.values).shape != relative.shape:
            raise ValueError("Recalibration bin shape differs from existing weights")
        return CartesianLossWeights(
            tuple(map(tuple, (np.asarray(weights.values) * relative).tolist())),
            weights.coordinate_epsilon,
        ).to_dict()
    old = (
        np.ones_like(relative)
        if weights.time_multipliers is None
        else np.asarray(weights.time_multipliers.values)
    )
    if old.shape != relative.shape:
        raise ValueError("Recalibration bin shape differs from existing weights")
    result = weights.to_dict()
    result["format_version"] = 2
    result["time_multipliers"] = BinnedMultipliers(
        tuple(map(tuple, (old * relative).tolist()))
    ).to_dict()
    return result


if __name__ == "__main__":
    main()
