from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .attention import make_document_block_mask
from .atom_bm import CARTESIAN_BM, check_coordinate_process, resolve_profile
from .batch import iter_batches, prefetch_batches
from .checkpoint import load_checkpoint, save_checkpoint, prune_regular_checkpoints
from .ema import ModelEMA
from .loss import loss_terms, reduce_loss, binned_loss, resolve_logit_offset_weight
from .model import ModelConfig, MolecularModel
from .optimizer import MolecularMuon
from .process import (
    check_internal_process,
    validate_internal_mobility,
    check_source_sigma,
    resolve_source_sigma,
    check_manifold_substep,
    check_rotation_variance,
    resolve_manifold_substep,
    check_center_spread,
    resolve_center_spread,
)
from .schedule import scheduled_learning_rate
from .weighting import TimeLossWeights, CartesianLossWeights

_METRIC_NAMES = (
    "loss",
    "center_loss",
    "rotation_loss",
    "internal_loss",
    "atom_label_loss",
    "raw_loss",
    "raw_center_loss",
    "raw_rotation_loss",
    "raw_internal_loss",
    "raw_atom_label_loss",
    "logit_offset_loss",
    "raw_logit_offset_loss",
)

_RECIPES = {
    domain: dict(
        steps=800_000,
        seed=50_731_031,
        schedule_end_step=800_000,
        warmup_steps=10_000,
        exponential_decay_factor=0.06720097636955107,
        checkpoint_every=20_000,
        validation_batches=64,
        expected_documents=0,
    )
    for domain in ("qm9", "geom-drugs")
}


def _arguments(default_domain: str = "qm9") -> argparse.Namespace:
    domain_parser = argparse.ArgumentParser(add_help=False)
    domain_parser.add_argument(
        "--domain", choices=tuple(_RECIPES), default=default_domain
    )
    selected, _ = domain_parser.parse_known_args()
    recipe = _RECIPES[selected.domain]
    parser = argparse.ArgumentParser(
        description="Train the molecular LPGM on QM9 or GEOM-Drugs."
    )
    parser.add_argument("--domain", choices=tuple(_RECIPES), default=selected.domain)
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--validation-root", default="")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=recipe["steps"])
    parser.add_argument("--seed", type=int, default=recipe["seed"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--consume", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--save-resume-checkpoint",
        action="store_true",
        help="Save restored state before any resumed optimizer update",
    )
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--loss-weights", required=True)
    parser.add_argument(
        "--allow-loss-weight-change",
        action="store_true",
        help="Allow a new objective on resume and reset best-validation tracking",
    )
    parser.add_argument(
        "--logit-offset-weight",
        type=float,
        default=None,
        help="Weight on per-atom squared mean logit; inherit on resume",
    )
    parser.add_argument(
        "--rotation-variance",
        type=float,
        default=None,
        help="Global angular variance, identical for every fragment",
    )
    parser.add_argument("--atom-bm-profile", default=None)
    parser.add_argument(
        "--coordinate-head", choices=("fragment", "cartesian"), default=None
    )
    parser.add_argument(
        "--manifold-substep",
        type=float,
        default=None,
        help="Training rotation discretization; inherit on resume",
    )
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--adamw-lr-multiplier", type=float, default=None)
    parser.add_argument("--internal-clock-power", type=float, default=None)
    parser.add_argument("--internal-mobility-sigma", type=float, default=None)
    parser.add_argument("--internal-mobility-fade-start", type=float, default=None)
    parser.add_argument(
        "--source-sigma",
        type=float,
        default=None,
        help="Source standard deviation in Å; inherit on resume, otherwise 3",
    )
    parser.add_argument(
        "--center-spread-sigma",
        type=float,
        default=None,
        help="Early centroid diffusion scale in Å; inherit on resume",
    )
    parser.add_argument(
        "--center-spread-timescale",
        type=float,
        default=None,
        help="Early centroid diffusion timescale in physical time",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=None,
        help="Constant EMA decay; zero disables it; resume inherits the saved value",
    )
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-9)
    parser.add_argument("--schedule-start-step", type=int, default=0)
    parser.add_argument(
        "--schedule-end-step", type=int, default=recipe["schedule_end_step"]
    )
    parser.add_argument("--warmup-steps", type=int, default=recipe["warmup_steps"])
    parser.add_argument("--warmdown-steps", type=int, default=100_000)
    parser.add_argument(
        "--exponential-decay-factor",
        type=float,
        default=recipe["exponential_decay_factor"],
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--loss-spike-factor", type=float, default=8.0)
    parser.add_argument("--loss-spike-window", type=int, default=128)
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=2,
        help="Keep this many regular snapshots; 0 keeps all. Special checkpoints are retained.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=recipe["checkpoint_every"]
    )
    parser.add_argument("--validate-every", type=int, default=5_000)
    parser.add_argument(
        "--validation-batches", type=int, default=recipe["validation_batches"]
    )
    parser.add_argument("--validate-at-start", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--prefetch-batches", type=int, default=4)
    parser.add_argument(
        "--expected-documents", type=int, default=recipe["expected_documents"]
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--feedforward-width", type=int, default=2048)
    parser.add_argument("--query-points", type=int, default=4)
    parser.add_argument("--value-points", type=int, default=8)
    parser.add_argument("--time-fourier-dim", type=int, default=128)
    parser.add_argument("--flex-block-size", type=int, default=128)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument(
        "--save-pre-warmdown-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--accumulate", type=int, default=2, help="Microbatches per optimizer update"
    )
    return parser.parse_args()


def _distributed_environment(
    requested_device: str, run_dir: str
) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return torch.device(requested_device), 0, 1
    if world_size > torch.cuda.device_count():
        raise ValueError("distributed world exceeds visible CUDA devices")
    cache_root = os.environ.get(
        "LPGM_TORCHINDUCTOR_CACHE_ROOT",
        str(Path(run_dir).resolve() / ".torchinductor-cache"),
    )
    if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
        cache = Path(cache_root) / f"rank-{local_rank}"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    return device, rank, world_size


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _autocast(device: torch.device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def _gradient_norm(parameters) -> torch.Tensor:
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        return torch.tensor(0.0)
    norms = torch.stack(
        [
            torch.linalg.vector_norm(gradient.detach(), ord=2, dtype=torch.float64)
            for gradient in gradients
        ]
    )
    total = torch.linalg.vector_norm(norms, ord=2)
    torch._assert_async(torch.isfinite(total), "non-finite gradient norm")
    return total


def _check_batch(
    batch,
    expected_domain: str,
    expected_documents: int,
    internal_clock_power: float = 16.0,
    internal_mobility_sigma: float = 0.15,
    internal_mobility_fade_start: float = 0.8,
    source_sigma: float = 3.0,
    manifold_substep: float = 0.005,
    center_spread_sigma: float = 3.0,
    center_spread_timescale: float = 0.1,
    coordinate_process: dict | None = None,
    rotation_variance: float = 6.0,
) -> None:
    if batch.metadata.get("sampling_only", False):
        raise ValueError(
            "sampling-only conditioning templates cannot be used for training"
        )
    if batch.metadata.get("domain") != expected_domain:
        raise ValueError(
            f"training expected {expected_domain!r}, received "
            f"{batch.metadata.get('domain')!r}"
        )
    process = batch.metadata.get("process", {})
    if batch.metadata.get("purpose") == "sampling-only":
        raise ValueError("Sampling templates cannot be used for training")
    check_coordinate_process(process, coordinate_process or {})
    check_source_sigma(process, {"source_sigma": source_sigma})
    if process.get("coordinate_process") not in CARTESIAN_BM:
        center_process, _ = resolve_center_spread(
            center_spread_sigma, center_spread_timescale, None
        )
        check_center_spread(process, center_process)
        check_manifold_substep(process, {"manifold_substep": manifold_substep})
        check_rotation_variance(
            process,
            dict(
                rotation_noise_profile="constant-v1",
                rotation_variance=rotation_variance,
            ),
        )
        if float(process["internal_clock_power"]) != internal_clock_power:
            raise ValueError("batch internal clock conflicts with training process")
        check_internal_process(
            process,
            dict(
                internal_clock_power=internal_clock_power,
                internal_mobility_sigma=internal_mobility_sigma,
                internal_mobility_fade_start=internal_mobility_fade_start,
            ),
        )
    documents = int(batch.metadata["n_documents"])
    if expected_documents > 0 and documents != expected_documents:
        raise ValueError(
            f"expected {expected_documents} documents, received {documents}"
        )


def _block_mask(batch, model: MolecularModel, device: torch.device):
    if device.type != "cuda":
        return None
    return make_document_block_mask(
        batch.document_id,
        heads=model.config.heads,
        block_size=model.config.flex_block_size,
        valid=batch.valid_atom,
    )


def _loss_metrics(loss):
    components = (
        ("coordinate", "atom_label")
        if hasattr(loss, "coordinate")
        else ("center", "rotation", "internal", "atom_label")
    )
    return {
        "loss": loss.total,
        **{k + "_loss": getattr(loss, k) for k in components},
        "raw_loss": loss.raw_total,
        **{"raw_" + k + "_loss": getattr(loss, "raw_" + k) for k in components},
        "logit_offset_loss": loss.logit_offset,
        "raw_logit_offset_loss": loss.raw_logit_offset,
    }


def _global_metrics(loss, batch, world_size: int) -> dict[str, float | int]:
    metrics = _loss_metrics(loss)
    names = tuple(metrics)
    values = torch.stack(
        (
            *[v.detach() for v in metrics.values()],
            *[
                loss.total.new_tensor(float(batch.metadata[k]))
                for k in ("n_documents", "n_atoms", "n_fragments")
            ],
        )
    )
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values[: len(names)] /= world_size
    result = {name: float(values[i]) for i, name in enumerate(names)}
    result.update(
        {
            name: int(values[len(names) + i])
            for i, name in enumerate(("documents", "atoms", "fragments"))
        }
    )
    return result


@torch.no_grad()
def _validate(
    forward_model,
    model: MolecularModel,
    iterator,
    *,
    batches: int,
    device: torch.device,
    weights: TimeLossWeights,
    expected_domain: str,
    expected_documents: int,
    internal_clock_power: float = 16.0,
    internal_mobility_sigma: float = 0.15,
    internal_mobility_fade_start: float = 0.8,
    source_sigma: float = 3.0,
    manifold_substep: float = 0.005,
    logit_offset_weight: float = 0.01,
    center_spread_sigma: float = 3.0,
    center_spread_timescale: float = 0.1,
    coordinate_process: dict | None = None,
    rotation_variance: float = 6.0,
) -> dict[str, float]:
    model.eval()
    totals = {}
    for _ in range(batches):
        batch = next(iterator)
        _check_batch(
            batch,
            expected_domain,
            expected_documents,
            internal_clock_power,
            internal_mobility_sigma,
            internal_mobility_fade_start,
            source_sigma,
            manifold_substep,
            center_spread_sigma,
            center_spread_timescale,
            coordinate_process,
            rotation_variance,
        )
        with _autocast(device):
            prediction = forward_model(batch, _block_mask(batch, model, device))
        reduced = reduce_loss(
            loss_terms(prediction, batch),
            time_weights=weights,
            logit_offset_weight=logit_offset_weight,
        )
        for name, value in _loss_metrics(reduced).items():
            totals[name] = totals.get(name, 0.0) + float(value)
    model.train()
    return {name: value / batches for name, value in totals.items()}


def _resume_loss_state(
    checkpoint: dict | None, weights: TimeLossWeights, allow_change: bool
) -> tuple[dict, bool]:
    if checkpoint is None:
        return {}, False
    saved = checkpoint.get("loss_weights")
    changed = (
        saved is None or TimeLossWeights.from_dict(saved).to_dict() != weights.to_dict()
    )
    if changed and not allow_change:
        raise ValueError(
            "loss weights differ from checkpoint; use --allow-loss-weight-change"
        )
    state = dict(checkpoint.get("training_state", {}))
    if changed:
        state.pop("best_validation_loss", None)
        state.pop("best_validation_step", None)
    return state, changed


def main(default_domain: str = "qm9") -> None:
    args = _arguments(default_domain)
    if not torch.cuda.is_available():
        raise RuntimeError("molecular LPGM training requires CUDA")
    if args.steps <= 0 or args.expected_documents < 0 or args.accumulate < 1:
        raise ValueError("invalid training length or document count")
    if args.loss_spike_factor < 0 or args.loss_spike_window < 1:
        raise ValueError("invalid loss-spike guard")
    device, rank, world_size = _distributed_environment(args.device, args.run_dir)
    primary = rank == 0
    process_seed = args.seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.set_float32_matmul_precision("high")

    run_dir = Path(args.run_dir)
    if primary:
        run_dir.mkdir(parents=True, exist_ok=True)
    _barrier(world_size)

    if args.resume:
        model, checkpoint_data = load_checkpoint(
            args.resume,
            device=device,
            expected_training_domain=args.domain,
        )
        if (
            args.coordinate_head is not None
            and args.coordinate_head != model.config.coordinate_head
        ):
            raise ValueError("coordinate head conflicts with checkpoint")
        start_step = int(checkpoint_data["step"])
    else:
        model = MolecularModel(
            ModelConfig(
                coordinate_head=args.coordinate_head or "fragment",
                width=args.width,
                depth=args.depth,
                heads=args.heads,
                feedforward_width=args.feedforward_width,
                query_points=args.query_points,
                value_points=args.value_points,
                time_fourier_dim=args.time_fourier_dim,
                flex_block_size=args.flex_block_size,
            )
        ).to(device)
        checkpoint_data = None
        start_step = 0
    saved_process = (checkpoint_data or {}).get("training_state", {}).get("process")
    coordinate_process = resolve_profile(args.atom_bm_profile, saved_process)
    args.source_sigma, _ = resolve_source_sigma(args.source_sigma, saved_process)
    training_process = dict(coordinate_process, source_sigma=args.source_sigma)
    is_bm = coordinate_process.get("coordinate_process") in CARTESIAN_BM
    if is_bm != (model.config.coordinate_head == "cartesian"):
        raise ValueError("Coordinate head and conditional process disagree")
    if not is_bm:
        defaults = dict(
            internal_clock_power=16.0,
            internal_mobility_sigma=0.15,
            internal_mobility_fade_start=0.8,
            manifold_substep=0.005,
            rotation_variance=6.0,
            center_spread_sigma=3.0,
            center_spread_timescale=0.1,
        )
        for key, default in defaults.items():
            requested = getattr(args, key)
            value = (
                (saved_process or {}).get(key, default)
                if requested is None
                else requested
            )
            setattr(args, key, float(value))
            training_process[key] = float(value)
        training_process["center_noise_profile"] = "early-diffusion-v1"
        training_process["rotation_noise_profile"] = "constant-v1"
        check_rotation_variance(training_process, saved_process or training_process)
        validate_internal_mobility(
            args.internal_clock_power,
            args.internal_mobility_sigma,
            args.internal_mobility_fade_start,
        )
        if not np.isfinite(args.internal_clock_power) or args.internal_clock_power <= 0:
            raise ValueError("Internal clock power must be finite and positive")
        check_center_spread(training_process, saved_process or training_process)
        check_manifold_substep(training_process, saved_process or training_process)
        check_internal_process(training_process, saved_process or training_process)
    args.logit_offset_weight = resolve_logit_offset_weight(
        args.logit_offset_weight, (checkpoint_data or {}).get("training_state")
    )
    saved_multiplier = float(
        (checkpoint_data or {}).get("optimizer", {}).get("adamw_lr_multiplier", 0.1)
    )
    if args.adamw_lr_multiplier is None:
        args.adamw_lr_multiplier = saved_multiplier
    weights = TimeLossWeights.load(args.loss_weights)
    if isinstance(weights, CartesianLossWeights) != (
        model.config.coordinate_head == "cartesian"
    ):
        raise ValueError("loss weights and coordinate head disagree")
    previous_state, loss_weights_changed = _resume_loss_state(
        checkpoint_data, weights, args.allow_loss_weight_change
    )
    optimizer = MolecularMuon(
        model.named_parameters(),
        learning_rate=args.learning_rate,
        adamw_lr_multiplier=args.adamw_lr_multiplier,
        weight_decay=args.weight_decay,
        momentum=args.muon_momentum,
        fused_adamw=True,
    )
    if checkpoint_data is not None and not args.reset_optimizer:
        if checkpoint_data.get("optimizer_name") not in {
            "molecular-muon",
        }:
            raise ValueError("checkpoint does not contain molecular Muon state")
        optimizer.load_state_dict(checkpoint_data["optimizer"])

    model.train()
    compiled_model = torch.compile(
        model, fullgraph=True, dynamic=False, mode=args.compile_mode
    )
    forward_model = (
        DistributedDataParallel(
            compiled_model,
            device_ids=[device.index],
            output_device=device.index,
            forward_sync_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        if world_size > 1
        else compiled_model
    )
    saved_ema = (checkpoint_data or {}).get("ema")
    if args.ema_decay is None:
        args.ema_decay = float(saved_ema["decay"]) if saved_ema else 0.9999
    if not np.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1:
        raise ValueError("EMA decay must lie in [0, 1)")
    if saved_ema and args.ema_decay != float(saved_ema["decay"]):
        raise ValueError("EMA decay conflicts with checkpoint")
    if args.ema_decay and start_step > 0 and saved_ema is None:
        raise ValueError("cannot resume EMA tracking from a checkpoint without EMA")
    # DDP has now broadcast initial parameters. Only rank zero maintains and
    # saves the EMA; all ranks update the same synchronized raw parameters.
    ema = ModelEMA(model, args.ema_decay) if primary and args.ema_decay else None
    if ema is not None and saved_ema:
        ema.load_state_dict(saved_ema)
        if ema.initialized_step + ema.num_updates != start_step:
            raise ValueError("EMA update count conflicts with checkpoint step")
    iterator = prefetch_batches(
        iter_batches(
            args.batch_root,
            live=args.live,
            consume=args.consume,
            seed=args.seed + rank * 1_000_003,
            pin_memory=args.prefetch_batches > 0,
            expected_domain=args.domain,
        ),
        device=device,
        depth=args.prefetch_batches,
    )
    validation_iterator = (
        prefetch_batches(
            iter_batches(
                args.validation_root,
                shuffle=False,
                pin_memory=args.prefetch_batches > 0,
                expected_domain=args.domain,
            ),
            device=device,
            depth=args.prefetch_batches,
        )
        if args.validation_root and primary
        else None
    )
    best_validation = float(previous_state.get("best_validation_loss", float("inf")))
    best_validation_step = int(previous_state.get("best_validation_step", start_step))
    log_path = run_dir / "metrics.jsonl"
    recent_losses: list[float] = []

    schedule_config = {
        "kind": "exponential-warmdown",
        "peak_learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "start_step": args.schedule_start_step,
        "end_step": args.schedule_end_step,
        "warmup_steps": args.warmup_steps,
        "warmdown_steps": args.warmdown_steps,
        "exponential_decay_factor": args.exponential_decay_factor,
    }
    if primary:
        (run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "arguments": vars(args),
                    "model_config": asdict(model.config),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "loss_weights": weights.to_dict(),
                    "logit_offset_weight": args.logit_offset_weight,
                    "schedule": schedule_config,
                    "process": training_process,
                    "world_size": world_size,
                    "microbatches_per_update": args.accumulate,
                    "gradient_clipping": False,
                    "resume_step": start_step,
                    "loss_weights_changed": loss_weights_changed,
                    "best_validation_reset": loss_weights_changed,
                    "ema_decay": args.ema_decay,
                    "ema_update_every": 1 if args.ema_decay else None,
                    "validation_weights": "raw",
                },
                indent=2,
            )
            + "\n"
        )

    def training_state() -> dict:
        return {
            "best_validation_loss": best_validation,
            "best_validation_step": best_validation_step,
            "schedule": schedule_config,
            "process": training_process,
            "logit_offset_weight": args.logit_offset_weight,
            "objective_change": (
                {"step": start_step, "parent_checkpoint": args.resume}
                if loss_weights_changed
                else previous_state.get("objective_change")
            ),
        }

    def write_checkpoint(path: Path, step: int) -> None:
        if primary:
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                step=step,
                loss_weights=weights.to_dict(),
                training_state=training_state(),
                training_domain=args.domain,
                ema_state=ema.state_dict() if ema is not None else None,
            )

    def validate(step: int) -> None:
        nonlocal best_validation, best_validation_step
        if not args.validation_root:
            return
        _barrier(world_size)
        if primary:
            assert validation_iterator is not None
            metrics = _validate(
                compiled_model,
                model,
                validation_iterator,
                batches=args.validation_batches,
                device=device,
                weights=weights,
                expected_domain=args.domain,
                expected_documents=args.expected_documents,
                internal_clock_power=args.internal_clock_power,
                internal_mobility_sigma=args.internal_mobility_sigma,
                internal_mobility_fade_start=args.internal_mobility_fade_start,
                source_sigma=args.source_sigma,
                manifold_substep=args.manifold_substep,
                logit_offset_weight=args.logit_offset_weight,
                center_spread_sigma=args.center_spread_sigma,
                center_spread_timescale=args.center_spread_timescale,
                coordinate_process=coordinate_process,
                rotation_variance=args.rotation_variance,
            )
            improved = metrics["loss"] < best_validation
            if improved:
                best_validation = metrics["loss"]
                best_validation_step = step
            record = {
                "step": step,
                **{f"validation_{key}": value for key, value in metrics.items()},
                "validation_improved": improved,
                "best_validation_loss": best_validation,
                "best_validation_step": best_validation_step,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if improved:
                write_checkpoint(run_dir / "checkpoint_best.pt", step)
        _barrier(world_size)

    if args.validate_at_start:
        if not args.validation_root:
            raise ValueError("--validate-at-start needs --validation-root")
        validate(start_step)

    if args.resume and args.save_resume_checkpoint:
        write_checkpoint(run_dir / f"checkpoint_resume_{start_step:09d}.pt", start_step)
        _barrier(world_size)

    previous = time.perf_counter()
    last_step = start_step
    time_bin_sums = torch.zeros((3, len(weights.components), 10), device=device)
    time_bin_steps = 0
    for step in range(start_step + 1, args.steps + 1):
        last_step = step
        learning_rate = scheduled_learning_rate(
            step,
            schedule="exponential-warmdown",
            peak_learning_rate=args.learning_rate,
            minimum_learning_rate=args.minimum_learning_rate,
            start_step=args.schedule_start_step,
            end_step=args.schedule_end_step,
            warmup_steps=args.warmup_steps,
            warmdown_steps=args.warmdown_steps,
            exponential_decay_factor=args.exponential_decay_factor,
        )
        optimizer.set_learning_rate(learning_rate)
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for microbatch in range(args.accumulate):
            batch = next(iterator)
            _check_batch(
                batch,
                args.domain,
                args.expected_documents,
                args.internal_clock_power,
                args.internal_mobility_sigma,
                args.internal_mobility_fade_start,
                args.source_sigma,
                args.manifold_substep,
                args.center_spread_sigma,
                args.center_spread_timescale,
                coordinate_process,
                args.rotation_variance,
            )
            sync = (
                forward_model.no_sync()
                if world_size > 1 and microbatch + 1 < args.accumulate
                else nullcontext()
            )
            with sync:
                with _autocast(device):
                    prediction = forward_model(batch, _block_mask(batch, model, device))
                terms = loss_terms(prediction, batch)
                reduced = reduce_loss(
                    terms,
                    time_weights=weights,
                    logit_offset_weight=args.logit_offset_weight,
                )
                if not torch.isfinite(reduced.total):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                (reduced.total / args.accumulate).backward()
            with torch.no_grad():
                time_bin_sums += torch.stack(binned_loss(terms, time_weights=weights))
            time_bin_steps += 1
            for name, value in _global_metrics(reduced, batch, world_size).items():
                divisor = (
                    1
                    if name in ("documents", "atoms", "fragments")
                    else args.accumulate
                )
                metrics[name] = metrics.get(name, 0) + value / divisor
        loss_value = float(metrics["loss"])
        if not np.isfinite(loss_value):
            write_checkpoint(run_dir / f"checkpoint_nonfinite_{step:09d}.pt", step - 1)
            _barrier(world_size)
            raise FloatingPointError(f"non-finite loss at step {step}")
        if args.loss_spike_factor > 0 and len(recent_losses) >= args.loss_spike_window:
            median = float(np.median(recent_losses))
            if loss_value > args.loss_spike_factor * median:
                event = {
                    "step": step,
                    "event": "loss_spike",
                    "loss": loss_value,
                    "rolling_median": median,
                }
                if primary:
                    with log_path.open("a") as handle:
                        handle.write(json.dumps(event) + "\n")
                    print(json.dumps(event), flush=True)
                write_checkpoint(
                    run_dir / f"checkpoint_loss_spike_{step:09d}.pt",
                    step - 1,
                )
                _barrier(world_size)
                raise RuntimeError(f"loss spike at step {step}")
        recent_losses.append(loss_value)
        if len(recent_losses) > args.loss_spike_window:
            del recent_losses[0]

        gradient_norm = _gradient_norm(model.parameters())
        optimizer.step()
        if ema is not None:
            ema.update(model)

        if step % 1000 == 0 or step == args.steps:
            if world_size > 1:
                dist.all_reduce(time_bin_sums)
            time_bin_means = (
                (time_bin_sums / (world_size * time_bin_steps)).cpu().tolist()
            )
            if primary:
                row = {
                    "step": step,
                    "time_bin_steps": time_bin_steps,
                    "time_bin_edges": np.linspace(0, 1, 11).tolist(),
                    "time_binned_loss": time_bin_means[0],
                    "time_binned_raw_loss": time_bin_means[1],
                    "time_binned_occupancy": time_bin_means[2],
                    "component_order": list(weights.components),
                }
                with (run_dir / "time_metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
            time_bin_sums.zero_()
            time_bin_steps = 0

        if primary and (step % args.log_every == 0 or step == start_step + 1):
            now = time.perf_counter()
            record = {
                "step": step,
                **metrics,
                "gradient_norm": float(gradient_norm),
                "learning_rate_muon": learning_rate,
                "learning_rate_adamw": optimizer.adamw_lr_multiplier * learning_rate,
                "step_seconds": (now - previous)
                / (1 if step == start_step + 1 else args.log_every),
                "world_size": world_size,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
                "ema_num_updates": ema.num_updates if ema is not None else 0,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            previous = now

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            write_checkpoint(run_dir / f"checkpoint_{step:09d}.pt", step)
            if primary:
                removed = prune_regular_checkpoints(run_dir, args.keep_last_checkpoints)
                if removed:
                    print(
                        json.dumps({"pruned_regular_checkpoints": removed}), flush=True
                    )
            _barrier(world_size)
            previous = time.perf_counter()
        if (
            args.save_pre_warmdown_checkpoint
            and step == args.schedule_end_step - args.warmdown_steps
        ):
            write_checkpoint(run_dir / f"checkpoint_pre_warmdown_{step:09d}.pt", step)
            _barrier(world_size)
            previous = time.perf_counter()
        if (
            args.validation_root
            and args.validate_every > 0
            and step % args.validate_every == 0
        ):
            validate(step)
            previous = time.perf_counter()

    write_checkpoint(run_dir / "checkpoint_final.pt", last_step)
    _barrier(world_size)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
