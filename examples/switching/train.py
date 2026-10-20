"""Fit conditional endpoint regression on a fixed pool of simulated observations."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from model import MarginalModel
from process import Process, observations


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("runs/demo"))
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--train-size", type=int, default=8000000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--resume", type=Path)
    ap.add_argument(
        "--time-floor",
        type=float,
        default=0.0,
        help="If positive, weight squared endpoint errors by 1/(floor+1-t)^2",
    )
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(20261016)
    rng = np.random.default_rng(20261016)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    config = dict(
        process=Process().dictionary(),
        steps=args.steps,
        batch_size=args.batch_size,
        training_observations=args.train_size,
        validation_observations=262144,
        width=128,
        depth=4,
        seed=20261016,
        ema_decay=0.999,
        loss="mean((X_t+(1-t)*v_theta(X_t,t)-Z_t*X_1)^2)",
        time_floor=args.time_floor,
        optimizer="Adam",
        schedule_peak_lr=0.002,
        minimum_lr=0.00001,
        resume_lr_cap=0.0002 if args.resume else None,
        resume_checkpoint=str(args.resume.resolve()) if args.resume else None,
        device=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    write_json(args.output / "config.json", config)
    start = time.perf_counter()
    pieces = []
    for n in range(0, args.train_size, 250000):
        batch, _ = observations(min(250000, args.train_size - n), rng)
        pieces.append(batch)
    train = torch.from_numpy(np.concatenate(pieces)).cuda()
    del pieces
    valid_np, _ = observations(262144, np.random.default_rng(920261016))
    valid = torch.from_numpy(valid_np).cuda()
    model = MarginalModel().cuda()
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, fused=True)
    if args.resume:
        parent = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(parent["model"])
        ema.load_state_dict(parent["ema"])
        optimizer.load_state_dict(parent["optimizer"])
    compiled = torch.compile(model, mode="reduce-overhead")
    ema_parameters = list(ema.parameters())
    parameters = list(model.parameters())
    print(
        json.dumps(
            dict(
                event="started",
                parameters=sum(p.numel() for p in parameters),
                data_seconds=time.perf_counter() - start,
                **config,
            )
        ),
        flush=True,
    )
    best = float("inf")
    log = (args.output / "training.jsonl").open("a", buffering=1)

    def validate(candidate):
        with torch.no_grad():
            losses = []
            for chunk in valid.split(8192):
                pred = candidate.endpoint_prediction(chunk[:, :2])
                residual = pred - chunk[:, 2]
                if args.time_floor:
                    residual = residual / (args.time_floor + 1 - chunk[:, 1])
                losses.append(residual.square().mean())
            return torch.stack(losses).mean().item()

    for step in range(1, args.steps + 1):
        progress = max(0, (step - 500) / (args.steps - 500))
        lr = (
            0.00001 + (0.002 - 0.00001) * 0.5 * (1 + math.cos(math.pi * progress))
        ) * min(step / 500, 1)
        if args.resume:
            lr = min(lr, 0.0002)
        optimizer.param_groups[0]["lr"] = lr
        idx = torch.randint(len(train), (args.batch_size,), device="cuda")
        data = train[idx]
        xt = data[:, :2]
        optimizer.zero_grad(set_to_none=True)
        velocity = compiled(xt)
        residual = xt[:, 0] + (1 - xt[:, 1]) * velocity - data[:, 2]
        if args.time_floor:
            residual = residual / (args.time_floor + 1 - xt[:, 1])
        loss = residual.square().mean()
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        with torch.no_grad():
            decay = min(0.999, (1 + step) / (10 + step))
            torch._foreach_mul_(ema_parameters, decay)
            torch._foreach_add_(ema_parameters, parameters, alpha=1 - decay)
        if step % 1000 == 0 or step == 1:
            val, ema_val = validate(model), validate(ema)
            row = dict(
                step=step,
                loss=loss.item(),
                gradient=grad.item(),
                validation=val,
                ema_validation=ema_val,
                lr=lr,
                seconds=time.perf_counter() - start,
            )
            if not all(
                math.isfinite(row[k])
                for k in ["loss", "gradient", "validation", "ema_validation"]
            ):
                raise RuntimeError(f"Nonfinite training: {row}")
            log.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            state = dict(
                model=model.state_dict(),
                ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),
                step=step,
                config=config,
                metrics=row,
            )
            if ema_val < best:
                best = ema_val
                torch.save(state, args.output / "best.pt")
            if step % 10000 == 0 or step == args.steps:
                torch.save(state, args.output / "latest.pt")
    log.close()
    write_json(
        args.output / "training-complete.json",
        dict(
            step=args.steps,
            best_validation=best,
            seconds=time.perf_counter() - start,
            max_memory_mb=torch.cuda.max_memory_allocated() / 1024**2,
        ),
    )


if __name__ == "__main__":
    main()
