"""Train, calibrate component/time weights, then continue the same optimizer and EMA."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def flags(values):
    result = []
    for key, value in values.items():
        key = key.replace("_", "-")
        if isinstance(value, bool):
            result.append("--" + ("" if value else "no-") + key)
        else:
            result.extend(("--" + key, str(value)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--producers", type=int, default=4)
    p.add_argument("--queue-limit", type=int, default=256)
    p.add_argument("--gpus", type=int, default=1)
    args = p.parse_args()
    cfg = tomllib.loads(args.config.read_text())
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / "pipeline.json").exists():
        raise FileExistsError(
            "Run already exists; inspect its recorded phase before recovery"
        )
    training = dict(cfg["training"])
    batch_seeds = cfg.get("batch_seeds", {})
    process = dict(cfg["process"])
    if "atom_bm_profile" in process:
        process["atom_bm_profile"] = str(ROOT / process["atom_bm_profile"])
    steps = training.pop("steps")
    balance_at = training.pop("balance_at")
    fit = training.pop("calibration_batches")
    holdout = training.pop("holdout_batches")
    if not 0 < balance_at < steps or args.producers < 1:
        raise ValueError("Invalid stage boundaries or producer count")
    if args.gpus < 1 or training["accumulate"] * args.gpus != 2:
        raise ValueError("Calibration uses two microbatches per effective batch")
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        JULIA_NUM_THREADS="1",
    )
    state = dict(config=cfg, phase="preparing", started_at=time.time())

    def status(phase, **values):
        state.update(phase=phase, **values)
        atomic_json(run / "pipeline.json", state)

    children = []

    def spawn(argv, log):
        handle = log.open("a")
        child = subprocess.Popen(
            list(map(str, argv)),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=ROOT,
            start_new_session=True,
        )
        handle.close()
        children.append(child)
        return child

    def stop(child):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

    def wait(child, guards=()):
        while child.poll() is None:
            if any(g.poll() is not None for g in guards):
                stop(child)
                raise RuntimeError("Batch producer stopped; inspect producer logs")
            time.sleep(2)
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, child.args)

    def writer(output, split, seed, **extra):
        options = dict(
            domain=cfg["dataset"],
            corpus_dir=str(args.corpus.resolve()),
            out_dir=str(output),
            split=split,
            seed=seed,
            token_capacity=4096,
            fragment_capacity=1365,
            max_documents=512,
            minimum_fill_fraction=0.95,
            **process,
            **extra,
        )
        return [
            "julia",
            f'--project={ROOT/"julia"}',
            ROOT / "julia/scripts/write_batches.jl",
            *[
                f'--{k.replace("_","-")}={str(v).lower() if isinstance(v,bool) else v}'
                for k, v in options.items()
            ],
        ]

    validation = run / "validation"
    weights = ROOT / cfg["loss_weights"]

    def stage(name, end, resume=None):
        directory = run / name
        directory.mkdir()
        queue = run / "queue"
        queue.mkdir(exist_ok=True)
        producers = [
            spawn(
                writer(
                    queue,
                    "train",
                    batch_seeds.get(name, training["seed"] + end),
                    continuous=True,
                    queue_limit=args.queue_limit,
                    shards=args.producers,
                    shard_index=i + 1,
                ),
                run / f"producer-{name}-{i+1}.log",
            )
            for i in range(args.producers)
        ]
        options = dict(
            domain=cfg["dataset"],
            batch_root=queue,
            validation_root=validation,
            run_dir=directory,
            steps=end,
            schedule_end_step=steps,
            loss_weights=weights,
            coordinate_head="cartesian" if cfg["model"] == "bm" else "fragment",
            **training,
            **cfg["architecture"],
            **process,
        )
        command = [
            sys.executable,
            *(
                ["-m", "torch.distributed.run", "--standalone",
                 f"--nproc-per-node={args.gpus}"]
                if args.gpus > 1 else []
            ),
            "-m",
            "mol_lpgm.train",
            *flags(options),
            "--live",
            "--consume",
        ]
        if resume:
            command += [
                "--resume",
                str(resume),
                "--allow-loss-weight-change",
                "--save-resume-checkpoint",
                "--validate-at-start",
            ]
        trainer = spawn(command, run / f"{name}.log")
        status(name, trainer_pid=trainer.pid, producer_pids=[p.pid for p in producers])
        watcher = None
        try:
            while trainer.poll() is None and not (directory / "metrics.jsonl").exists():
                if any(p.poll() is not None for p in producers):
                    raise RuntimeError("Producer failed during startup")
                time.sleep(2)
            if resume:
                restored = directory / f"checkpoint_resume_{balance_at:09d}.pt"
                while trainer.poll() is None and not restored.exists():
                    if any(p.poll() is not None for p in producers):
                        raise RuntimeError("Producer failed during resume")
                    time.sleep(2)
                if not restored.exists():
                    raise RuntimeError("Trainer did not save restored state")
                wait(
                    spawn(
                        [
                            sys.executable,
                            ROOT / "scripts/verify_resume.py",
                            resume,
                            restored,
                            "--output",
                            run / "handoff.json",
                        ],
                        run / "handoff.log",
                    ),
                    [trainer, *producers],
                )
                # Retain the stage-boundary state; discard this run's redundant snapshots.
                restored.unlink()
                for snapshot in resume.parent.glob("checkpoint*.pt"):
                    if snapshot != resume:
                        snapshot.unlink()
            if trainer.poll() is None:
                watcher = spawn(
                    [
                        sys.executable,
                        ROOT / "scripts/watch_loss.py",
                        "--run-dir",
                        directory,
                        "--output",
                        run / "loss",
                        "--trainer-pid",
                        trainer.pid,
                    ],
                    run / f"plot-{name}.log",
                )
            wait(trainer, producers)
            if watcher:
                wait(watcher)
        finally:
            for child in producers:
                stop(child)
            if watcher:
                stop(watcher)
        return directory / "checkpoint_final.pt"

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        wait(
            spawn(
                writer(
                    validation,
                    "valid",
                    batch_seeds.get("validation", training["seed"] + 1),
                    batches=training["validation_batches"],
                ),
                run / "validation.log",
            )
        )
        initial = stage("initial", balance_at)
        status("calibration")
        batches = run / "calibration-batches"
        wait(
            spawn(
                writer(
                    batches, "train",
                    batch_seeds.get("calibration", training["seed"] + 2),
                    batches=fit + holdout,
                ),
                run / "calibration-data.log",
            )
        )
        calibration = run / "calibration"
        wait(
            spawn(
                [
                    sys.executable,
                    ROOT / "scripts/calibrate_loss_balance.py",
                    "--checkpoint",
                    initial,
                    "--batch-root",
                    batches,
                    "--loss-weights",
                    weights,
                    "--output",
                    calibration,
                    "--calibration-batches",
                    fit,
                    "--holdout-batches",
                    holdout,
                    "--gradient-storage",
                    "cpu",
                ],
                run / "calibration.log",
            )
        )
        report = json.loads((calibration / "report.json").read_text())

        def spread(summary):
            losses = [v for row in summary["loss_by_time"].values() for v in row]
            gradients = [
                v for row in summary["full_gradient_by_time"].values() for v in row
            ]
            joint = [l**0.25 * g**0.75 for l, g in zip(losses, gradients)]
            return max(joint) / min(joint)

        before, after = map(spread, [report["holdout_before"], report["holdout_after"]])
        passed = math.isfinite(after) and after < before
        atomic_json(
            calibration / "acceptance.json",
            dict(before=before, after=after, passed=passed),
        )
        if not passed:
            raise RuntimeError(
                "Held-out loss/gradient calibration did not improve balance"
            )
        weights = calibration / "weights.json"
        final = stage("balanced", steps, resume=initial)
        status("completed", final_checkpoint=str(final), completed_at=time.time())
    except BaseException as exc:
        status(
            "failed", failed_phase=state["phase"], error=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        for child in children:
            stop(child)


if __name__ == "__main__":
    main()
