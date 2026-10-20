"""Publish a live training-loss figure after each 1,000-update block."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELDS = (
    "loss",
    "center_loss",
    "rotation_loss",
    "internal_loss",
    "atom_label_loss",
    "logit_offset_loss",
)


class LossHistory:
    def __init__(self, interval=1000):
        self.interval = interval
        self.blocks = {}
        self.validation = {}
        self.latest = None
        self.events = []
        self.fields = FIELDS

    def add(self, row):
        if "validation_loss" in row:
            self.validation[row["step"]] = row
        elif "loss" in row and "gradient_norm" in row:
            if self.latest is None and "coordinate_loss" in row:
                self.fields = (
                    "loss",
                    "coordinate_loss",
                    "atom_label_loss",
                    "logit_offset_loss",
                )
            if self.latest and row["step"] <= self.latest["step"]:
                raise ValueError("Training steps must increase monotonically")
            block = self.blocks.setdefault(
                (row["step"] - 1) // self.interval,
                {"count": 0, "sums": {name: 0.0 for name in self.fields}, "last": row},
            )
            block["count"] += 1
            for name in self.fields:
                value = row.get(name, 0.0) if name == "logit_offset_loss" else row[name]
                if not math.isfinite(value):
                    raise ValueError(f'Nonfinite {name} at step {row["step"]}')
                block["sums"][name] += value
            block["last"] = row
            self.latest = row
        elif "event" in row:
            self.events.append(row)

    def completed(self):
        return [
            b for _, b in sorted(self.blocks.items()) if b["count"] == self.interval
        ]


def read_complete_lines(stream, history):
    """Leave an in-progress JSONL record unread until its newline arrives."""
    while True:
        offset = stream.tell()
        line = stream.readline()
        if not line or not line.endswith("\n"):
            stream.seek(offset)
            return
        history.add(json.loads(line))


def render(history, run, output, status):
    if history.latest is None:
        return
    blocks = history.completed()
    validation = [row for _, row in sorted(history.validation.items())]
    step = history.latest["step"]
    utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 1]},
        layout="constrained",
    )
    x = [b["last"]["step"] for b in blocks]
    means = {
        name: [b["sums"][name] / b["count"] for b in blocks] for name in history.fields
    }
    axes[0].plot(
        x,
        means["loss"],
        label=f"Training: mean of each {history.interval:,} updates",
        color="#126e82",
        lw=2,
    )
    axes[0].plot(
        [r["step"] for r in validation],
        [r["validation_loss"] for r in validation],
        "o-",
        label="Validation: 64 fixed batches",
        color="#b46b1c",
        lw=1.6,
        ms=4,
    )
    axes[0].set(ylabel="Total weighted loss", yscale="log")
    axes[0].legend(loc="upper right", fontsize=9)
    colors = ["#3577a4", "#965dc0", "#18866a", "#d17042", "#71717a"]
    for name, color in zip(history.fields[1:], colors):
        if (
            name == "logit_offset_loss"
            and not any(means[name])
            and not any(r.get("validation_" + name, 0.0) for r in validation)
        ):
            continue
        label = name.removesuffix("_loss").replace("_", " ")
        axes[1].plot(x, means[name], label=label, color=color, lw=1.8)
        axes[1].plot(
            [r["step"] for r in validation],
            [r.get("validation_" + name, 0.0) for r in validation],
            "o",
            color=color,
            ms=3,
        )
    axes[1].set(ylabel="Weighted components", yscale="log")
    axes[1].legend(loc="upper right", ncols=2, fontsize=9)
    axes[1].set_title(
        "Lines: training block means · dots: validation", fontsize=10, loc="left"
    )
    lr_rows = [b["last"] for b in blocks]
    if not lr_rows or lr_rows[-1]["step"] != step:
        lr_rows.append(history.latest)
    for name, color in [("muon", "#126e82"), ("adamw", "#b46b1c")]:
        axes[2].plot(
            [r["step"] for r in lr_rows],
            [r["learning_rate_" + name] for r in lr_rows],
            label=name.title(),
            color=color,
            lw=1.5,
        )
    axes[2].set(ylabel="Learning rate", xlabel="Training updates", yscale="log")
    axes[2].legend(loc="upper right", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        config_path = run / "run_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        # A same-objective restart may retain pre-resume metric history.
        first_logged = min(
            (b["last"]["step"] - b["count"] for b in history.blocks.values()),
            default=config.get("resume_step", 0),
        )
        ax.set_xlim(left=min(config.get("resume_step", 0), first_logged))
        ax.ticklabel_format(axis="x", style="plain")
    fig.suptitle(f"{run.name}\n{status} · latest update {step:,} · {utc}", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        destination = output.with_suffix(suffix)
        temporary = destination.with_name(destination.name + ".tmp")
        fig.savefig(temporary, format=suffix[1:], dpi=160)
        os.replace(temporary, destination)
    plt.close(fig)
    data = {
        "run": str(run),
        "status": status,
        "updated_at_utc": utc,
        "latest_step": step,
        "complete_blocks_through": x[-1] if x else 0,
        "update_interval": history.interval,
        "block_end_steps": x,
        "training_block_means": means,
        "validation": validation,
        "events": history.events,
        "last_training_record": history.latest,
        "note": "This run/objective only. Incomplete training blocks are omitted from the means.",
        "resume_step": config.get("resume_step", 0),
        "loss_weights_changed": config.get("loss_weights_changed", False),
        "process": config.get("process", {}),
    }
    destination = output.with_suffix(".json")
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, destination)
    render_time_bins(run, output, status)
    print(
        json.dumps(
            {
                key: data[key]
                for key in (
                    "status",
                    "updated_at_utc",
                    "latest_step",
                    "complete_blocks_through",
                )
            }
        ),
        flush=True,
    )


def render_time_bins(run, output, status):
    path = run / "time_metrics.jsonl"
    if not path.exists():
        return
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines(keepends=True)
        if line.endswith("\n")
    ]
    if not rows:
        return
    from matplotlib.colors import LogNorm

    row = rows[-1]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), layout="constrained")
    for ax, key, title in zip(
        axes,
        ("time_binned_loss", "time_binned_raw_loss"),
        (
            "Weighted loss contribution",
            "Raw loss contribution (different physical units)",
        ),
    ):
        values = np.asarray(row[key])
        im = ax.imshow(values, aspect="auto", norm=LogNorm(), cmap="viridis")
        for i in range(len(row["component_order"])):
            for j in range(10):
                ax.text(
                    j,
                    i,
                    f"{values[i,j]:.3g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if im.norm(values[i, j]) > 0.6 else "white",
                )
        ax.set_yticks(range(len(row["component_order"])), row["component_order"])
        ax.set_xticks(range(10), [f"{i/10:.1f}–{(i+1)/10:.1f}" for i in range(10)])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[-1].set_xlabel("Physical time t · bins sum to the component loss")
    fig.suptitle(
        f'{run.name} · {status}\nUpdate {row["step"]:,} · {row["time_bin_steps"]:,} microbatches, averaged over training microbatches'
    )
    for suffix in (".png", ".svg"):
        destination = output.with_name(output.name + "-bins").with_suffix(suffix)
        temporary = destination.with_name(destination.name + ".tmp")
        fig.savefig(temporary, format=suffix[1:], dpi=160)
        os.replace(temporary, destination)
    plt.close(fig)


def process_identity(pid):
    try:
        # Start time protects the watcher from PID reuse.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Output basename, without extension"
    )
    parser.add_argument("--interval", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--trainer-pid", type=int)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or args.poll_seconds <= 0:
        raise ValueError("Interval and poll period must be positive")
    run = args.run_dir.resolve()
    lock = args.output.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = process_identity(args.trainer_pid) if args.trainer_pid else None
    history = LossHistory(args.interval)
    previous = None
    with (run / "metrics.jsonl").open() as stream:
        while True:
            read_complete_lines(stream, history)
            final = (run / "checkpoint_final.pt").is_file()
            stopped = (
                args.trainer_pid and process_identity(args.trainer_pid) != identity
            )
            status = (
                "Completed" if final else "Trainer stopped" if stopped else "Training"
            )
            fingerprint = (
                history.latest is not None,
                len(history.completed()),
                len(history.validation),
                len(history.events),
                status,
            )
            if fingerprint != previous or args.once:
                render(history, run, args.output, status)
                previous = fingerprint
            if args.once or final or stopped:
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
