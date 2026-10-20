from __future__ import annotations

import math


def scheduled_learning_rate(
    step: int,
    *,
    schedule: str,
    peak_learning_rate: float,
    minimum_learning_rate: float,
    start_step: int,
    end_step: int,
    warmup_steps: int = 0,
    warmdown_steps: int = 0,
    exponential_decay_factor: float = 0.5,
) -> float:
    """Return the absolute-step learning rate for a resumable training run."""
    if peak_learning_rate <= 0:
        raise ValueError("peak learning rate must be positive")
    if minimum_learning_rate < 0:
        raise ValueError("minimum learning rate cannot be negative")
    if minimum_learning_rate > peak_learning_rate:
        raise ValueError("minimum learning rate cannot exceed peak learning rate")
    if end_step <= start_step:
        raise ValueError("schedule end step must exceed its start step")
    if warmup_steps < 0 or start_step + warmup_steps >= end_step:
        raise ValueError("warmup must fit strictly inside the schedule interval")
    if warmdown_steps < 0:
        raise ValueError("warmdown steps cannot be negative")
    if schedule not in {
        "constant",
        "cosine",
        "linear-warmdown",
        "exponential-warmdown",
    }:
        raise ValueError(f"unknown learning-rate schedule: {schedule}")
    if schedule in {"linear-warmdown", "exponential-warmdown"}:
        if warmdown_steps <= 0:
            raise ValueError(f"{schedule} requires positive warmdown steps")
        if start_step + warmup_steps > end_step - warmdown_steps:
            raise ValueError("warmup and warmdown intervals overlap")
        if (
            schedule == "exponential-warmdown"
            and not 0.0 < exponential_decay_factor <= 1.0
        ):
            raise ValueError("exponential decay factor must lie in (0, 1]")
    elif warmdown_steps:
        raise ValueError("warmdown steps require a warmdown learning-rate schedule")

    bounded_step = min(max(step, start_step), end_step)
    elapsed = bounded_step - start_step
    if warmup_steps > 0 and elapsed <= warmup_steps:
        return peak_learning_rate * elapsed / warmup_steps
    if schedule == "constant":
        return peak_learning_rate
    if schedule in {"linear-warmdown", "exponential-warmdown"}:
        warmdown_start = end_step - warmdown_steps
        regular_learning_rate = peak_learning_rate
        if schedule == "exponential-warmdown":
            decay_start = start_step + warmup_steps
            decay_steps = warmdown_start - decay_start
            decay_progress = (
                1.0
                if decay_steps <= 0
                else min(
                    max((bounded_step - decay_start) / decay_steps, 0.0),
                    1.0,
                )
            )
            regular_learning_rate *= exponential_decay_factor**decay_progress
        if bounded_step <= warmdown_start:
            return regular_learning_rate
        if bounded_step >= end_step:
            return minimum_learning_rate
        progress = (bounded_step - warmdown_start) / warmdown_steps
        warmdown_start_rate = peak_learning_rate * (
            exponential_decay_factor if schedule == "exponential-warmdown" else 1.0
        )
        return (
            warmdown_start_rate
            + (minimum_learning_rate - warmdown_start_rate) * progress
        )

    decay_start = start_step + warmup_steps
    progress = (bounded_step - decay_start) / (end_step - decay_start)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_learning_rate + (peak_learning_rate - minimum_learning_rate) * cosine


def set_optimizer_learning_rate(optimizer, learning_rate: float) -> None:
    optimizer_setter = getattr(optimizer, "set_learning_rate", None)
    if callable(optimizer_setter):
        optimizer_setter(learning_rate)
        return
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
