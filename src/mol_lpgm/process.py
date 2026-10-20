from __future__ import annotations

import math

import torch

from .batch import MolecularBatch
from .geometry import exp_so3, log_so3
from .structured import _scatter_sum

INFERENCE_TIME_SCHEDULES = ("linear", "sine", "sqrt")


def check_center_spread(actual: dict, expected: dict) -> None:
    """Validate the centroid diffusion profile."""
    for process in (actual, expected):
        sigma = float(process.get("center_spread_sigma", 0.0))
        tau = float(process.get("center_spread_timescale", 0.1))
        if not math.isfinite(sigma) or sigma < 0 or not math.isfinite(tau) or tau <= 0:
            raise ValueError("invalid center spread parameters")
        if sigma and process.get("center_noise_profile") != "early-diffusion-v1":
            raise ValueError("unsupported center noise profile")
    if not math.isclose(
        float(actual.get("center_spread_sigma", 0.0)),
        float(expected.get("center_spread_sigma", 0.0)),
        rel_tol=1e-6,
    ):
        raise ValueError("center spread conflicts with checkpoint/training process")
    if float(actual.get("center_spread_sigma", 0.0)) and not math.isclose(
        float(actual.get("center_spread_timescale", 0.1)),
        float(expected.get("center_spread_timescale", 0.1)),
        rel_tol=1e-6,
    ):
        raise ValueError(
            "center spread timescale conflicts with checkpoint/training process"
        )


def resolve_center_spread(
    sigma: float | None,
    timescale: float | None,
    saved_process: dict | None,
    allow_change: bool = False,
) -> tuple[dict, bool]:
    saved = saved_process or {}
    check_center_spread(saved, saved)
    proposed = dict(
        center_spread_sigma=(
            float(saved.get("center_spread_sigma", 3.0))
            if sigma is None
            else float(sigma)
        ),
        center_spread_timescale=(
            float(saved.get("center_spread_timescale", 0.1))
            if timescale is None
            else float(timescale)
        ),
    )
    if proposed["center_spread_sigma"] > 0:
        proposed["center_noise_profile"] = "early-diffusion-v1"
    check_center_spread(proposed, proposed)
    changed = False
    if saved_process is not None:
        try:
            check_center_spread(proposed, saved)
        except ValueError:
            changed = True
    if changed and not allow_change:
        raise ValueError("center spread differs from checkpoint; start a fresh run")
    return proposed, changed


def center_spread_innovation(
    start: float, stop: float, stop_s: float, sigma: float, timescale: float
) -> float:
    """Additional exact centroid innovation variance before global recentering."""
    if not 0 <= start <= stop <= 1 or not 0 <= stop_s <= 1:
        raise ValueError("invalid bridge interval")
    if (
        not math.isfinite(sigma)
        or sigma < 0
        or not math.isfinite(timescale)
        or timescale <= 0
    ):
        raise ValueError("invalid center spread parameters")
    u, v = start / timescale, stop / timescale
    dh = math.exp(-u * u) * -math.expm1(-(v * v - u * u))
    return sigma * sigma * (1 - stop_s) ** 2 * dh


def resolve_manifold_substep(
    requested: float | None, saved_process: dict | None
) -> float:
    saved = float((saved_process or {}).get("manifold_substep", 0.005))
    step = saved if requested is None else float(requested)
    if not math.isfinite(step) or not 0 < step <= 1:
        raise ValueError("manifold substep must lie in (0, 1]")
    if saved_process is not None and not math.isclose(step, saved, rel_tol=1e-6):
        raise ValueError("manifold_substep conflicts with checkpoint")
    return step


def check_manifold_substep(actual: dict, expected: dict) -> None:
    resolve_manifold_substep(float(actual.get("manifold_substep", 0.005)), expected)


def resolve_source_sigma(
    requested: float | None, saved_process: dict | None, allow_change: bool = False
) -> tuple[float, bool]:
    """Resolve the source standard deviation without changing a resumed process."""
    saved = float((saved_process or {}).get("source_sigma", 3.0))
    sigma = saved if requested is None else float(requested)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("source sigma must be finite and positive")
    changed = saved_process is not None and not math.isclose(sigma, saved, rel_tol=1e-6)
    if changed and not allow_change:
        raise ValueError("source sigma differs from checkpoint; start a fresh run")
    return sigma, changed


def check_source_sigma(actual: dict, expected: dict) -> None:
    sigma, _ = resolve_source_sigma(None, actual)
    expected_sigma, _ = resolve_source_sigma(None, expected)
    if not math.isclose(sigma, expected_sigma, rel_tol=1e-6):
        raise ValueError("source_sigma conflicts with training process")


def validate_internal_mobility(power: float, sigma: float, fade_start: float) -> None:
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("mobility sigma must be finite and nonnegative")
    if sigma > 0 and (
        not math.isfinite(power) or power < 1 or not 0.5 <= fade_start < 1
    ):
        raise ValueError("mobile bridge requires power >= 1 and fade start in [0.5, 1)")


def check_internal_process(actual: dict, expected: dict) -> None:
    """Check the internal bridge parameters against checkpoint metadata."""
    for key, default in (
        ("internal_clock_power", 16.0),
        ("internal_mobility_sigma", 0.15),
        ("internal_mobility_fade_start", 0.8),
    ):
        if (
            key.endswith("fade_start")
            and float(expected.get("internal_mobility_sigma", 0)) == 0
        ):
            continue
        if not math.isclose(
            float(actual.get(key, default)),
            float(expected.get(key, default)),
            rel_tol=1e-6,
        ):
            raise ValueError(f"{key} conflicts with training process")


def internal_log_retention(t: float, power: float) -> float:
    if not 0 <= t <= 1:
        raise ValueError("physical time must lie in [0, 1]")
    return (
        -math.inf
        if t == 1
        else power * (3 * math.log1p(-t) + math.log1p(3 * t + 6 * t * t))
    )


def mobile_internal_variance(
    t: float, variance: float, power: float, sigma: float, fade_start: float
) -> float:
    loga = internal_log_retention(t, power)
    u = max(0.0, (t - fade_start) / (1 - fade_start))
    fade = (1 - u) ** 3 * (1 + 3 * u + 6 * u * u)
    return (
        variance * math.exp(loga) * -math.expm1(loga)
        + sigma * sigma * -math.expm1(2 * loga) * fade
    )


def mobile_internal_coefficients(
    start: float,
    stop: float,
    variance: float,
    power: float,
    sigma: float,
    fade_start: float,
) -> tuple[float, float, float]:
    """Exact transition of dq=k(t)(q1-q)dt + sqrt(V'+2kV)dW, in physical time."""
    validate_internal_mobility(power, sigma, fade_start)
    if not 0 <= start <= stop <= 1:
        raise ValueError("invalid bridge times")
    if stop == 1:
        return 0.0, 1.0, 0.0
    if start == stop:
        return 1.0, 0.0, 0.0
    logratio = internal_log_retention(stop, power) - internal_log_retention(
        start, power
    )
    retention = math.exp(logratio)
    conditional = mobile_internal_variance(
        stop, variance, power, sigma, fade_start
    ) - retention**2 * mobile_internal_variance(
        start, variance, power, sigma, fade_start
    )
    if conditional < -1e-12:
        raise ValueError(f"negative mobile bridge transition variance: {conditional}")
    return retention, -math.expm1(logratio), max(conditional, 0.0)


def inference_time_grid(steps: int, schedule: str = "sqrt") -> list[float]:
    """Endpoint-preserving solver grid in physical time."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if schedule not in INFERENCE_TIME_SCHEDULES:
        raise ValueError(f"unsupported inference schedule: {schedule}")
    uniform = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    if schedule == "sine":
        physical = torch.sin((math.pi / 2.0) * uniform)
    elif schedule == "sqrt":
        physical = torch.sqrt(uniform)
    else:
        physical = uniform
    physical[0] = 0.0
    physical[-1] = 1.0
    return physical.tolist()


def project_document_atoms(coords: torch.Tensor, batch: MolecularBatch) -> torch.Tensor:
    valid = batch.valid_atom[:, None].to(coords.dtype)
    document_sum = _scatter_sum(
        coords * valid, batch.document_id, batch.fragment_capacity
    )
    document_count = _scatter_sum(valid, batch.document_id, batch.fragment_capacity)
    mean = document_sum / document_count.clamp_min(1.0)
    return (coords - mean[batch.document_id]) * valid


def project_internal_noise(noise: torch.Tensor, batch: MolecularBatch) -> torch.Tensor:
    valid = batch.valid_atom[:, None].to(noise.dtype)
    summed = _scatter_sum(noise * valid, batch.fragment_id, batch.fragment_capacity)
    mean = summed / batch.fragment_atom_count.clamp_min(1).to(noise.dtype)[:, None]
    return (noise - mean[batch.fragment_id]) * valid


def euclidean_relative_step(
    target: torch.Tensor,
    noise: torch.Tensor,
    variance: torch.Tensor,
    start_s: float,
    stop_s: float,
) -> torch.Tensor:
    delta_s = stop_s - start_s
    alpha = delta_s / max(1.0 - start_s, 1e-8)
    conditional_variance = (
        variance * delta_s * max(1.0 - stop_s, 0.0) / max(1.0 - start_s, 1e-8)
    )
    return alpha * target + torch.sqrt(conditional_variance.clamp_min(0.0)) * noise


def rotation_relative_step(
    target_vector: torch.Tensor,
    variance: torch.Tensor,
    start_s: float,
    stop_s: float,
    *,
    max_substep: float,
    generator: torch.Generator,
) -> torch.Tensor:
    target = exp_so3(target_vector)
    identity = torch.eye(3, dtype=target_vector.dtype, device=target_vector.device)
    current_rotation = identity.expand(target_vector.shape[0], 3, 3).clone()
    current_s = start_s
    while current_s < stop_s - 1e-8:
        delta_s = min(max_substep, stop_s - current_s)
        noise = torch.randn(
            target_vector.shape,
            device=target_vector.device,
            dtype=target_vector.dtype,
            generator=generator,
        )
        noise = noise * torch.sqrt((variance * delta_s / 2.0).clamp_min(0.0))[:, None]
        noisy = exp_so3(noise) @ current_rotation
        pull = log_so3(target @ noisy.transpose(-1, -2))
        alpha = delta_s / max(1.0 - current_s, 1e-8)
        current_rotation = exp_so3(alpha * pull) @ noisy
        current_s += delta_s
    return current_rotation


def check_rotation_variance(actual: dict, expected: dict) -> None:
    """Require the same fixed angular variance in producer, trainer and sampler."""
    for process in (actual, expected):
        if process.get("rotation_noise_profile") != "constant-v1":
            raise ValueError(
                "fragment process requires a fixed global rotation variance"
            )
        value = float(process.get("rotation_variance", float("nan")))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("global rotation variance must be finite and positive")
    if float(actual["rotation_variance"]) != float(expected["rotation_variance"]):
        raise ValueError("global rotation variance conflicts with checkpoint")
