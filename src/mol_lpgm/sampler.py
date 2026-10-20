from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .attention import make_document_block_mask
from .batch import MolecularBatch, clone_with_state
from .geometry import apply_rotation, bridge_clock
from .process import (
    euclidean_relative_step,
    mobile_internal_coefficients,
    check_internal_process,
    check_source_sigma,
    check_manifold_substep,
    check_rotation_variance,
    check_center_spread,
    center_spread_innovation,
    inference_time_grid,
    project_document_atoms,
    project_internal_noise,
    rotation_relative_step,
)
from .atom_bm import (
    CARTESIAN_BM,
    PROFILE_KEYS,
    coefficients as atom_bm_coefficients,
    check_coordinate_process,
)
from .structured import (
    StructuredPrediction,
    gauge_project,
    project_document_centers,
    segment_centers_and_residuals,
)


def _beta22_cdf(time: float) -> float:
    return 3.0 * time * time - 2.0 * time * time * time


def _beta22_pdf(time: float) -> float:
    return 6.0 * time * (1.0 - time)


def discrete_flow_step(
    state: torch.Tensor,
    endpoint_logits: torch.Tensor,
    start_time: float,
    stop_time: float,
    *,
    generator: torch.Generator,
    uniform_weight: float = 0.2,
) -> torch.Tensor:
    """PyTorch translation of Flowfusion's DistNoisy DFM Euler step."""
    if not 0.0 <= start_time < stop_time <= 1.0:
        raise ValueError("discrete step requires 0 <= start < stop <= 1")
    classes = endpoint_logits.shape[-1]
    kappa1 = _beta22_cdf(start_time)
    dkappa1 = _beta22_pdf(start_time)
    kappa2 = uniform_weight * (1.0 - kappa1) * kappa1
    dkappa2 = uniform_weight * dkappa1 * (1.0 - 2.0 * kappa1)
    kappa3 = 1.0 - kappa1 - kappa2
    dkappa3 = -(dkappa1 + dkappa2)
    beta = dkappa3 / max(kappa3, 1e-10)
    endpoint = endpoint_logits.float().softmax(dim=-1)
    current = F.one_hot(state, num_classes=classes).float()
    velocity = (
        (dkappa1 - kappa1 * beta) * endpoint
        + (dkappa2 - kappa2 * beta) / classes
        + beta * current
    )
    probabilities = (current + (stop_time - start_time) * velocity).clamp_min(0.0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def sample(
    model,
    template: MolecularBatch,
    *,
    steps: int = 500,
    seed: int = 17,
    time_schedule: str = "sqrt",
    initial_coordinates: torch.Tensor | None = None,
    trajectory_callback: (
        Callable[[int, float, torch.Tensor, torch.Tensor], None] | None
    ) = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample every molecule; an optional centered initial state permits paired runs."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = template.xt.device
    if device.type != "cuda":
        raise ValueError("production sampling requires compiled CUDA FlexAttention")
    process = template.metadata["process"]
    check_coordinate_process(process, getattr(model, "training_process", {}))
    standard_bm = process.get("coordinate_process") in CARTESIAN_BM
    if standard_bm != (model.config.coordinate_head == "cartesian"):
        raise ValueError(
            "standard Brownian sampling requires a Cartesian coordinate head"
        )
    if not standard_bm:
        check_rotation_variance(process, getattr(model, "training_process", {}))
        check_internal_process(process, getattr(model, "training_process", {}))
        check_center_spread(process, getattr(model, "training_process", {}))
    if hasattr(model, "training_process"):
        check_source_sigma(process, model.training_process)
        if not standard_bm:
            check_manifold_substep(process, model.training_process)
    for key, expected in getattr(model, "training_process", {}).items():
        if key in (
            *PROFILE_KEYS,
            "coordinate_process",
            "internal_clock_power",
            "internal_mobility_sigma",
            "internal_mobility_fade_start",
            "source_sigma",
            "manifold_substep",
            "center_spread_sigma",
            "center_spread_timescale",
        ):
            continue
        if process.get(key) != expected:
            raise ValueError(
                f"sampling template {key} conflicts with checkpoint process"
            )
    source_sigma = float(process["source_sigma"])
    center_variance = float(process.get("center_variance", 0))
    manifold_substep = float(process.get("manifold_substep", 0.005))
    mask_class = int(process["mask_class"])

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    coords = torch.randn(
        template.xt.shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    coords = project_document_atoms(source_sigma * coords, template)
    if initial_coordinates is not None:
        if initial_coordinates.shape != coords.shape:
            raise ValueError("initial coordinates must match the packed template")
        if not torch.isfinite(initial_coordinates).all():
            raise ValueError("initial coordinates must be finite")
        coords = initial_coordinates.to(device=device, dtype=torch.float32).clone()
    atom_class = torch.full_like(template.atom_class, mask_class)
    atom_class = torch.where(template.valid_atom, atom_class, template.atom_class)
    if trajectory_callback is not None:
        trajectory_callback(0, 0.0, coords, atom_class)

    block_mask = make_document_block_mask(
        template.document_id,
        heads=model.config.heads,
        block_size=model.config.flex_block_size,
        valid=template.valid_atom,
    )
    physical_grid = inference_time_grid(steps, time_schedule)

    def clock(component: str, time_value: float) -> float:
        return float(
            bridge_clock(
                time_value,
                kind=process[f"{component}_clock"],
                power=float(process[f"{component}_clock_power"]),
            )
        )

    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    for step_index, (start_time, stop_time) in enumerate(
        zip(physical_grid[:-1], physical_grid[1:]), start=1
    ):
        state = clone_with_state(template, coords, atom_class, start_time)
        with autocast:
            raw = model(state, block_mask)
        if not standard_bm:
            prediction, residual, _ = gauge_project(
                StructuredPrediction(
                    center=raw.center, rotation=raw.rotation, internal=raw.internal
                ),
                state,
            )
        if standard_bm:
            noise = torch.randn(
                coords.shape, device=device, dtype=torch.float32, generator=generator
            )
            noise = project_document_atoms(noise, state)
            _, pull, variance = atom_bm_coefficients(start_time, stop_time, process)
            coords = coords + pull * raw.displacement.float() + variance**0.5 * noise
        else:
            centers, _ = segment_centers_and_residuals(coords, state)

            center_noise = torch.randn(
                prediction.center.shape,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            center_noise = project_document_centers(center_noise, state)
            center_delta = euclidean_relative_step(
                prediction.center,
                center_noise,
                coords.new_full((state.fragment_capacity, 1), center_variance),
                clock("center", start_time),
                clock("center", stop_time),
            )
            if float(process.get("center_spread_sigma", 0.0)) > 0:
                extra_noise = torch.randn(
                    prediction.center.shape,
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
                extra_noise = project_document_centers(extra_noise, state)
                extra_variance = center_spread_innovation(
                    start_time,
                    stop_time,
                    clock("center", stop_time),
                    float(process["center_spread_sigma"]),
                    float(process["center_spread_timescale"]),
                )
                center_delta = center_delta + extra_variance**0.5 * extra_noise

            internal_noise = torch.randn(
                prediction.internal.shape,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            internal_noise = project_internal_noise(internal_noise, state)
            if float(process.get("internal_mobility_sigma", 0)) > 0:
                _, pull, variance = mobile_internal_coefficients(
                    start_time,
                    stop_time,
                    float(process["internal_variance"]),
                    float(process["internal_clock_power"]),
                    float(process["internal_mobility_sigma"]),
                    float(process["internal_mobility_fade_start"]),
                )
                internal_delta = (
                    pull * prediction.internal + variance**0.5 * internal_noise
                )
            else:
                internal_delta = euclidean_relative_step(
                    prediction.internal,
                    internal_noise,
                    state.internal_variance[state.fragment_id, None],
                    clock("internal", start_time),
                    clock("internal", stop_time),
                )
            relative_rotation = rotation_relative_step(
                prediction.rotation,
                coords.new_full(
                    (state.fragment_capacity,), float(process["rotation_variance"])
                ),
                clock("rotation", start_time),
                clock("rotation", stop_time),
                max_substep=manifold_substep,
                generator=generator,
            )
            coords = (
                centers[state.fragment_id]
                + center_delta[state.fragment_id]
                + apply_rotation(
                    relative_rotation[state.fragment_id],
                    residual + internal_delta,
                )
            )
        coords = project_document_atoms(coords, state)
        atom_class = discrete_flow_step(
            atom_class,
            raw.atom_logits,
            start_time,
            stop_time,
            generator=generator,
            uniform_weight=float(process["atom_label_uniform_weight"]),
        )
        atom_class = torch.where(template.valid_atom, atom_class, template.atom_class)
        if trajectory_callback is not None:
            trajectory_callback(step_index, stop_time, coords, atom_class)
    return coords, atom_class
