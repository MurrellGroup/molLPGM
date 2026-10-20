from __future__ import annotations

import torch


def hat(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        (*vector.shape[:-1], 3, 3)
    )


def exp_so3(vector: torch.Tensor) -> torch.Tensor:
    theta2 = vector.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-12))
    small = theta2 < 1e-8
    coefficient_a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2.square() / 120.0,
        torch.sin(theta) / theta,
    )
    coefficient_b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2.square() / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
    )
    skew = hat(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return (
        identity
        + coefficient_a[..., None] * skew
        + coefficient_b[..., None] * (skew @ skew)
    )


def log_so3(rotation: torch.Tensor) -> torch.Tensor:
    """Shortest rotation vector, with a stable largest-diagonal branch near π."""
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    scale = theta / (2.0 * torch.sin(theta).clamp_min(1e-7))
    regular = scale[..., None] * vee
    small = theta < 1e-4
    regular = torch.where(small[..., None], 0.5 * vee, regular)

    # At the cut locus the skew part vanishes, so its signs cannot recover the
    # relative signs of the rotation axis.  Use the largest diagonal element
    # and symmetric off-diagonal entries instead, as in a matrix-to-quaternion
    # conversion.  The dominant axis component is chosen positive to make the
    # otherwise non-unique exact-π logarithm deterministic.
    diagonal = rotation.diagonal(dim1=-2, dim2=-1)
    largest = diagonal.argmax(dim=-1)
    r00, r11, r22 = diagonal.unbind(dim=-1)
    x = 0.5 * torch.sqrt((1.0 + r00 - r11 - r22).clamp_min(0.0))
    y = 0.5 * torch.sqrt((1.0 - r00 + r11 - r22).clamp_min(0.0))
    z = 0.5 * torch.sqrt((1.0 - r00 - r11 + r22).clamp_min(0.0))
    axis_from_x = torch.stack(
        (
            x,
            (rotation[..., 0, 1] + rotation[..., 1, 0]) / (4.0 * x).clamp_min(1e-7),
            (rotation[..., 0, 2] + rotation[..., 2, 0]) / (4.0 * x).clamp_min(1e-7),
        ),
        dim=-1,
    )
    axis_from_y = torch.stack(
        (
            (rotation[..., 0, 1] + rotation[..., 1, 0]) / (4.0 * y).clamp_min(1e-7),
            y,
            (rotation[..., 1, 2] + rotation[..., 2, 1]) / (4.0 * y).clamp_min(1e-7),
        ),
        dim=-1,
    )
    axis_from_z = torch.stack(
        (
            (rotation[..., 0, 2] + rotation[..., 2, 0]) / (4.0 * z).clamp_min(1e-7),
            (rotation[..., 1, 2] + rotation[..., 2, 1]) / (4.0 * z).clamp_min(1e-7),
            z,
        ),
        dim=-1,
    )
    axis = torch.where(
        (largest == 0)[..., None],
        axis_from_x,
        torch.where((largest == 1)[..., None], axis_from_y, axis_from_z),
    )
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    near_pi = (torch.pi - theta).abs() < 1e-3
    return torch.where(near_pi[..., None], theta[..., None] * axis, regular)


def apply_rotation(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...j->...i", rotation, vector)


def bridge_clock(
    physical_time: torch.Tensor | float,
    *,
    kind: str,
    power: float = 1.0,
) -> torch.Tensor | float:
    """Evaluate a batch-declared bridge clock in physical time."""
    if power <= 0.0:
        raise ValueError("bridge clock power must be positive")
    time = (
        torch.clamp(physical_time, 0.0, 1.0)
        if isinstance(physical_time, torch.Tensor)
        else min(max(physical_time, 0.0), 1.0)
    )
    if kind == "power":
        return 1.0 - (1.0 - time) ** power
    if kind in ("smoothstep", "smoothstep_power"):
        smooth = time**2 * (3.0 - 2.0 * time)
    elif kind in ("smootherstep", "smootherstep_power"):
        smooth = time**3 * (10.0 - 15.0 * time + 6.0 * time**2)
    else:
        raise ValueError(f"unsupported bridge clock: {kind}")
    return (
        smooth
        if kind in ("smoothstep", "smootherstep")
        else 1.0 - (1.0 - smooth) ** power
    )
