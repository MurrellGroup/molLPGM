"""Atom-wise Brownian bridge in the scalar clock u(t)=1-(1-t**a)**b."""

from __future__ import annotations
import math
from pathlib import Path
import tomllib

TIME_CHANGED_BM = "atom-brownian-kumaraswamy-v1"
CARTESIAN_BM = (TIME_CHANGED_BM,)
FRAGMENT = "fragment-se3-plus-internal"
PROFILE_KEYS = ("atom_bm_variance", "atom_bm_clock_a", "atom_bm_clock_b")


def profile_keys(profile):
    return PROFILE_KEYS


def validate_profile(profile):
    if profile.get("coordinate_process") != TIME_CHANGED_BM:
        raise ValueError("unsupported Brownian process")
    if any(
        not math.isfinite(float(profile[k])) or float(profile[k]) <= 0
        for k in PROFILE_KEYS
    ):
        raise ValueError(
            "Brownian variance and clock parameters must be finite and positive"
        )


def load_profile(path):
    profile = tomllib.loads(Path(path).read_text())
    validate_profile(profile)
    return {k: profile[k] for k in ("coordinate_process", *PROFILE_KEYS)}


def check_coordinate_process(actual, expected):
    mode = actual.get("coordinate_process", FRAGMENT)
    if mode not in (FRAGMENT, TIME_CHANGED_BM) or mode != expected.get(
        "coordinate_process", FRAGMENT
    ):
        raise ValueError("coordinate process conflicts with model configuration")
    if mode == TIME_CHANGED_BM:
        validate_profile(actual)
        validate_profile(expected)
        for key in PROFILE_KEYS:
            if actual[key] != expected[key]:
                raise ValueError(f"Brownian parameter differs: {key}")


def resolve_profile(path, saved=None):
    profile = (
        load_profile(path)
        if path
        else (
            {k: saved[k] for k in ("coordinate_process", *PROFILE_KEYS)}
            if saved and saved.get("coordinate_process") == TIME_CHANGED_BM
            else {}
        )
    )
    if saved is not None:
        check_coordinate_process(profile, saved)
    return profile


def clock_log_retention(t, profile):
    """Compute log(1-u) without cancellation near t=1."""
    if not 0 <= t <= 1:
        raise ValueError("time outside [0,1]")
    if t == 0:
        return 0.0
    if t == 1:
        return -math.inf
    return profile["atom_bm_clock_b"] * math.log(
        -math.expm1(profile["atom_bm_clock_a"] * math.log(t))
    )


def cartesian_clock(t, profile):
    return -math.expm1(clock_log_retention(t, profile))


def marginal_variance(t, profile):
    a = clock_log_retention(t, profile)
    return profile["atom_bm_variance"] * math.exp(a) * -math.expm1(a)


def coefficients(start, stop, profile):
    if not 0 <= start <= stop <= 1:
        raise ValueError("invalid Brownian interval")
    if start == stop:
        return 1.0, 0.0, 0.0
    if stop == 1:
        return 0.0, 1.0, 0.0
    a0, a1 = clock_log_retention(start, profile), clock_log_retention(stop, profile)
    pull = -math.expm1(a1 - a0)
    return math.exp(a1 - a0), pull, profile["atom_bm_variance"] * math.exp(a1) * pull
