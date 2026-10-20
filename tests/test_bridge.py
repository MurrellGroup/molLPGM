from pathlib import Path

import numpy as np
import pytest

from mol_lpgm.atom_bm import (
    load_profile,
    coefficients,
    marginal_variance,
    check_coordinate_process,
    resolve_profile,
)
from mol_lpgm.checkpoint import prune_regular_checkpoints

ROOT = Path(__file__).parents[1]
PROFILE = load_profile(ROOT / "configs/processes/bm.toml")


def test_time_changed_brownian_marginal_and_exact_transition_composition():
    # Independent closed form, including global centering only (in the sampler).
    for t in np.linspace(0, 1, 101):
        u = 1 - (1 - t**2.3) ** 3.1
        assert marginal_variance(t, PROFILE) == pytest.approx(
            14.8 * u * (1 - u), abs=2e-14
        )
    rng = np.random.default_rng(726)
    for s, t, v in np.sort(rng.uniform(0, 1, (1000, 3)), axis=1):
        r, p, q = coefficients(s, t, PROFILE)
        r2, p2, q2 = coefficients(t, v, PROFILE)
        np.testing.assert_allclose(
            coefficients(s, v, PROFILE),
            (r * r2, p * r2 + p2, q * r2 * r2 + q2),
            atol=2e-14,
            rtol=2e-12,
        )
        assert r + p == pytest.approx(1) and q >= 0
    # Tiny near-terminal retention must not round to zero through 1-u.
    r, p, q = coefficients(1 - 2e-12, 1 - 1e-12, PROFILE)
    assert 0 < r < 1 and 0 < p < 1 and q > 0
    assert coefficients(0.9999, 1, PROFILE) == (0.0, 1.0, 0.0)
    assert coefficients(1, 1, PROFILE) == (1.0, 0.0, 0.0)
