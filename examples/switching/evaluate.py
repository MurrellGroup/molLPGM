"""Distribution checks against independent reference samples."""

import numpy as np


def vector(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Expected at least two finite scalar samples")
    return x


def compare(generated, reference):
    x, y = vector(generated), vector(reference)
    scale = float(y.std())
    # Exact empirical CDFs, including unequal sample counts and ties.
    grid = np.sort(np.concatenate([x, y]))
    difference = np.searchsorted(np.sort(x), grid, side="right") / len(x)
    difference -= np.searchsorted(np.sort(y), grid, side="right") / len(y)
    distance = float(np.sum(np.diff(grid) * np.abs(difference[:-1])))
    quantiles = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return dict(
        n_generated=len(x),
        n_reference=len(y),
        wasserstein_1=distance,
        wasserstein_1_over_reference_sd=distance / scale if scale > 0 else None,
        ks_distance=float(np.max(np.abs(difference))),
        generated_mean=float(x.mean()),
        reference_mean=float(y.mean()),
        generated_sd=float(x.std()),
        reference_sd=scale,
        quantile_probabilities=quantiles.tolist(),
        generated_quantiles=np.quantile(x, quantiles).tolist(),
        reference_quantiles=np.quantile(y, quantiles).tolist(),
    )
