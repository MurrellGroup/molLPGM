"""Geometric invariants of the publication measurements."""

import numpy as np

from metrics import path_metrics


def test_fragment_rigid_motion_and_atom_weighting():
    rng = np.random.default_rng(72)
    endpoint = rng.normal(size=(9, 3))
    groups = np.array([0] * 4 + [1] * 5)
    moved = endpoint.copy()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moved[:4] = moved[:4] @ rotation + [8, 0, 0]
    moved[4:] += [-3, 1, 0]
    x = np.stack([moved, endpoint])
    values = path_metrics(x, groups)
    assert values["mse_fragment"].max() < 1e-27
    assert values["mse_total"][0] > 1
    assert values["mse_total"][-1] < 1e-27
    np.testing.assert_allclose(values["rg_fragment"][0], values["rg_fragment"][-1])
    expected = sum(
        np.square(endpoint[groups == g] - endpoint[groups == g].mean(0)).sum()
        for g in (0, 1)
    ) / len(endpoint)
    np.testing.assert_allclose(values["rg_fragment"][-1] ** 2, expected)


def test_reflection_is_not_treated_as_rotation():
    endpoint = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    reflected = endpoint * [-1, 1, 1]
    values = path_metrics(np.stack([reflected, endpoint]), np.zeros(4))
    assert values["mse_total"][0] > 0.1
    np.testing.assert_allclose(values["rg_total"][0], values["rg_total"][-1])
    np.testing.assert_allclose(values["mse_total"], values["mse_fragment"])
