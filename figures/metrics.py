"""Kabsch errors and radii of gyration for paired molecular trajectories.

Input: a JSON list of pairs, each containing ``sample_index``, ``fragment``
and ``bm``. Model entries name trajectory.json files relative to the list.
The trajectory format is the same one used by the standalone viewer.
"""

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np


def kabsch_mse(positions, groups):
    """Proper-rotation alignment to the path's own endpoint, atom-weighted."""
    x = np.asarray(positions, dtype=np.float64)
    groups = np.asarray(groups)
    error = np.zeros(len(x))
    for group in np.unique(groups):
        local = x[:, groups == group]
        centered = local - local.mean(axis=1, keepdims=True)
        target = centered[-1]
        u, _, vh = np.linalg.svd(np.einsum("tni,nj->tij", centered, target))
        sign = np.ones((len(x), 3))
        sign[:, -1] = np.linalg.det(u @ vh)
        rotation = (u * sign[:, None, :]) @ vh
        error += np.square(centered @ rotation - target).sum(axis=(1, 2)) / x.shape[1]
    return error


def path_metrics(positions, groups):
    """Squared Euclidean error per atom (Å²) and equal-atom radii (Å)."""
    x = np.asarray(positions, dtype=np.float64)
    groups = np.asarray(groups)
    if x.ndim != 3 or x.shape[2] != 3 or x.shape[1] != len(groups):
        raise ValueError("Expected positions[frames, atoms, 3] and one group per atom")
    if len(x) < 2 or not len(groups) or not np.isfinite(x).all():
        raise ValueError(
            "A finite, nonempty trajectory with both endpoints is required"
        )
    global_center = x.mean(axis=1, keepdims=True)
    total_rg2 = np.square(x - global_center).sum(axis=2).mean(axis=1)
    within_rg2 = np.zeros(len(x))
    centroid_rg2 = np.zeros(len(x))
    for group in np.unique(groups):
        local = x[:, groups == group]
        center = local.mean(axis=1, keepdims=True)
        within_rg2 += np.square(local - center).sum(axis=(1, 2)) / x.shape[1]
        centroid_rg2 += (
            local.shape[1]
            / x.shape[1]
            * np.square(center - global_center).sum(axis=(1, 2))
        )
    np.testing.assert_allclose(
        total_rg2, within_rg2 + centroid_rg2, rtol=1e-12, atol=1e-12
    )
    return dict(
        mse_total=kabsch_mse(x, np.zeros(len(groups), dtype=int)),
        mse_fragment=kabsch_mse(x, groups),
        rg_total=np.sqrt(total_rg2),
        rg_fragment=np.sqrt(within_rg2),
    )


def read_path(filename):
    meta = (tomllib.loads if filename.suffix == ".toml" else json.loads)(
        filename.read_text()
    )
    x = np.fromfile(filename.parent / "positions.f32", dtype="<f4").reshape(
        meta["frame_count"], meta["atom_count"], 3
    )
    return meta, x


def prepare(pairs_file, output, conditional=False):
    pairs = (
        tomllib.loads(pairs_file.read_text())["pairs"]
        if pairs_file.suffix == ".toml"
        else json.loads(pairs_file.read_text())
    )
    if not pairs or len({p["sample_index"] for p in pairs}) != len(pairs):
        raise ValueError("Provide nonempty pairs with unique sample indices")
    arrays, time = {}, None
    for pair in pairs:
        loaded = {
            model: read_path(pairs_file.parent / pair[model])
            for model in ("fragment", "bm")
        }
        a, b = loaded["fragment"], loaded["bm"]
        for key in ("times", "source_name", "atom_index", "segment_id"):
            if a[0][key] != b[0][key]:
                raise ValueError(
                    f'Paired {key} does not match for sample {pair["sample_index"]}'
                )
        if len(set(a[0]["segment_id"])) < 2:
            raise ValueError(
                "The comparison requires at least two fragments per molecule"
            )
        np.testing.assert_array_equal(a[1][0], b[1][0])
        if conditional:
            np.testing.assert_array_equal(a[1][-1], b[1][-1])
        if time is None:
            time = np.asarray(a[0]["times"])
        np.testing.assert_array_equal(time, a[0]["times"])
        for model, (meta, x) in loaded.items():
            for metric, value in path_metrics(x, meta["segment_id"]).items():
                arrays.setdefault(f"{model}_{metric}", []).append(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        time=time,
        sample_indices=[p["sample_index"] for p in pairs],
        path_kind="conditional" if conditional else "generated",
        **arrays,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditional", action="store_true")
    args = parser.parse_args()
    prepare(args.pairs, args.output, args.conditional)
