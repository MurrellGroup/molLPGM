"""Plot own-endpoint Kabsch errors and radii of gyration of paired GEOM paths.

Replot downloaded data:
    python plot.py --data per-sample.npz --output-dir .

Compute input arrays from trajectories with metrics.py. Radii use equal atom
weights, not atomic masses. This script requires only NumPy and Matplotlib.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

MODELS = ("fragment", "bm")
SCALES = ("total", "fragment")
COLORS = {"fragment": "#2464A5", "bm": "#D62728"}
MODEL_LABELS = {"fragment": "Fragment Noise", "bm": "Brownian Atoms"}
STYLES = {"total": (0, (4.8, 2.7)), "fragment": "-"}
STEM = "geom-error-and-radius"


def plot(data_path, output):
    data = np.load(data_path)
    conditional = "path_kind" in data and str(data["path_kind"]) == "conditional"
    dataset = str(data["dataset"]) if "dataset" in data else "geom"
    stem = f"{dataset}-conditional-error-and-radius" if conditional else f"{dataset}-error-and-radius"
    time = data["time"]
    keys = [
        f"{m}_{metric}_{s}" for metric in ("mse", "rg") for m in MODELS for s in SCALES
    ]
    curves = {key: data[key] for key in keys}
    samples = len(data["sample_indices"])
    assert time.ndim == 1 and len(time) >= 2 and samples > 0
    assert time[0] == 0 and time[-1] == 1 and np.all(np.diff(time) > 0)
    assert all(
        x.shape == (samples, len(time)) and np.isfinite(x).all() and (x >= 0).all()
        for x in curves.values()
    )
    for model in MODELS:
        for metric in ("mse", "rg"):
            assert np.all(
                curves[f"{model}_{metric}_fragment"]
                <= curves[f"{model}_{metric}_total"] + 1e-9
            )
        for scale in SCALES:
            assert curves[f"{model}_mse_{scale}"][:, -1].max() < 1e-20
    for scale in SCALES:
        np.testing.assert_array_equal(
            curves[f"fragment_rg_{scale}"][:, 0], curves[f"bm_rg_{scale}"][:, 0]
        )
        if conditional:
            np.testing.assert_array_equal(
                curves[f"fragment_mse_{scale}"][:, 0], curves[f"bm_mse_{scale}"][:, 0]
            )
            np.testing.assert_array_equal(
                curves[f"fragment_rg_{scale}"][:, -1], curves[f"bm_rg_{scale}"][:, -1]
            )
    means = {key: value.mean(axis=0) for key, value in curves.items()}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.labelcolor": "#242424",
            "text.color": "#242424",
            "xtick.color": "#383838",
            "ytick.color": "#383838",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#383838",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": stem,
            "path.simplify": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.65))
    fig.subplots_adjust(left=0.083, right=0.985, bottom=0.165, top=0.83, wspace=0.29)
    limits = {}
    for ax, metric, ylabel in zip(
        axes, ("mse", "rg"), ("Kabsch-aligned MSE (Å²)", "Radius of gyration (Å)")
    ):
        for model in MODELS:
            for scale in SCALES:
                values = curves[f"{model}_{metric}_{scale}"]
                segments = np.stack(
                    [np.broadcast_to(time, values.shape), values], axis=-1
                )
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors=COLORS[model],
                        linewidths=0.35,
                        linestyles=STYLES[scale],
                        alpha=0.045,
                        zorder=1,
                        rasterized=False,
                    )
                )
        for model in MODELS:
            for scale in SCALES:
                ax.plot(
                    time,
                    means[f"{model}_{metric}_{scale}"],
                    color=COLORS[model],
                    linestyle=STYLES[scale],
                    lw=1.8,
                    alpha=0.7,
                    solid_capstyle="round",
                    dash_capstyle="butt",
                    zorder=3,
                )
        largest = max(
            curves[f"{model}_{metric}_{scale}"].max()
            for model in MODELS
            for scale in SCALES
        )
        unit = 5 if metric == "mse" else 1
        ymax = float(unit * np.ceil(largest / unit))
        limits[metric] = [0, ymax]
        ax.set(
            xlim=(0, 1),
            ylim=(0, ymax),
            xlabel="Bridge time $t$" if conditional else "Generation time $t$",
            ylabel=ylabel,
        )
        ax.set_xticks(
            [0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"]
        )
        ax.set_yticks(np.arange(0, ymax + 0.01, 10 if metric == "mse" else 2))
        ax.tick_params(direction="out", length=3, width=0.7, pad=3.5)
        assert len(ax.lines) == 4 and len(ax.collections) == 4
    handles = [
        Line2D(
            [],
            [],
            color=COLORS["fragment"],
            lw=2,
            alpha=0.7,
            label=MODEL_LABELS["fragment"],
        ),
        Line2D([], [], color=COLORS["bm"], lw=2, alpha=0.7, label=MODEL_LABELS["bm"]),
        Line2D(
            [],
            [],
            color="#444444",
            lw=1.6,
            linestyle=STYLES["total"],
            label="Whole molecule",
        ),
        Line2D([], [], color="#444444", lw=1.6, label="Within fragment"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.982),
        frameon=False,
        ncol=4,
        fontsize=8.3,
        handlelength=2.5,
        columnspacing=1.5,
        handletextpad=0.65,
    )
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            output / f"{stem}.{ext}",
            dpi=600,
            metadata={"Creator": "molLPGM"} if ext == "pdf" else None,
        )
    plt.close(fig)
    with (output / "mean-curves.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", *keys])
        writer.writerows(
            [float(t), *[float(means[key][i]) for key in keys]]
            for i, t in enumerate(time)
        )
    details = dict(
        samples_per_model=samples,
        frames_per_path=len(time),
        path_kind="conditional" if conditional else "generated",
        mean_curves_per_panel=4,
        individual_curves_per_panel=4 * samples,
        selection="Paired molecules with at least two fragments; no quality filtering by this script.",
        endpoint=(
            "Both conditional paths share the same Gaussian source and dataset endpoint; errors are to that shared endpoint."
            if conditional
            else "Each trajectory is compared only with its own generated final coordinates."
        ),
        mse="Squared Euclidean displacement per atom after proper Kabsch alignment. Whole molecule uses one alignment; within fragment uses one alignment per fragment, atom-weighted within each molecule.",
        radius="Equal-atom-weight radius of gyration in angstrom. Whole molecule: sqrt(mean_i ||x_i - global_centroid||^2). Within fragment: sqrt(mean_i ||x_i - centroid_of_fragment_i||^2).",
        averaging="Compute each metric per molecule, then average over molecules. For radius, take the square root before averaging molecules.",
        initial_radii="Exactly identical for both models in each paired case, for both whole-molecule and within-fragment radii.",
        normalization=False,
        baseline_subtraction=False,
        smoothing=False,
        path_subsampling=False,
        downsampling=False,
        range_clipping=False,
        figure_inches=[7.4, 3.65],
        y_limits=limits,
        colors=COLORS,
        model_labels=MODEL_LABELS,
        legend={
            "color": "model",
            "dashed": "whole molecule",
            "solid": "within fragment",
        },
        individual_alpha=0.045,
        individual_width_pt=0.35,
        mean_width_pt=1.8,
        mean_alpha=0.7,
        mean_outline=False,
        source_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        export="Full vector PDF and SVG; 600 dpi PNG.",
        initial_means={key: float(value[0]) for key, value in means.items()},
        final_means={key: float(value[-1]) for key, value in means.items()},
    )
    (output / "figure.json").write_text(json.dumps(details, indent=2) + "\n")
    if data_path.resolve() != (output / "per-sample.npz").resolve():
        shutil.copy2(data_path, output / "per-sample.npz")
    if Path(__file__).resolve() != (output / "plot.py").resolve():
        shutil.copy2(__file__, output / "plot.py")
    print(json.dumps(details, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.data, args.output_dir)


if __name__ == "__main__":
    main()
