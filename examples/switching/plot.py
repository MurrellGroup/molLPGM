"""Export clean conditional-path, learned-path and endpoint-distribution panels."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

BLUE, RED = "#2468b4", "#ce3f46"


def render(data, output, ylim=(-2.2, 2.2), endpoint_xlim=(-1.5, 1.5), layout="below"):
    if layout not in ("below", "right"):
        raise ValueError("layout must be below or right")
    sideways = layout == "right"
    time = data["time"]
    conditional, generated, regimes = (
        data[k] for k in ("conditional", "generated", "regimes")
    )
    if (
        conditional.shape != regimes.shape
        or conditional.shape[0] != len(time)
        or generated.shape[0] != len(time)
    ):
        raise ValueError("Inconsistent path or regime shapes")
    if not np.isfinite(conditional).all() or not np.isfinite(generated).all():
        raise ValueError("Paths must be finite")
    if not np.isin(regimes, [-1, 1]).all():
        raise ValueError("Regime labels must be -1 or +1")
    rng = np.random.default_rng(1937)
    quantile_paths = "source_quantiles" in data
    if quantile_paths:
        if len(data["source_quantiles"]) != generated.shape[1]:
            raise ValueError("Each marginal path must have a source quantile")
        n = min(250, conditional.shape[1])
        selected = [
            rng.choice(conditional.shape[1], n, replace=False),
            np.arange(generated.shape[1]),
        ]
    else:
        n = min(250, conditional.shape[1], generated.shape[1])
        shared = rng.choice(
            min(conditional.shape[1], generated.shape[1]), n, replace=False
        )
        selected = [shared, shared]
    foreground = min(10, n)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    ):
        if sideways:
            fig = plt.figure(figsize=(10.2, 3.15), layout="constrained")
            fig.get_layout_engine().set(w_pad=0.02, wspace=0)
            grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.43], wspace=0)
            a = fig.add_subplot(grid[0, 0])
            b = fig.add_subplot(grid[0, 1], sharex=a, sharey=a)
            c = fig.add_subplot(grid[0, 2], sharey=b)
        else:
            fig = plt.figure(figsize=(8.0, 5.1), layout="constrained")
            grid = fig.add_gridspec(
                2, 2, height_ratios=[1.15, 1], hspace=0.12, wspace=0.08
            )
            a = fig.add_subplot(grid[0, 0])
            b = fig.add_subplot(grid[0, 1], sharex=a, sharey=a)
            c = fig.add_subplot(grid[1, :])
        for ax, paths, ids in zip((a, b), (conditional, generated), selected):
            for order, index in enumerate(ids):
                highlighted = order >= n - foreground
                alpha, width = (0.8, 0.85) if highlighted else (0.08, 0.4)
                if ax is b and quantile_paths:
                    alpha, width = 0.65, 0.65
                if ax is a:
                    if "conditional_vertices" in data:
                        lo, hi = data["conditional_offsets"][index : index + 2]
                        vertices = data["conditional_vertices"][lo:hi]
                        points = vertices[:, :2]
                        colors_sign = vertices[:-1, 2]
                    else:
                        points = np.column_stack((time, paths[:, index]))
                        colors_sign = regimes[:-1, index]
                    segments = np.stack((points[:-1], points[1:]), axis=1)
                    colors = np.where(
                        colors_sign[:, None] > 0,
                        np.array(matplotlib.colors.to_rgba(BLUE)),
                        np.array(matplotlib.colors.to_rgba(RED)),
                    )
                    ax.add_collection(
                        LineCollection(
                            segments, colors=colors, linewidths=width, alpha=alpha
                        )
                    )
                else:
                    ax.plot(time, paths[:, index], color=BLUE, alpha=alpha, lw=width)
            ax.set(xlim=(0, 1), xlabel="Generation time", ylabel="$x$")
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.margins(x=0)
        values = np.concatenate(
            [conditional[:, selected[0]].ravel(), generated[:, selected[1]].ravel()]
        )
        lower, upper = float(values.min()), float(values.max())
        pad = 0.04 * max(upper - lower, 0.01)
        a.set_ylim(*(ylim if ylim is not None else (lower - pad, upper + pad)))
        b.set_ylabel("")
        b.tick_params(labelleft=False)
        a.legend(
            [Line2D([], [], color=RED, lw=1.5), Line2D([], [], color=BLUE, lw=1.5)],
            ["$Z_t=-1$", "$Z_t=+1$"],
            frameon=False,
            ncol=2,
            loc="lower right" if sideways else "upper left",
            handlelength=1.2,
            columnspacing=1,
        )
        endpoints = (
            data["endpoint_samples"] if "endpoint_samples" in data else generated[-1]
        )
        support = np.concatenate([endpoints, data["target"]])
        bins = np.linspace(support.min(), support.max(), 161)
        orientation = "horizontal" if sideways else "vertical"
        # Normalize using all samples; axis limits only crop the visible window.
        c.hist(
            endpoints,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=BLUE,
            alpha=0.28,
            edgecolor="none",
            label="Generated",
            orientation=orientation,
        )
        if "density_x" in data and "density_y" in data:
            coordinates = (
                (data["density_y"], data["density_x"])
                if sideways
                else (data["density_x"], data["density_y"])
            )
            c.plot(
                *coordinates,
                color="#202020",
                lw=1.25,
                label="Target",
                alpha=0.5 if sideways else 1.0,
            )
        else:
            c.hist(
                data["target"],
                bins=bins,
                density=True,
                histtype="step",
                color="#202020",
                lw=1.1,
                label="Target samples",
                orientation=orientation,
                alpha=0.5 if sideways else 1.0,
            )
        if sideways:
            c.set(xlabel="Density", xlim=(0, None))
            c.set_xticks([0, 1, 2])
            c.tick_params(axis="y", left=False, labelleft=False)
            c.spines["left"].set_visible(False)
            c.legend(frameon=False, loc="lower right", fontsize=8, handlelength=1.2)
        else:
            c.set(xlabel="$x$ at $t=1$", ylabel="Density", xlim=endpoint_xlim)
            c.set_ylim(bottom=0)
            c.legend(frameon=False, loc="upper right")
        endpoint_title = "Endpoint density" if sideways else "Endpoint distribution"
        for ax, label, title in [
            (a, "A", "Conditional paths"),
            (b, "B", "Learned marginal paths"),
            (c, "C", endpoint_title),
        ]:
            ax.set_title(title, loc="left", pad=8, fontsize=10)
            label_offset = (
                (-0.14 if sideways else -0.045)
                if ax is c
                else (-0.06 if sideways else -0.1)
            )
            ax.text(
                label_offset,
                1.04,
                label,
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
            )
        fig.canvas.draw()
        if sideways:
            # A coordinate must occupy exactly the same physical height in B and C.
            points = np.array([[0.0, y] for y in (-2.0, 0.0, 2.0)])
            assert np.allclose(
                b.transData.transform(points)[:, 1],
                c.transData.transform(points)[:, 1],
                atol=1e-8,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "svg", "png"):
            fig.savefig(
                output.with_suffix("." + extension), dpi=600, bbox_inches="tight"
            )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layout", choices=("below", "right"), default="below")
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as data:
        render(data, args.output, layout=args.layout)


if __name__ == "__main__":
    main()
