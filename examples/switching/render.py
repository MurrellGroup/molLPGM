"""Render the saved switching figures; requires only NumPy and Matplotlib."""

import argparse
from pathlib import Path

import numpy as np

from plot import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("reference"))
    parser.add_argument("--output", type=Path, default=Path("figures"))
    parser.add_argument("--layout", choices=("right", "below", "both"), default="both")
    args = parser.parse_args()
    files = sorted(args.data.glob("switching-*.npz"))
    if not files:
        parser.error(f"No switching-*.npz files in {args.data}")
    layouts = ("below", "right") if args.layout == "both" else (args.layout,)
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            for layout in layouts:
                suffix = "-right" if layout == "right" else ""
                render(data, args.output / (path.stem + suffix), layout=layout)
        print(f"Rendered {path.stem}", flush=True)


if __name__ == "__main__":
    main()
