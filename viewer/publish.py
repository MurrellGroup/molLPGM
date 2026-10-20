"""Build a standalone viewer site from one or two recorded molecular sample panels."""

import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        len(args.panel) not in (1, 2)
        or len(args.panel) != len(args.label)
        or len(args.panel) != len(args.evaluation)
    ):
        raise ValueError("Supply one or two matching panel/label/evaluation arguments")
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    for name in (
        "index.html",
        "style.css",
        "viewer.js",
        "figure.js",
        "figure-utils.mjs",
    ):
        shutil.copy2(source / name, args.output / name)
    shutil.copytree(source / "vendor", args.output / "vendor")
    entries = []
    quality = {}
    panels = []
    for index, (panel, label, evaluation) in enumerate(
        zip(args.panel, args.label, args.evaluation)
    ):
        catalog = json.loads((panel / "trajectories.json").read_text())
        rows = [
            json.loads(line)
            for line in (evaluation / "samples.jsonl").read_text().splitlines()
        ]
        checks = {Path(row["file"]).stem.removeprefix("samp_"): row for row in rows}
        prefix = f"model{index+1}"
        shutil.copytree(panel, args.output / prefix)
        items = []
        for item in catalog["trajectories"]:
            suffix = item["id"].removeprefix("sample_")
            row = checks.get(suffix)
            if row is None:
                raise ValueError(f'Missing evaluation for {item["id"]}')
            key = f'{prefix}-{item["id"]}'
            quality[key] = {k: bool(row[k]) for k in ("valid", "connected")}
            entry = dict(
                id=key,
                label=f'{label} · {item["label"]}',
                model_label=label,
                trajectory=f'{prefix}/{item["trajectory"]}',
            )
            items.append(entry)
        panels.append(items)
    if len(panels) == 2:
        if len(panels[0]) != len(panels[1]):
            raise ValueError("Paired panels need equal sample counts")
        for left, right in zip(*panels):
            # Ensure both recordings describe the same molecular conditioning.
            a, b = [
                json.loads((args.output / e["trajectory"]).read_text())
                for e in (left, right)
            ]
            for key in ("source_name", "atom_index", "segment_id"):
                if a[key] != b[key]:
                    raise ValueError(f"Paired conditioning differs: {key}")
            left["paired_id"] = right["id"]
            right["paired_id"] = left["id"]
            entries.extend((left, right))
    else:
        entries = panels[0]
    (args.output / "quality.json").write_text(
        json.dumps(dict(format_version=1, samples=quality)) + "\n"
    )
    catalog = dict(
        format_version=1,
        trajectories=entries,
        quality_file="quality.json",
        quality_scope="paired" if len(panels) == 2 else "single",
        default_quality="valid_connected",
    )
    (args.output / "trajectories.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"Serve {args.output} with python -m http.server; open /index.html?figure=1")


if __name__ == "__main__":
    main()
