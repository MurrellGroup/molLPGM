"""Generate XYZ samples and viewer trajectories in the same sampler execution."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from mol_lpgm.trajectory import _document_metadata, _write_trajectory
from mol_lpgm.batch import load_batch
from mol_lpgm.checkpoint import load_checkpoint
from mol_lpgm.sample import ATOM_SYMBOLS, _batch_paths, _write_batch
from mol_lpgm.sampler import sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ema", action="store_true", help="Use saved EMA weights")
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--viewer-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=555921)
    parser.add_argument("--time-schedule", choices=["linear", "sqrt"], default="sqrt")
    args = parser.parse_args()
    if args.samples <= 0 or args.steps <= 0:
        raise ValueError("Sample and step counts must be positive")
    for output in (args.output_dir, args.viewer_dir):
        if output.exists():
            raise FileExistsError(output)
    paths = _batch_paths(args.template_root)
    torch.set_float32_matmul_precision("high")
    model, saved = load_checkpoint(
        args.checkpoint, device=args.device, use_ema=args.ema
    )
    model.eval()
    compiled = torch.compile(model, fullgraph=True, dynamic=False)
    for output in (args.output_dir, args.viewer_dir):
        output.mkdir(parents=True)
    records, entries, checks, timings = [], [], [], []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(args.device)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for batch_index, path in enumerate(paths):
            host = load_batch(
                path, pin_memory=True, expected_domain=saved["training_domain"]
            )
            count = min(int(host.metadata["n_documents"]), args.samples - len(records))
            if count <= 0:
                break
            descriptions = _document_metadata(host, list(range(count)))
            batch = host.to(args.device)
            frames, labels, times = [], [], []

            def capture(_step, physical_time, coords, classes):
                frames.append(coords.detach().float().cpu().numpy().copy())
                labels.append(classes.detach().cpu().numpy().copy())
                times.append(float(physical_time))

            seed = args.seed + batch_index
            batch_started = time.monotonic()
            coords, classes = sample(
                compiled,
                batch,
                steps=args.steps,
                seed=seed,
                time_schedule=args.time_schedule,
                trajectory_callback=capture,
            )
            np.testing.assert_array_equal(
                frames[-1], coords.detach().float().cpu().numpy()
            )
            np.testing.assert_array_equal(labels[-1], classes.detach().cpu().numpy())
            batch_records, futures = _write_batch(
                batch,
                coords,
                classes,
                first_index=len(records) + 1,
                remaining=count,
                output_dir=args.output_dir,
                sampling_seed=seed,
                executor=executor,
            )
            for future in futures:
                future.result()
            for description, record in zip(descriptions, batch_records, strict=True):
                indices = description["indices"]
                name = f'sample_{record["sample_index"]:06d}'
                _write_trajectory(
                    args.viewer_dir / name,
                    frames=[f[indices] for f in frames],
                    labels=[f[indices] for f in labels],
                    times=times,
                    description=description,
                    batch=host,
                    domain=saved["training_domain"],
                    checkpoint=args.checkpoint,
                    checkpoint_step=saved["step"],
                    seed=seed,
                    steps=args.steps,
                    time_schedule=args.time_schedule,
                    checkpoint_weights=model.checkpoint_weights,
                )
                xyz = np.loadtxt(args.output_dir / record["file"], dtype=str, ndmin=2)
                expected_labels = [ATOM_SYMBOLS[int(v)] for v in labels[-1][indices]]
                assert expected_labels == xyz[:, 0].tolist()
                error = float(
                    np.abs(frames[-1][indices] - xyz[:, 1:].astype(float)).max()
                )
                assert error <= 5.01e-9, error
                checks.append(error)
                record["xyz_sha256"] = hashlib.sha256(
                    (args.output_dir / record["file"]).read_bytes()
                ).hexdigest()
                entries.append(
                    {
                        "id": name,
                        "label": f'Sample {record["sample_index"]:04d} · {len(indices)} atoms · {description["source_name"]}',
                        "trajectory": f"{name}/trajectory.json",
                    }
                )
            records.extend(batch_records)
            timing = {
                "batch": batch_index + 1,
                "samples_written": len(records),
                "seconds": time.monotonic() - batch_started,
            }
            timings.append(timing)
            print(json.dumps(timing), flush=True)
    assert len(records) == args.samples, "Not enough template molecules"
    manifest = {
        "format_version": 1,
        "domain": saved["training_domain"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(saved["step"]),
        "checkpoint_sha256": hashlib.file_digest(
            args.checkpoint.open("rb"), "sha256"
        ).hexdigest(),
        "checkpoint_weights": model.checkpoint_weights,
        "ema_decay": saved["ema"]["decay"] if args.ema else None,
        "ema_num_updates": saved["ema"]["num_updates"] if args.ema else None,
        "training_process": model.training_process,
        "template_root": str(args.template_root.resolve()),
        "samples": len(records),
        "steps": args.steps,
        "seed": args.seed,
        "time_schedule": args.time_schedule,
        "compile_mode": "default",
        "device": args.device,
        "viewer_dir": str(args.viewer_dir.resolve()),
        "trajectory_from_same_execution": True,
        "endpoint_labels_match": True,
        "max_xyz_rounding_error_angstrom": max(checks),
        "frames_per_trajectory": args.steps + 1,
        "batch_timings": timings,
        "total_seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(args.device),
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.viewer_dir / "trajectories.json").write_text(
        json.dumps(
            {"format_version": 1, "default": entries[0]["id"], "trajectories": entries},
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in ("records", "batch_timings")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
