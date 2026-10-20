from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .batch import MolecularBatch, load_batch, prefetch_batches
from .checkpoint import load_checkpoint
from .process import INFERENCE_TIME_SCHEDULES
from .sampler import sample

ATOM_SYMBOLS = (
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "Bi",
    "X",
)


def _arguments(default_domain: str = "qm9") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate molecules from packed fragmentation templates."
    )
    parser.add_argument(
        "--domain",
        choices=("qm9", "geom-drugs"),
        default=default_domain,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--ema", action="store_true", help="Sample the saved EMA weights"
    )
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=734_921)
    parser.add_argument(
        "--time-schedule", choices=INFERENCE_TIME_SCHEDULES, default="sqrt"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--write-workers", type=int, default=8)
    return parser.parse_args()


def _batch_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "metadata.toml").is_file()
    )


def _load_batches(paths: list[Path], domain: str):
    for path in paths:
        yield load_batch(path, pin_memory=True, expected_domain=domain)


def _fragment_signature(fragment_ids: np.ndarray) -> str:
    _, normalized = np.unique(fragment_ids, return_inverse=True)
    return hashlib.sha256(np.asarray(normalized, dtype="<i4").tobytes()).hexdigest()


def _write_xyz(
    path: Path,
    positions: np.ndarray,
    atom_classes: np.ndarray,
) -> None:
    if positions.shape != (len(atom_classes), 3):
        raise ValueError("positions and atom classes do not align")
    if np.any(atom_classes >= len(ATOM_SYMBOLS)):
        raise ValueError("sample contains an out-of-vocabulary atom class")
    path.write_text(
        "".join(
            f"{ATOM_SYMBOLS[int(label)]} "
            f"{float(x):.8f} {float(y):.8f} {float(z):.8f}\n"
            for (x, y, z), label in zip(positions, atom_classes)
        )
    )


def _write_batch(
    batch: MolecularBatch,
    coords: torch.Tensor,
    atom_classes: torch.Tensor,
    *,
    first_index: int,
    remaining: int,
    output_dir: Path,
    sampling_seed: int,
    executor: ThreadPoolExecutor,
) -> tuple[list[dict], list[Future]]:
    coords_cpu = coords.detach().float().cpu().numpy()
    classes_cpu = atom_classes.detach().cpu().numpy()
    valid = batch.valid_atom.detach().cpu().numpy()
    documents = batch.document_id.detach().cpu().numpy()
    fragments = batch.fragment_id.detach().cpu().numpy()
    targets = batch.target_atom_class.detach().cpu().numpy()
    source_names = batch.metadata.get("source_names", [])
    records: list[dict] = []
    futures: list[Future] = []
    document_count = min(int(batch.metadata["n_documents"]), remaining)
    for document in range(document_count):
        selected = valid & (documents == document)
        positions = np.ascontiguousarray(coords_cpu[selected])
        predicted = np.ascontiguousarray(classes_cpu[selected])
        fragment_ids = np.ascontiguousarray(fragments[selected])
        target_classes = np.ascontiguousarray(targets[selected])
        sample_index = first_index + document
        filename = f"samp_{sample_index:06d}.xyz"
        futures.append(
            executor.submit(_write_xyz, output_dir / filename, positions, predicted)
        )
        records.append(
            {
                "sample_index": sample_index,
                "file": filename,
                "source_name": (
                    source_names[document]
                    if document < len(source_names)
                    else f"document-{document}"
                ),
                "atom_count": int(selected.sum()),
                "fragment_count": int(np.unique(fragment_ids).size),
                "fragmentation_sha256": _fragment_signature(fragment_ids),
                "template_atom_classes_sha256": hashlib.sha256(
                    np.asarray(target_classes, dtype=np.uint8).tobytes()
                ).hexdigest(),
                "sampling_seed": sampling_seed,
            }
        )
    return records, futures


def main(default_domain: str = "qm9") -> None:
    args = _arguments(default_domain)
    if args.samples <= 0 or args.steps <= 0:
        raise ValueError("sample and step counts must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    paths = _batch_paths(args.template_root)
    if not paths:
        raise ValueError(f"no template batches in {args.template_root}")

    selected_paths: list[Path] = []
    available = 0
    for path in paths:
        batch = load_batch(path, expected_domain=args.domain)
        available += int(batch.metadata["n_documents"])
        selected_paths.append(path)
        if available >= args.samples:
            break
    if available < args.samples:
        raise ValueError(
            f"templates contain {available} molecules, need {args.samples}"
        )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("compiled FlexAttention sampling requires CUDA")
    torch.set_float32_matmul_precision("high")
    model, checkpoint = load_checkpoint(
        args.checkpoint,
        device=device,
        expected_training_domain=args.domain,
        use_ema=args.ema,
    )
    model.eval()
    compiled = torch.compile(
        model, fullgraph=True, dynamic=False, mode=args.compile_mode
    )
    args.output_dir.mkdir(parents=True)
    batches = prefetch_batches(
        _load_batches(selected_paths, args.domain),
        device=device,
        depth=args.prefetch_batches,
    )

    records: list[dict] = []
    futures: list[Future] = []
    timings: list[dict] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.write_workers) as executor:
        for batch_index, batch in enumerate(batches):
            sampling_seed = args.seed + batch_index
            torch.cuda.synchronize(device)
            batch_started = time.perf_counter()
            coords, classes = sample(
                compiled,
                batch,
                steps=args.steps,
                seed=sampling_seed,
                time_schedule=args.time_schedule,
            )
            batch_records, batch_futures = _write_batch(
                batch,
                coords,
                classes,
                first_index=len(records) + 1,
                remaining=args.samples - len(records),
                output_dir=args.output_dir,
                sampling_seed=sampling_seed,
                executor=executor,
            )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - batch_started
            records.extend(batch_records)
            futures.extend(batch_futures)
            timing = {
                "batch": batch_index + 1,
                "molecules": len(batch_records),
                "seconds": elapsed,
                "samples_per_second": len(batch_records) / elapsed,
            }
            timings.append(timing)
            print(json.dumps(timing), flush=True)
            if len(records) == args.samples:
                break
        for future in futures:
            future.result()

    total_seconds = time.perf_counter() - started
    manifest = {
        "format_version": 1,
        "domain": args.domain,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_weights": "ema" if args.ema else "raw",
        "ema_decay": checkpoint["ema"]["decay"] if args.ema else None,
        "template_root": str(args.template_root.resolve()),
        "samples": args.samples,
        "steps": args.steps,
        "seed": args.seed,
        "time_schedule": args.time_schedule,
        "compile_mode": args.compile_mode,
        "batch_timings": timings,
        "total_seconds": total_seconds,
        "samples_per_second": args.samples / total_seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
