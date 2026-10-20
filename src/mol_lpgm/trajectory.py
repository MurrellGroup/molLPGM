"""Write recorded sampling paths in the standalone viewer format."""

from pathlib import Path
import json
import numpy as np
from .batch import MolecularBatch
from .sample import ATOM_SYMBOLS


def _document_metadata(batch: MolecularBatch, documents: list[int]) -> list[dict]:
    valid = batch.valid_atom.numpy()
    document_id = batch.document_id.numpy()
    fragment_id = batch.fragment_id.numpy()
    atom_index = batch.atom_index.numpy()
    source_names = batch.metadata.get("source_names", [])
    result = []
    for document in documents:
        selected = valid & (document_id == document)
        indices = np.flatnonzero(selected)
        if not len(indices):
            raise ValueError(f"document {document} contains no atoms")
        _, normalized_fragments = np.unique(fragment_id[indices], return_inverse=True)
        result.append(
            {
                "document": document,
                "indices": indices,
                "atom_index": atom_index[indices].tolist(),
                "fragment_id": normalized_fragments.astype(np.int32),
                "source_name": (
                    source_names[document]
                    if document < len(source_names)
                    else f"document-{document}"
                ),
            }
        )
    return result


def _write_trajectory(
    destination: Path,
    *,
    frames: list[np.ndarray],
    labels: list[np.ndarray],
    times: list[float],
    description: dict,
    batch: MolecularBatch,
    domain: str,
    checkpoint: Path | None,
    checkpoint_step: int | None,
    seed: int,
    steps: int,
    time_schedule: str,
    mode: str = "model_sample",
    checkpoint_weights: str = "raw",
) -> None:
    positions = np.ascontiguousarray(np.stack(frames), dtype="<f4")
    atom_labels = np.ascontiguousarray(np.stack(labels), dtype=np.uint8)
    if positions.shape != (len(times), len(description["indices"]), 3):
        raise ValueError("captured coordinate trajectory has an invalid shape")
    if atom_labels.shape != positions.shape[:2]:
        raise ValueError("captured atom-label trajectory has an invalid shape")
    if not np.isfinite(positions).all():
        raise ValueError("captured trajectory contains nonfinite coordinates")
    if np.any(atom_labels >= len(ATOM_SYMBOLS)):
        raise ValueError("captured trajectory contains an invalid atom label")

    centered = positions - positions.mean(axis=1, keepdims=True)
    radii = np.linalg.norm(centered, axis=-1)
    frame_display_radius = np.maximum(np.quantile(radii, 0.995, axis=1), 1.0)
    destination.mkdir(parents=True)
    (destination / "positions.f32").write_bytes(positions.tobytes(order="C"))
    (destination / "atom_labels.u8").write_bytes(atom_labels.tobytes(order="C"))
    fragment_ids = description["fragment_id"]
    fragment_count = int(fragment_ids.max()) + 1
    metadata = {
        "format_version": 1,
        "domain": domain,
        "mode": mode,
        "entity_name": "fragment",
        "positions_file": "positions.f32",
        "atom_labels_file": "atom_labels.u8",
        "frame_count": len(times),
        "atom_count": positions.shape[1],
        "segment_count": fragment_count,
        "times": times,
        "atom_type": atom_labels[-1].tolist(),
        "atom_label_names": list(ATOM_SYMBOLS),
        "atom_index": description["atom_index"],
        "segment_id": fragment_ids.tolist(),
        "display_radius": max(float(np.quantile(radii, 0.995)), 1.0),
        "frame_display_radius": frame_display_radius.tolist(),
        "document": description["document"],
        "source_name": description["source_name"],
        "checkpoint": str(checkpoint.resolve()) if checkpoint is not None else None,
        "checkpoint_step": checkpoint_step,
        "checkpoint_weights": checkpoint_weights,
        "seed": seed,
        "inference_time_schedule": time_schedule,
        "sampling_parameters": {"steps": steps},
        "process": batch.metadata.get("process", {}),
    }
    (destination / "trajectory.json").write_text(
        json.dumps(metadata, separators=(",", ":")) + "\n"
    )
