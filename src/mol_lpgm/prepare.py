from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from .fragmentation import (
    ATOM_NUMBERS,
    SPLIT_CODE,
    connectivity_edges,
    coordinate_molecule as build_coordinate_molecule,
    fragment_molecule,
    sha256,
    smiles_to_coordinate_mapping,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the neutral QM9 source exported by Julia into a "
            "memory-mappable fragmented corpus."
        )
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--length-cutoff", type=int, default=32)
    parser.add_argument("--minimum-fragment-atoms", type=int, default=3)
    parser.add_argument(
        "--minimum-template-rank-ratio",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--split-seed", type=int, default=20260726)
    parser.add_argument("--train-fraction", type=float, default=0.90)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=0,
        help="Optional subset limit; zero processes the complete corpus.",
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_rdkit():
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import BRICS, rdDepictor, rdDetermineBonds
    except ImportError as error:
        raise RuntimeError(
            "QM9 preprocessing requires the optional rdkit dependency"
        ) from error
    return Chem, BRICS, rdDepictor, rdDetermineBonds, rdBase


def _split_codes(
    molecule_count: int,
    *,
    seed: int,
    train_fraction: float,
    valid_fraction: float,
) -> np.ndarray:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train fraction must lie in (0, 1)")
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError("valid fraction must lie in [0, 1)")
    if train_fraction + valid_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum below one")
    permutation = np.random.default_rng(seed).permutation(molecule_count)
    train_end = int(train_fraction * molecule_count)
    valid_end = train_end + int(valid_fraction * molecule_count)
    codes = np.full(molecule_count, SPLIT_CODE["test"], dtype=np.uint8)
    codes[permutation[:train_end]] = SPLIT_CODE["train"]
    codes[permutation[train_end:valid_end]] = SPLIT_CODE["val"]
    return codes


def _hydrogen_nearest_heavy_order(
    molecule,
    atom_numbers: np.ndarray,
) -> np.ndarray:
    heavy = np.flatnonzero(atom_numbers != 1)
    hydrogens_by_parent: dict[int, list[int]] = {}
    for hydrogen in np.flatnonzero(atom_numbers == 1):
        parents = [
            neighbor.GetIdx()
            for neighbor in molecule.GetAtomWithIdx(int(hydrogen)).GetNeighbors()
            if neighbor.GetAtomicNum() != 1
        ]
        if len(parents) != 1:
            raise ValueError("hydrogen does not have exactly one heavy parent")
        hydrogens_by_parent.setdefault(parents[0], []).append(int(hydrogen))
    order: list[int] = []
    for heavy_atom in heavy:
        order.append(int(heavy_atom))
        order.extend(sorted(hydrogens_by_parent.get(int(heavy_atom), [])))
    if len(order) != len(atom_numbers) or len(set(order)) != len(order):
        raise ValueError("hydrogen-nearest-heavy ordering is incomplete")
    return np.asarray(order, dtype=np.int64)


def _write_metadata(
    path: Path,
    *,
    arguments: argparse.Namespace,
    counts: dict,
    rdkit_version: str,
    source: Path,
    source_metadata: dict,
) -> None:
    source_files = (
        "metadata.toml",
        "coords.f32",
        "atom_number.u8",
        "molecule_offsets.i64",
        "smiles.utf8",
        "smiles_offsets.i64",
    )
    lines = [
        "format_version = 1",
        'dataset = "QM9_MurrellLab_MGMisc"',
        'coordinate_units = "angstrom"',
        'atom_ordering = "heavy-atom-then-attached-hydrogens"',
        'fragmentation = "BRICS-minimum-size-and-rank-coarsened"',
        f"length_cutoff = {arguments.length_cutoff}",
        f"minimum_fragment_atoms = {arguments.minimum_fragment_atoms}",
        (
            "minimum_template_rank_ratio = "
            f"{arguments.minimum_template_rank_ratio:.10g}"
        ),
        f'rdkit_version = "{rdkit_version}"',
        "atom_numbers = [" + ", ".join(str(x) for x in ATOM_NUMBERS) + "]",
        f"atom_class_count = {len(ATOM_NUMBERS) + 1}",
        f"mask_class = {len(ATOM_NUMBERS)}",
        f"conformer_count = {counts['conformers']}",
        f"atom_count = {counts['atoms']}",
        f"molecule_group_count = {counts['molecules_kept']}",
        f"fragment_count = {counts['fragments']}",
        f"maximum_atoms = {counts['maximum_atoms']}",
        f"maximum_fragments = {counts['maximum_fragments']}",
        "",
        "[splits]",
        f"train = {counts['split_train']}",
        f"valid = {counts['split_val']}",
        f"test = {counts['split_test']}",
        "",
        "[source]",
        f'root = "{source}"',
        (f'repository = "{source_metadata.get("source_repository", "")}"'),
        f'filename = "{source_metadata.get("source_filename", "")}"',
    ]
    for filename in source_files:
        source_path = source / filename
        lines.extend(
            (
                f'{filename.replace(".", "_")}_path = "{source_path}"',
                f"{filename.replace('.', '_')}_bytes = {source_path.stat().st_size}",
                f'{filename.replace(".", "_")}sha256 = "{sha256(source_path)}"',
            )
        )
    path.write_text("\n".join(lines) + "\n")


def preprocess(arguments: argparse.Namespace) -> dict:
    Chem, BRICS, rdDepictor, rdDetermineBonds, rdBase = _require_rdkit()
    source = Path(arguments.source_dir).resolve()
    output = Path(arguments.output).resolve()
    if output.exists() and any(output.iterdir()) and not arguments.overwrite:
        raise FileExistsError(
            f"{output} is nonempty; pass --overwrite for this exact destination"
        )
    output.mkdir(parents=True, exist_ok=True)

    source_metadata = tomllib.loads((source / "metadata.toml").read_text())
    if source_metadata.get("format_version") != 1:
        raise ValueError("unsupported QM9 source format")
    molecule_count = int(source_metadata["molecule_count"])
    source_atom_count = int(source_metadata["atom_count"])
    if arguments.max_molecules > 0:
        molecule_count = min(molecule_count, arguments.max_molecules)
    molecule_offsets = np.memmap(
        source / "molecule_offsets.i64",
        dtype="<i8",
        mode="r",
        shape=(int(source_metadata["molecule_count"]) + 1,),
    )
    smiles_offsets = np.memmap(
        source / "smiles_offsets.i64",
        dtype="<i8",
        mode="r",
        shape=(int(source_metadata["molecule_count"]) + 1,),
    )
    source_coordinates = np.memmap(
        source / "coords.f32",
        dtype="<f4",
        mode="r",
        shape=(source_atom_count, 3),
    )
    source_numbers = np.memmap(
        source / "atom_number.u8",
        dtype=np.uint8,
        mode="r",
        shape=(source_atom_count,),
    )
    smiles_bytes = np.memmap(
        source / "smiles.utf8",
        dtype=np.uint8,
        mode="r",
    )
    split_code = _split_codes(
        int(source_metadata["molecule_count"]),
        seed=arguments.split_seed,
        train_fraction=arguments.train_fraction,
        valid_fraction=arguments.valid_fraction,
    )

    number_to_class = {
        atomic_number: category for category, atomic_number in enumerate(ATOM_NUMBERS)
    }
    temporary_names = {
        "coords": "coords.f32.tmp",
        "template": "template.f32.tmp",
        "atom_class": "atom_class.u8.tmp",
        "fragment_id": "fragment_id.u16.tmp",
    }
    handles = {
        name: (output / filename).open("wb")
        for name, filename in temporary_names.items()
    }
    offsets = [0]
    kept_source_index: list[int] = []
    kept_split: list[int] = []
    counts = {
        "molecules_seen": 0,
        "molecules_kept": 0,
        "molecules_invalid_smiles": 0,
        "molecules_inconsistent_graph": 0,
        "molecules_inconsistent_atoms": 0,
        "molecules_over_cutoff": 0,
        "molecules_degenerate_fragments": 0,
        "singlefragment_molecules": 0,
        "multifragment_molecules": 0,
        "conformers": 0,
        "atoms": 0,
        "fragments": 0,
        "cut_bonds": 0,
        "restored_bonds": 0,
        "maximum_atoms": 0,
        "maximum_fragments": 0,
        "split_train": 0,
        "split_val": 0,
        "split_test": 0,
    }
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    started = time.monotonic()
    try:
        for molecule_index in range(molecule_count):
            counts["molecules_seen"] += 1
            atom_start = int(molecule_offsets[molecule_index])
            atom_stop = int(molecule_offsets[molecule_index + 1])
            atom_count = atom_stop - atom_start
            if atom_count > arguments.length_cutoff:
                counts["molecules_over_cutoff"] += 1
                continue
            smile_start = int(smiles_offsets[molecule_index])
            smile_stop = int(smiles_offsets[molecule_index + 1])
            smiles = bytes(smiles_bytes[smile_start:smile_stop]).decode("utf-8")
            molecule = Chem.MolFromSmiles(smiles, parser)
            if molecule is None:
                counts["molecules_invalid_smiles"] += 1
                continue
            molecule = Chem.AddHs(molecule)
            if molecule.GetNumAtoms() != atom_count:
                counts["molecules_inconsistent_atoms"] += 1
                continue
            numbers = np.asarray(
                source_numbers[atom_start:atom_stop],
                dtype=np.int64,
            )
            coordinates = np.asarray(
                source_coordinates[atom_start:atom_stop],
                dtype=np.float32,
            )
            if not np.isfinite(coordinates).all() or any(
                number not in number_to_class for number in numbers
            ):
                counts["molecules_inconsistent_atoms"] += 1
                continue
            try:
                coordinate_molecule = build_coordinate_molecule(
                    numbers,
                    coordinates,
                    Chem=Chem,
                    rdDetermineBonds=rdDetermineBonds,
                )
                smiles_to_source = smiles_to_coordinate_mapping(
                    molecule,
                    coordinate_molecule,
                    Chem=Chem,
                )
            except (RuntimeError, ValueError):
                counts["molecules_inconsistent_graph"] += 1
                continue
            graph_numbers = np.asarray(
                [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
                dtype=np.int64,
            )
            if not np.array_equal(graph_numbers, numbers[smiles_to_source]):
                counts["molecules_inconsistent_atoms"] += 1
                continue
            mapped_graph_edges = tuple(
                sorted(
                    tuple(
                        sorted(
                            (
                                int(smiles_to_source[left]),
                                int(smiles_to_source[right]),
                            )
                        )
                    )
                    for left, right in connectivity_edges(molecule)
                )
            )
            if mapped_graph_edges != connectivity_edges(coordinate_molecule):
                counts["molecules_inconsistent_graph"] += 1
                continue

            fragmentation = fragment_molecule(
                molecule,
                minimum_atoms=arguments.minimum_fragment_atoms,
                minimum_rank_ratio=arguments.minimum_template_rank_ratio,
                BRICS=BRICS,
                rdDepictor=rdDepictor,
            )
            if fragmentation.degenerate_fragments:
                counts["molecules_degenerate_fragments"] += 1
                continue
            fragment_id = np.empty_like(fragmentation.fragment_id)
            fragment_id[smiles_to_source] = fragmentation.fragment_id
            template = np.empty_like(fragmentation.template)
            template[smiles_to_source] = fragmentation.template
            order = _hydrogen_nearest_heavy_order(
                coordinate_molecule,
                numbers,
            )
            numbers = numbers[order]
            coordinates = coordinates[order]
            fragment_id = fragment_id[order]
            template = template[order]
            atom_class = np.asarray(
                [number_to_class[int(number)] for number in numbers],
                dtype=np.uint8,
            )
            fragment_count = int(fragment_id.max()) + 1

            np.asarray(coordinates, dtype="<f4").tofile(handles["coords"])
            np.asarray(template, dtype="<f4").tofile(handles["template"])
            atom_class.tofile(handles["atom_class"])
            fragment_id.astype("<u2", copy=False).tofile(handles["fragment_id"])
            code = int(split_code[molecule_index])
            split_name = ("train", "val", "test")[code]
            counts["molecules_kept"] += 1
            counts["conformers"] += 1
            counts["atoms"] += atom_count
            counts["fragments"] += fragment_count
            counts["cut_bonds"] += fragmentation.cut_bonds
            counts["restored_bonds"] += fragmentation.restored_bonds
            counts[f"split_{split_name}"] += 1
            counts[
                (
                    "multifragment_molecules"
                    if fragment_count > 1
                    else "singlefragment_molecules"
                )
            ] += 1
            counts["maximum_atoms"] = max(counts["maximum_atoms"], atom_count)
            counts["maximum_fragments"] = max(
                counts["maximum_fragments"], fragment_count
            )
            offsets.append(counts["atoms"])
            kept_source_index.append(molecule_index)
            kept_split.append(code)
            if (
                arguments.progress_every > 0
                and (molecule_index + 1) % arguments.progress_every == 0
            ):
                print(
                    json.dumps(
                        {
                            "molecules": molecule_index + 1,
                            "kept": counts["molecules_kept"],
                            "multi_fragment": counts["multifragment_molecules"],
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()

    index_arrays = {
        "conformer_offsets.i64": np.asarray(offsets, dtype="<i8"),
        "source_index.i32": np.asarray(kept_source_index, dtype="<i4"),
        "molecule_id.i32": np.asarray(kept_source_index, dtype="<i4"),
        "conformer_id.u8": np.zeros(len(kept_source_index), dtype=np.uint8),
        "split.u8": np.asarray(kept_split, dtype=np.uint8),
    }
    for filename, array in index_arrays.items():
        temporary = output / f"{filename}.tmp"
        array.tofile(temporary)
        os.replace(temporary, output / filename)
    for temporary_name in temporary_names.values():
        temporary = output / temporary_name
        os.replace(temporary, temporary.with_suffix(""))

    _write_metadata(
        output / "metadata.toml",
        arguments=arguments,
        counts=counts,
        rdkit_version=rdBase.rdkitVersion,
        source=source,
        source_metadata=source_metadata,
    )
    summary = {
        **counts,
        "seconds": time.monotonic() - started,
        "length_cutoff": arguments.length_cutoff,
        "minimum_fragment_atoms": arguments.minimum_fragment_atoms,
        "minimum_template_rank_ratio": arguments.minimum_template_rank_ratio,
        "split_seed": arguments.split_seed,
        "train_fraction": arguments.train_fraction,
        "valid_fraction": arguments.valid_fraction,
        "atom_numbers": ATOM_NUMBERS,
        "mask_class": len(ATOM_NUMBERS),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    summary = preprocess(_arguments())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
