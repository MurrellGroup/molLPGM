from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

from .fragmentation import (
    ATOM_NUMBERS,
    SPLIT_CODE,
    connectivity_edges,
    coordinate_molecule,
    fragment_molecule,
    sha256,
    smiles_to_coordinate_mapping,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the ADiT GEOM-Drugs tensor and aligned explicit-H "
            "SMILES into the memory-mappable molecular LPGM corpus format."
        )
    )
    parser.add_argument("--geom-pt", required=True)
    parser.add_argument("--train-smiles", required=True)
    parser.add_argument("--val-smiles", required=True)
    parser.add_argument("--test-smiles", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--length-cutoff",
        type=int,
        default=0,
        help="Maximum explicit atom count; zero retains the full range.",
    )
    parser.add_argument("--minimum-fragment-atoms", type=int, default=3)
    parser.add_argument("--minimum-template-rank-ratio", type=float, default=1e-3)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="Optional molecule-group limit; zero means all groups.",
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_dependencies():
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import BRICS, rdDepictor, rdDetermineBonds
    except ImportError as error:  # pragma: no cover - environment failure
        raise RuntimeError(
            "GEOM preprocessing requires the optional rdkit dependency"
        ) from error
    try:
        # The serialized public tensor contains PyG storage classes.
        import torch_geometric  # noqa: F401
    except ImportError as error:  # pragma: no cover - environment failure
        raise RuntimeError(
            "loading the public GEOM tensor requires torch-geometric"
        ) from error
    return Chem, BRICS, rdDepictor, rdDetermineBonds, rdBase


def _graph_signature(molecule, *, Chem) -> tuple[int, int, str]:
    """Conformer-independent molecular identity used for group validation."""
    return (
        molecule.GetNumAtoms(),
        molecule.GetNumBonds(),
        Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=False,
            allHsExplicit=True,
        ),
    )


def _molecule_groups(identifiers: Sequence[str]) -> list[tuple[int, int]]:
    if not identifiers:
        return []
    groups: list[tuple[int, int]] = []
    start = 0
    previous = identifiers[0].rsplit("_", 1)[0]
    for index in range(1, len(identifiers)):
        current = identifiers[index].rsplit("_", 1)[0]
        if current != previous:
            groups.append((start, index))
            start = index
            previous = current
    groups.append((start, len(identifiers)))
    return groups


def _exceeds_length_cutoff(atom_count: int, length_cutoff: int) -> bool:
    if length_cutoff < 0:
        raise ValueError("length cutoff cannot be negative")
    return length_cutoff > 0 and atom_count > length_cutoff


def _parse_identifier(identifier: str) -> tuple[str, int, int]:
    prefix, conformer = identifier.rsplit("_", 1)
    prefix, molecule = prefix.rsplit("_", 1)
    split = prefix.removeprefix("geom_")
    if split not in SPLIT_CODE:
        raise ValueError(f"unrecognized GEOM identifier {identifier!r}")
    conformer_number = int(conformer)
    if not 0 <= conformer_number <= np.iinfo(np.uint8).max:
        raise ValueError("GEOM conformer index exceeds the uint8 corpus format")
    return split, int(molecule), conformer_number


def _write_metadata(
    path: Path,
    *,
    arguments: argparse.Namespace,
    counts: dict,
    rdkit_version: str,
    source_paths: dict[str, Path],
) -> None:
    lines = [
        "format_version = 1",
        'dataset = "GEOM-DRUGS_ADiT"',
        'coordinate_units = "angstrom"',
        'fragmentation = "BRICS-minimum-size-and-rank-coarsened"',
        f"length_cutoff = {arguments.length_cutoff}",
        f"minimum_fragment_atoms = {arguments.minimum_fragment_atoms}",
        (
            "minimum_template_rank_ratio = "
            f"{arguments.minimum_template_rank_ratio:.10g}"
        ),
        f"rdkit_version = {json.dumps(rdkit_version)}",
        "atom_numbers = [" + ", ".join(str(x) for x in ATOM_NUMBERS) + "]",
        f"atom_class_count = {len(ATOM_NUMBERS) + 1}",
        f"mask_class = {len(ATOM_NUMBERS)}",
        f"conformer_count = {counts['conformers']}",
        f"atom_count = {counts['atoms']}",
        f"molecule_group_count = {counts['groups_kept']}",
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
    ]
    for name, source_path in source_paths.items():
        lines.extend(
            (
                f"{name}_path = {json.dumps(str(source_path))}",
                f"{name}_bytes = {source_path.stat().st_size}",
                f'{name}_sha256 = "{sha256(source_path)}"',
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _load_source(arguments: argparse.Namespace):
    geom_path = Path(arguments.geom_pt).resolve()
    smile_paths = {
        "train_smiles": Path(arguments.train_smiles).resolve(),
        "val_smiles": Path(arguments.val_smiles).resolve(),
        "test_smiles": Path(arguments.test_smiles).resolve(),
    }
    for source in (geom_path, *smile_paths.values()):
        if not source.is_file():
            raise FileNotFoundError(source)
    data, slices, _data_class = torch.load(
        geom_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    identifiers = [str(value) for value in data["id"]]
    smiles: list[str] = []
    for source in smile_paths.values():
        smiles.extend(
            torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        )
    if len(smiles) != len(identifiers):
        raise ValueError(
            f"{len(smiles)} SMILES do not align with " f"{len(identifiers)} conformers"
        )
    return geom_path, smile_paths, data, slices, identifiers, smiles


def preprocess(arguments: argparse.Namespace) -> dict:
    Chem, BRICS, rdDepictor, rdDetermineBonds, rdBase = _require_dependencies()
    if arguments.length_cutoff < 0:
        raise ValueError("length cutoff cannot be negative")
    output = Path(arguments.output).resolve()
    if output.exists() and any(output.iterdir()) and not arguments.overwrite:
        raise FileExistsError(
            f"{output} is nonempty; pass --overwrite for this exact destination"
        )
    output.mkdir(parents=True, exist_ok=True)

    (
        geom_path,
        smile_paths,
        data,
        slices,
        identifiers,
        smiles,
    ) = _load_source(arguments)
    groups = _molecule_groups(identifiers)
    if arguments.max_groups > 0:
        groups = groups[: arguments.max_groups]

    parser = Chem.SmilesParserParams()
    parser.removeHs = False
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
    source_index: list[int] = []
    molecule_id: list[int] = []
    conformer_id: list[int] = []
    split_code: list[int] = []
    counts = {
        "groups_seen": 0,
        "groups_kept": 0,
        "groups_invalid_smiles": 0,
        "groups_inconsistent_graph": 0,
        "groups_inconsistent_atoms": 0,
        "groups_over_cutoff": 0,
        "groups_degenerate_fragments": 0,
        "conformers": 0,
        "atoms": 0,
        "fragments": 0,
        "cut_bonds": 0,
        "restored_bonds": 0,
        "nonfinite_conformers": 0,
        "maximum_atoms": 0,
        "maximum_fragments": 0,
        "split_train": 0,
        "split_val": 0,
        "split_test": 0,
    }
    started = time.monotonic()
    try:
        for group_number, (start, stop) in enumerate(groups, start=1):
            counts["groups_seen"] += 1
            molecule = Chem.MolFromSmiles(smiles[start], parser)
            if molecule is None:
                counts["groups_invalid_smiles"] += 1
                continue
            signature = _graph_signature(molecule, Chem=Chem)
            if any(
                variant_molecule is None
                or _graph_signature(variant_molecule, Chem=Chem) != signature
                for variant in set(smiles[start:stop])
                if variant != smiles[start]
                for variant_molecule in [Chem.MolFromSmiles(variant, parser)]
            ):
                counts["groups_inconsistent_graph"] += 1
                continue

            atom_count = molecule.GetNumAtoms()
            if _exceeds_length_cutoff(atom_count, arguments.length_cutoff):
                counts["groups_over_cutoff"] += 1
                continue
            graph_numbers = np.asarray(
                [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
                dtype=np.int64,
            )
            if any(number not in number_to_class for number in graph_numbers):
                counts["groups_inconsistent_atoms"] += 1
                continue

            representative_start = int(slices["pos"][start])
            representative_stop = int(slices["pos"][start + 1])
            source_numbers = np.asarray(
                data["atom_types"][representative_start:representative_stop]
                .cpu()
                .numpy(),
                dtype=np.int64,
            )
            source_coordinates = np.asarray(
                data["pos"][representative_start:representative_stop].cpu().numpy(),
                dtype=np.float32,
            )
            if len(source_numbers) != atom_count:
                counts["groups_inconsistent_atoms"] += 1
                continue
            try:
                coordinate_graph = coordinate_molecule(
                    source_numbers,
                    source_coordinates,
                    Chem=Chem,
                    rdDetermineBonds=rdDetermineBonds,
                )
                smiles_to_source = smiles_to_coordinate_mapping(
                    molecule, coordinate_graph, Chem=Chem
                )
            except (RuntimeError, ValueError):
                counts["groups_inconsistent_graph"] += 1
                continue
            if not np.array_equal(graph_numbers, source_numbers[smiles_to_source]):
                counts["groups_inconsistent_atoms"] += 1
                continue

            reference_edges = connectivity_edges(coordinate_graph)
            fragmentation = fragment_molecule(
                molecule,
                minimum_atoms=arguments.minimum_fragment_atoms,
                minimum_rank_ratio=arguments.minimum_template_rank_ratio,
                BRICS=BRICS,
                rdDepictor=rdDepictor,
            )
            if fragmentation.degenerate_fragments:
                counts["groups_degenerate_fragments"] += 1
                continue
            source_fragment_id = np.empty_like(fragmentation.fragment_id)
            source_fragment_id[smiles_to_source] = fragmentation.fragment_id
            source_template = np.empty_like(fragmentation.template)
            source_template[smiles_to_source] = fragmentation.template
            atom_class = np.asarray(
                [number_to_class[int(number)] for number in source_numbers],
                dtype=np.uint8,
            )
            fragment_count = int(source_fragment_id.max()) + 1

            valid_group = True
            for source_conformer in range(start, stop):
                atom_start = int(slices["pos"][source_conformer])
                atom_stop = int(slices["pos"][source_conformer + 1])
                conformer_numbers = (
                    data["atom_types"][atom_start:atom_stop].cpu().numpy()
                )
                if atom_stop - atom_start != atom_count or not np.array_equal(
                    conformer_numbers, source_numbers
                ):
                    valid_group = False
                    break
                coordinates = data["pos"][atom_start:atom_stop].cpu().numpy()
                if not np.isfinite(coordinates).all():
                    continue
                try:
                    conformer_graph = coordinate_molecule(
                        conformer_numbers,
                        coordinates,
                        Chem=Chem,
                        rdDetermineBonds=rdDetermineBonds,
                    )
                except RuntimeError:
                    valid_group = False
                    break
                if connectivity_edges(conformer_graph) != reference_edges:
                    valid_group = False
                    break
            if not valid_group:
                counts["groups_inconsistent_graph"] += 1
                continue

            wrote_group = False
            for original_index in range(start, stop):
                atom_start = int(slices["pos"][original_index])
                atom_stop = int(slices["pos"][original_index + 1])
                coordinates = data["pos"][atom_start:atom_stop].cpu().numpy()
                if not np.isfinite(coordinates).all():
                    counts["nonfinite_conformers"] += 1
                    continue
                split, molecule_number, conformer_number = _parse_identifier(
                    identifiers[original_index]
                )
                np.asarray(coordinates, dtype="<f4").tofile(handles["coords"])
                np.asarray(source_template, dtype="<f4").tofile(handles["template"])
                atom_class.tofile(handles["atom_class"])
                source_fragment_id.astype("<u2", copy=False).tofile(
                    handles["fragment_id"]
                )
                counts["conformers"] += 1
                counts["atoms"] += atom_count
                counts["fragments"] += fragment_count
                counts[f"split_{split}"] += 1
                counts["maximum_atoms"] = max(counts["maximum_atoms"], atom_count)
                counts["maximum_fragments"] = max(
                    counts["maximum_fragments"], fragment_count
                )
                offsets.append(counts["atoms"])
                source_index.append(original_index)
                molecule_id.append(molecule_number)
                conformer_id.append(conformer_number)
                split_code.append(SPLIT_CODE[split])
                wrote_group = True
            if wrote_group:
                counts["groups_kept"] += 1
                counts["cut_bonds"] += fragmentation.cut_bonds
                counts["restored_bonds"] += fragmentation.restored_bonds
            if (
                arguments.progress_every > 0
                and group_number % arguments.progress_every == 0
            ):
                print(
                    json.dumps(
                        {
                            "groups": group_number,
                            "conformers": counts["conformers"],
                            "atoms": counts["atoms"],
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
        "source_index.i32": np.asarray(source_index, dtype="<i4"),
        "molecule_id.i32": np.asarray(molecule_id, dtype="<i4"),
        "conformer_id.u8": np.asarray(conformer_id, dtype=np.uint8),
        "split.u8": np.asarray(split_code, dtype=np.uint8),
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
        source_paths={"geom": geom_path, **smile_paths},
    )
    summary = {
        **counts,
        "seconds": time.monotonic() - started,
        "length_cutoff": arguments.length_cutoff,
        "minimum_fragment_atoms": arguments.minimum_fragment_atoms,
        "minimum_template_rank_ratio": arguments.minimum_template_rank_ratio,
        "atom_numbers": ATOM_NUMBERS,
        "mask_class": len(ATOM_NUMBERS),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    print(json.dumps(preprocess(_arguments()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
