from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

# Element vocabulary shared across both datasets. QM9
# itself uses only H, C, N, O, F; the final index is the mask/source class.
ATOM_NUMBERS = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 83)
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class Fragmentation:
    fragment_id: np.ndarray
    template: np.ndarray
    cut_bonds: int
    restored_bonds: int
    degenerate_fragments: int


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> bool:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return False
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        return True


def coordinate_molecule(atom_numbers, coordinates, *, Chem, rdDetermineBonds):
    editable = Chem.RWMol()
    for number in atom_numbers:
        editable.AddAtom(Chem.Atom(int(number)))
    molecule = editable.GetMol()
    conformer = Chem.Conformer(len(atom_numbers))
    for index, coordinate in enumerate(coordinates):
        conformer.SetAtomPosition(index, tuple(float(value) for value in coordinate))
    molecule.AddConformer(conformer)
    rdDetermineBonds.DetermineConnectivity(molecule)
    return molecule


def connectivity_edges(molecule) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            for bond in molecule.GetBonds()
        )
    )


def smiles_to_coordinate_mapping(
    graph_molecule, coordinate_molecule, *, Chem
) -> np.ndarray:
    def heavy_subgraph(molecule):
        original_indices = [
            atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
        ]
        original_to_local = {
            original: local for local, original in enumerate(original_indices)
        }
        editable = Chem.RWMol()
        for original in original_indices:
            editable.AddAtom(
                Chem.Atom(molecule.GetAtomWithIdx(original).GetAtomicNum())
            )
        for bond in molecule.GetBonds():
            left = bond.GetBeginAtomIdx()
            right = bond.GetEndAtomIdx()
            if left in original_to_local and right in original_to_local:
                editable.AddBond(
                    original_to_local[left],
                    original_to_local[right],
                    Chem.BondType.SINGLE,
                )
        return editable.GetMol(), original_indices

    graph_heavy, graph_indices = heavy_subgraph(graph_molecule)
    coordinate_heavy, coordinate_indices = heavy_subgraph(coordinate_molecule)
    parameters = Chem.SubstructMatchParameters()
    parameters.useChirality = False
    parameters.uniquify = True
    parameters.maxMatches = 1
    parameters.maxRecursiveMatches = 1_000
    match = coordinate_heavy.GetSubstructMatch(graph_heavy, parameters)
    if len(match) != len(graph_indices):
        raise ValueError("SMILES and coordinate heavy-atom graphs differ")

    mapping = np.full(graph_molecule.GetNumAtoms(), -1, dtype=np.int64)
    for graph_local, coordinate_local in enumerate(match):
        mapping[graph_indices[graph_local]] = coordinate_indices[coordinate_local]

    def hydrogens_by_parent(molecule):
        result: dict[int, list[int]] = {}
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() != 1:
                continue
            parents = [
                neighbor.GetIdx()
                for neighbor in atom.GetNeighbors()
                if neighbor.GetAtomicNum() != 1
            ]
            if len(parents) != 1:
                raise ValueError("explicit hydrogen lacks one heavy parent")
            result.setdefault(parents[0], []).append(atom.GetIdx())
        return result

    graph_hydrogens = hydrogens_by_parent(graph_molecule)
    coordinate_hydrogens = hydrogens_by_parent(coordinate_molecule)
    for graph_parent, graph_children in graph_hydrogens.items():
        coordinate_parent = int(mapping[graph_parent])
        coordinate_children = coordinate_hydrogens.get(coordinate_parent, [])
        if len(graph_children) != len(coordinate_children):
            raise ValueError("explicit hydrogen counts differ")
        for graph_child, coordinate_child in zip(
            sorted(graph_children), sorted(coordinate_children)
        ):
            mapping[graph_child] = coordinate_child
    if (mapping < 0).any() or len(set(mapping.tolist())) != len(mapping):
        raise ValueError("SMILES-to-coordinate mapping is incomplete")
    return mapping


def _components(union_find: _UnionFind, atom_count: int) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for atom in range(atom_count):
        result.setdefault(union_find.find(atom), []).append(atom)
    return result


def _template_is_nondegenerate(
    atoms: Sequence[int], template: np.ndarray, minimum_rank_ratio: float
) -> bool:
    if len(atoms) < 3:
        return False
    xy = template[np.asarray(atoms), :2]
    xy = xy - xy.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(xy, compute_uv=False)
    return bool(
        singular_values[0] > 1e-6
        and singular_values[1] >= minimum_rank_ratio * singular_values[0]
    )


def fragment_molecule(
    molecule,
    *,
    minimum_atoms: int,
    minimum_rank_ratio: float,
    BRICS,
    rdDepictor,
) -> Fragmentation:
    atom_count = molecule.GetNumAtoms()
    rdDepictor.Compute2DCoords(molecule, canonOrient=True, clearConfs=True)
    conformer = molecule.GetConformer()
    template = np.asarray(
        [
            (
                conformer.GetAtomPosition(index).x,
                conformer.GetAtomPosition(index).y,
                0.0,
            )
            for index in range(atom_count)
        ],
        dtype=np.float32,
    )
    cuts = tuple(
        sorted(
            tuple(sorted(atom_pair))
            for atom_pair, _labels in BRICS.FindBRICSBonds(molecule)
        )
    )
    cut_set = set(cuts)
    union_find = _UnionFind(atom_count)
    for bond in molecule.GetBonds():
        pair = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        if pair not in cut_set:
            union_find.union(*pair)

    restored = 0
    while True:
        components = _components(union_find, atom_count)
        bad_roots = {
            root
            for root, atoms in components.items()
            if len(atoms) < minimum_atoms
            or not _template_is_nondegenerate(atoms, template, minimum_rank_ratio)
        }
        if not bad_roots:
            break
        candidates = []
        for left, right in cuts:
            left_root = union_find.find(left)
            right_root = union_find.find(right)
            if left_root == right_root:
                continue
            if left_root not in bad_roots and right_root not in bad_roots:
                continue
            candidates.append(
                (
                    min(union_find.size[left_root], union_find.size[right_root]),
                    left,
                    right,
                )
            )
        if not candidates:
            break
        _, left, right = min(candidates)
        restored += int(union_find.union(left, right))

    ordered = sorted(_components(union_find, atom_count).values(), key=min)
    fragment_id = np.empty(atom_count, dtype=np.uint16)
    degenerate = 0
    for fragment, atoms in enumerate(ordered):
        fragment_id[np.asarray(atoms)] = fragment
        degenerate += int(
            not _template_is_nondegenerate(atoms, template, minimum_rank_ratio)
        )
    return Fragmentation(
        fragment_id=fragment_id,
        template=template,
        cut_bonds=len(cuts),
        restored_bonds=restored,
        degenerate_fragments=degenerate,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
