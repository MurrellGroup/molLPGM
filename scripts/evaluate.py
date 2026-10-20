"""ADiT molecular validity/uniqueness protocol, plus joint valid-and-connected rate.
Reference: facebookresearch/all-atom-diffusion-transformer, commit b9ce505,
src/eval/molecule_generation.py. OpenBabel construction follows the pymatgen
BabelMolAdaptor used by ADiT; no Pymatgen crystal dependencies are required.
"""

from concurrent.futures import ProcessPoolExecutor
import argparse, hashlib, importlib.metadata, json
from pathlib import Path
import numpy as np
from openbabel import openbabel as ob, pybel
from rdkit import Chem, RDLogger

REFERENCE = "https://github.com/facebookresearch/all-atom-diffusion-transformer/blob/b9ce505f170597a7c8ca50d13ce8e15df21cf8c9/src/eval/molecule_generation.py"


def initialize():
    RDLogger.DisableLog("rdApp.*")
    ob.obErrorLog.StopLogging()


def evaluate(path):
    path = Path(path)
    r = dict(
        file=path.name, valid=False, connected=False, valid_connected=False, smiles=None
    )
    try:
        r["sha256"] = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        xyz = np.loadtxt(path, dtype=str, ndmin=2)
        coords = xyz[:, 1:].astype(float)
        symbols = xyz[:, 0].tolist()
        assert (
            coords.shape == (len(symbols), 3)
            and np.isfinite(coords).all()
            and "X" not in symbols
        )
        nums = [Chem.GetPeriodicTable().GetAtomicNumber(s) for s in symbols]
        mol = ob.OBMol()
        mol.BeginModify()
        for number, position in zip(nums, coords, strict=True):
            atom = ob.OBAtom()
            atom.SetAtomicNum(number)
            atom.SetVector(*map(float, position))
            mol.AddAtom(atom)
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()
        mol.SetTotalSpinMultiplicity(1 if sum(nums) % 2 == 0 else 2)
        mol.SetTotalCharge(0)
        mol.Center()
        mol.EndModify()
        pdb = pybel.Molecule(mol).write("pdb")
        m = Chem.MolFromPDBBlock(pdb, removeHs=True)
        if m is None:
            raise ValueError("RDKit could not sanitize reconstructed molecule")
        Chem.MolToSmiles(m, isomericSmiles=True)
        frags = Chem.GetMolFrags(m, asMols=True)
        largest = max(frags, default=m, key=lambda f: f.GetNumAtoms())
        r["smiles"] = Chem.MolToSmiles(largest, isomericSmiles=True)
        r["valid"] = True
        r["connected"] = len(frags) <= 1
        r["valid_connected"] = r["connected"]
    except Exception as exc:
        r["error"] = f"{type(exc).__name__}: {exc}"
    return r


def summarize(rows):
    n = len(rows)
    valid = [r for r in rows if r["valid"]]
    connected = [r for r in rows if r["valid_connected"]]

    def ratio(count, total):
        return dict(count=count, total=total, fraction=count / total if total else None)

    return dict(
        samples=n,
        metrics=dict(
            valid=ratio(len(valid), n),
            valid_connected=ratio(len(connected), n),
            uniqueness=ratio(len({r["smiles"] for r in valid}), len(valid)),
        ),
        protocol=dict(
            reference=REFERENCE,
            paper="https://arxiv.org/abs/2503.03965",
            bond_perception="OpenBabel ConnectTheDots + PerceiveBondOrders; zero total charge and default electron-parity spin; centered PDB serialization as in Pymatgen BabelMolAdaptor",
            valid="RDKit PDB load/sanitization with removeHs=True and canonical isomeric SMILES conversion",
            valid_connected="Valid and one connected component; denominator all generated samples",
            uniqueness="Distinct canonical isomeric SMILES of largest component among valid samples / valid samples",
            no_coordinate_relaxation=True,
        ),
        versions={
            n: importlib.metadata.version(n)
            for n in ("rdkit", "openbabel-wheel", "numpy")
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = sorted(args.input_dir.glob("*.xyz"))
    assert files, "No sample files found"
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        rows = list(pool.map(evaluate, files, chunksize=16))
    (args.output_dir / "samples.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    summary = summarize(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
