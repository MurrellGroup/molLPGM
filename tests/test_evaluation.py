import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "evaluate", Path(__file__).parents[1] / "scripts/evaluate.py"
)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def test_validity_and_connectivity_are_distinct(tmp_path):
    evaluator.initialize()
    water = tmp_path / "water.xyz"
    water.write_text("O 0 0 0\nH .96 0 0\nH -.24 .93 0\n")
    row = evaluator.evaluate(water)
    assert row["valid"] and row["connected"]
    separated = tmp_path / "separated.xyz"
    separated.write_text("H 0 0 0\nH .74 0 0\nH 10 0 0\nH 10.74 0 0\n")
    row = evaluator.evaluate(separated)
    assert row["valid"] and not row["connected"]
    unresolved = tmp_path / "unresolved.xyz"
    unresolved.write_text("X 0 0 0\n")
    row = evaluator.evaluate(unresolved)
    assert not row["valid"] and not row["connected"]


def test_uniqueness_uses_largest_component_among_valid_samples(tmp_path):
    evaluator.initialize()
    water = tmp_path / "water.xyz"
    water.write_text("O 0 0 0\nH .96 0 0\nH -.24 .93 0\n")
    disconnected = tmp_path / "disconnected.xyz"
    disconnected.write_text(water.read_text() + "O 10 0 0\nH 10.96 0 0\nH 9.76 .93 0\n")
    invalid = tmp_path / "invalid.xyz"
    invalid.write_text("X 0 0 0\n")
    rows = [evaluator.evaluate(p) for p in (water, disconnected, invalid)]
    assert rows[0]["smiles"] == rows[1]["smiles"] == "O"
    summary = evaluator.summarize(rows)
    assert set(summary["metrics"]) == {"valid", "valid_connected", "uniqueness"}
    assert summary["metrics"]["valid"]["count"] == 2
    assert summary["metrics"]["valid_connected"]["count"] == 1
    assert summary["metrics"]["valid_connected"]["total"] == 3
    assert summary["metrics"]["uniqueness"] == {"count": 1, "total": 2, "fraction": 0.5}
