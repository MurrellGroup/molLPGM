# Reference sample quality

10,000 unfiltered samples per model from final 800K EMA weights (decay 0.9999),
with 2,000 square-root-spaced sampling steps. Within each dataset the models use
identical selected cases, atom order, fragment partitions, initial coordinates
and sampling seeds. Fragment Noise uses a fixed global angular variance of 6.

| Dataset | Model | Valid | Valid & connected | Unique among valid | Samples |
| --- | --- | ---: | ---: | ---: | --- |
| GEOM-Drugs | Fragment Noise | 9,842/10,000 (98.42%) | 9,343/10,000 (93.43%) | 9,842/9,842 (100.00%) | [ZIP](geom/fragment/samples.zip) |
| GEOM-Drugs | Brownian Atoms | 8,998/10,000 (89.98%) | 8,636/10,000 (86.36%) | 8,992/8,998 (99.93%) | [ZIP](geom/bm/samples.zip) |
| QM9 | Fragment Noise | 9,863/10,000 (98.63%) | 9,831/10,000 (98.31%) | 9,679/9,863 (98.13%) | [ZIP](qm9/fragment/samples.zip) |
| QM9 | Brownian Atoms | 9,865/10,000 (98.65%) | 9,857/10,000 (98.57%) | 9,664/9,865 (97.96%) | [ZIP](qm9/bm/samples.zip) |

All rows use the same [ADiT evaluator](../scripts/evaluate.py) and library versions.
Validity and valid-and-connected use all 10,000 samples as denominator.
Uniqueness uses distinct canonical isomeric SMILES of the largest connected
component among valid samples, divided by the number of valid samples.
See the [evaluation and comparison protocol](../docs/comparison.md).

GEOM uses 10,000 distinct test molecules; QM9 uses 5,003 validation and 4,997 test
molecules. These experiments condition on the atom counts and BRICS partitions
of the retained cases. Independent training and loss calibration differ between
methods; the table is not a replicated causal estimate.

Each model directory contains the XYZ archive, `sampling.json` with checkpoint
and case provenance, `metrics.json`, and `per-sample.jsonl.gz` with results and
coordinate checksums. XYZ files contain one `element x y z` row per atom, without
a count/comment header. Coordinates are in Å. To reproduce a row:

```bash
pip install -e '.[evaluation]'
unzip results/qm9/fragment/samples.zip -d artifacts/qm9-fragment
python scripts/evaluate.py --input-dir artifacts/qm9-fragment/samples \
  --output-dir artifacts/qm9-fragment-evaluation
```

Archive and result checksums are in [SHA256SUMS](SHA256SUMS).
