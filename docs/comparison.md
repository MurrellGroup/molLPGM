# Model comparison and evaluation

Fragment Noise and Brownian Atoms share the backbone, fragment membership
conditioning in every attention layer, atom-label process, Gaussian source
standard deviation of 3 Å, effective batch size, optimizer schedule, and EMA
decay. All four models use 800K updates, including a 100K warmdown. GEOM uses
768-wide, 16-layer networks; QM9 uses 512-wide, 12-layer networks. The small
parameter-count differences arise from the coordinate output heads.

## Information used at generation

Both methods receive the atom count and fragment partition, which determines
fragment sizes. Their networks receive current coordinates, noisy atom labels,
physical time, and this partition. Neither the networks nor the samplers use
clean coordinates, clean atom labels, bonds, fragment templates, or other
per-example geometric information. Fragment Noise uses one global angular
variance, $`\rho^2=6`$, for every fragment in both datasets.

The fragment partition is derived from each held-out molecular graph with the
same BRICS preparation for both methods. These are partition-conditioned
experiments, rather than unconditional generation over atom counts and
partitions.

## Objectives and optimization

The fragment objective separates centroid, angular, internal-coordinate, and
atom-label errors; the Brownian objective has Cartesian-coordinate and atom-label
terms. Fragment coordinate errors average over fragments, while Brownian
coordinate errors average over atoms. Their coordinate target normalization and
base time weighting also differ, as described in the [mathematics](mathematics.md).

Fragment training starts with the supplied component/time weights; Brownian
training starts with unit bin multipliers. At 100K, the same loss/gradient
balancing rule proposes updated weights, checked on independent batches. The
GEOM Fragment Noise checkpoint uses its initial weights throughout; QM9 Fragment
Noise and both Brownian Atoms checkpoints use their accepted 100K weights
thereafter. Exact final weights are stored in each checkpoint. The model,
optimizer, EMA history, and absolute learning-rate schedule continue across this
boundary.

These are single independently trained runs, not replicated estimates over
training seeds. Paired sampling controls the selected cases and initial
coordinates, but does not remove training variation or objective differences.

## Sampling and evaluation sets

The [four reference sample sets](../results/README.md) each contain 10,000
unfiltered samples from final EMA weights, using 2,000 square-root-spaced steps.
Within each dataset, both methods share cases, atom order, fragment assignments,
and identical initial coordinates. All requested samples enter the metrics.

GEOM uses 10,000 distinct test molecules. QM9 uses 10,000 distinct cases pooled
from validation and test: **5,003 validation and 4,997 test**. The QM9 result is
therefore a pooled held-out evaluation, not a test-only result. Both datasets use
the same BRICS preprocessing and exclusion rules for both methods; evaluation
is conditional on the retained, fragmentable cases.

## ADiT metrics

The only evaluator is [scripts/evaluate.py](../scripts/evaluate.py). It follows
[ADiT's molecular validity and uniqueness protocol](https://github.com/facebookresearch/all-atom-diffusion-transformer/blob/b9ce505f170597a7c8ca50d13ce8e15df21cf8c9/src/eval/molecule_generation.py):
OpenBabel bond perception, PDB serialization, and RDKit sanitization and canonical
isomeric SMILES conversion. The `evaluation` installation extra pins the NumPy,
RDKit and OpenBabel versions used for the reference results.

- **Valid:** successful reconstruction, sanitization and SMILES conversion,
  divided by all generated samples.
- **Valid & connected:** valid with one reconstructed connected component,
  divided by all generated samples.
- **Uniqueness:** distinct canonical isomeric SMILES of the largest connected
  component among valid samples, divided by the number of valid samples.

No coordinate relaxation or quality selection is applied.
Per-sample records include a SHA-256 checksum of
the evaluated coordinates. The viewer's display bonds do not define any metric.

## Path figures

The [figure measurements](../figures/README.md) use 100 paired GEOM cases with at
least two fragments, selected without quality filtering. Generated-path errors
refer to each model's own final sample. Conditional-path pairs share both source
and dataset endpoint. Neither plot measures error to an independently generated
reference structure, and neither is itself a chemical-quality metric.
