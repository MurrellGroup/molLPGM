# Models and training

[Overview](../README.md) · [Mathematics](mathematics.md) · [Viewer](../viewer/README.md)

The fragment and Brownian models share their neural backbone, fragment conditioning, atom-label process, data preparation and optimization schedule. Their coordinate processes, prediction heads and coordinate losses differ. GEOM and QM9 use the same implementation at two model sizes.

## Conditioning and architecture

At time $`t`$, the network receives current coordinates, current noisy atom labels, physical time, and the fragment partition. Atom count is supplied by the conditioning example. Clean coordinates, clean labels, bonds and latent rotation frames are not passed to the network.

Fragment membership enters in every attention layer. For each layer and attention head, a learned two-entry table adds one bias for atoms in the same fragment and another for atoms in different fragments. Atoms may attend across fragments within their molecule. Fragment IDs are used only for equality and pooling, so renumbering fragments does not change the model. Each atom also receives an embedding of its fragment's size. There is no atom-order embedding or RoPE.

The backbone consists of time-conditioned transformer blocks with adaptive LayerNorm, point-distance attention and SwiGLU feedforward layers. Learned query/key points are anchored at the current atom coordinates; their squared distances contribute to attention scores. Attended value points are expressed relative to the query atom. This construction uses relative coordinates but does **not** enforce rotational equivariance: training applies random global rotations. Each head uses four query/key points and eight value points. A Fourier time embedding conditions every block.

After the final normalization, atom features are averaged within each fragment. The fragment model predicts one centroid vector and one angular vector from each pooled feature, and one internal vector per atom. BM instead predicts one Cartesian vector per atom from the concatenation of that atom's feature and its fragment's pooled feature. Both have the same per-atom label head. All backbone layers have the same role; there is no special subset of late geometric layers.

| Dataset | Width | Layers | Feedforward width | Attention heads | Fragment parameters | BM parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GEOM-Drugs | 768 | 16 | 3072 | 8 | 216,378,263 | 216,375,953 |
| QM9 | 512 | 12 | 2048 | 8 | 73,591,607 | 73,590,065 |

Compiled FlexAttention uses block sparsity between packed molecules. Same/different-fragment biases are evaluated inside attention without materializing a dense pair tensor. During backpropagation, per-query temporary storage avoids contention when accumulating gradients into the shared bias table; the parameterization is unchanged. Training uses bfloat16 for the backbone and float32 for geometric reductions and solves.

## Process settings

The equations and objectives are in [Mathematics](mathematics.md). These settings are shared across dataset sizes.

| Setting | Fragment LPGM | BM |
| --- | --- | --- |
| Source standard deviation | 3 Å before molecule centering | Same |
| Centroid/rotation clock | $`s(t)=10t^3-15t^4+6t^5`$ | — |
| Internal clock | $`1-(1-s)^{16}`$ | — |
| Centroid base variance $`\nu_c`$ | $`100/9`$ Å² | — |
| Early centroid diffusion | $`B=3`$ Å, $`\tau=0.1`$ | — |
| Internal base variance $`\nu_q`$ | 0.4 Å² | — |
| Internal mobility | $`\sigma_q=0.15`$ Å, fade from $`t_f=0.8`$ | — |
| Cartesian clock | — | $`u(t)=1-(1-t^{2.3})^{3.1}`$ |
| Cartesian variance coefficient | — | $`\beta=14.8`$ Å² |
| Atom labels | 13 element classes plus mask; shared noisy discrete bridge | Same |

Fragment frames are defined by Kabsch alignment to a deterministic RDKit 2D template. Initial relative rotations are uniform over rotations with angle at most $`0.95\pi`$. The angular variance parameter is the global constant $`\rho^2=6`$, identical for every fragment in GEOM and QM9. It approximately matches the average conditional rotational spread and is a process setting, not a per-example input. Rotation integration uses a maximum step of 0.005 in the rotation clock. The inertia solve for the structured target and prediction gauge uses relative regularization $`10^{-5}`$.

The BM clock parameters approximately match the fragment conditional path's endpoint spread. BM retains fragment membership conditioning but has no fragment dynamics, structured coordinate targets, or internal-target normalization.

## Data

Both datasets include explicit hydrogens. RDKit BRICS cuts define fragments; fragments with fewer than three atoms or a degenerate template are merged with neighbors. Molecules failing graph/coordinate consistency or fragmentation checks are excluded. Training samples conformers uniformly and applies molecule centering and a random global rotation. Physical time is sampled uniformly from $`[0.001,0.999]`$, independently for each molecule.

The shared [BRICS fragmentation implementation](../src/mol_lpgm/fragmentation.py)
includes cutting, deterministic merging, atom mapping, and 2D template creation;
both dataset preparation commands below use it.

| Prepared dataset | Molecules | Conformers | Train conformers | Validation conformers | Test conformers |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEOM-Drugs | 286,799 | 1,381,379 | 1,105,258 | 138,009 | 138,112 |
| QM9 | 124,008 | 124,008 | 111,591 | 6,197 | 6,220 |

GEOM preparation uses the [ADiT GEOM-Drugs release](https://huggingface.co/datasets/chaitjo/GEOM-DRUGS_ADiT) and preserves its molecule-level splits, retaining the full atom-count range (up to 181 atoms in the prepared corpus). QM9 uses [MurrellLab/MGMisc](https://huggingface.co/datasets/MurrellLab/MGMisc/blob/main/qm9.jls); the supplied source has 130,831 molecules before preparation. Its seeded 90/5/5 split is assigned before filtering, giving the retained counts above. The split seed is 20260726. Prepared metadata records the source checksums and inclusion statistics.

## Optimization

| Setting | Both models |
| --- | --- |
| Training length | 800,000 optimizer updates |
| Batch size | Two independently packed microbatches of up to 4096 atoms per update |
| Muon | Residual matrices; momentum 0.95, Nesterov, five Newton–Schulz steps |
| AdamW | Embeddings, time conditioning, normalization, biases and output heads; $`\beta_1=0.9`$, $`\beta_2=0.95`$ |
| Weight decay | 0.01 |
| Peak learning rates | Muon 0.02; AdamW 0.002 |
| Warmup | Linear, first 10,000 updates |
| Main decay | Exponential to Muon 0.00134402 at update 700,000 |
| Warmdown | Linear over updates 700,000–800,000 to Muon $`10^{-9}`$ |
| AdamW learning rate | Always 0.1 times the Muon rate |
| EMA | Decay 0.9999, updated every optimizer step |
| Validation | 64 fixed batches every 5,000 updates |

The coordinate and atom-label losses are defined on the [mathematics page](mathematics.md). The fixed mean-logit penalty has coefficient 0.01. EMA weights are used for sampling.

### Loss balancing

Losses are grouped into ten equal-width bins of physical time. There are 40 component/time cells for fragment LPGM (centroid, rotation, internal, label) and 20 for BM (coordinate, label). Fragment training starts with the supplied positive time curves and multipliers in [fragment-weights.json](../configs/fragment-weights.json); BM starts with unit multipliers on its base coordinate and label losses in [bm-weights.json](../configs/bm-weights.json).

At update 100,000, the pipeline measures each cell's mean loss $`L_j`$ and full-model RMS gradient norm $`G_j`$ on 64 fresh microbatches, combined in pairs as in training. It multiplies the existing cell weights by

```math
m_j=\frac{C}{L_j^{0.25}G_j^{0.75}}.
```

The common factor $`C`$ preserves the RMS norm of the combined gradient of the balanced terms, using their measured gradient cross-products. Cell contributions retain full-batch denominators, so they sum to the training objective. A separate set of 32 fresh microbatches checks the resulting balance. Accepted weights remain fixed for the rest of training; the mean-logit penalty is outside this reweighting. Training resumes with the same model, optimizer, EMA history and absolute learning-rate schedule. The GEOM Fragment Noise reference uses its initial weights throughout; the other three reference models use accepted 100K weights. The pipeline stops for review if the independent calibration check does not pass.

## Sampling and evaluation

Both samplers use 2,000 steps on the physical-time grid $`t_k=\sqrt{k/2000}`$ for the reference comparisons. Euclidean bridge coefficients are analytic. The fragment sampler combines centroid updates, internal updates and shared rotations; BM uses only Cartesian updates. Network predictions are refreshed once per sampling step, and labels use the shared discrete Euler update. Analytic conditional coefficients do not make the learned finite-step sampler exact.

For comparisons, both models receive matched held-out atom counts and fragment partitions and use the same initial coordinate seeds. Trajectories and final samples are recorded in the same execution. Metrics are computed on every requested sample, before any viewer filtering:

- **Valid:** ADiT OpenBabel bond perception and PDB serialization followed by
  RDKit sanitization and canonical isomeric SMILES conversion.
- **Valid & connected:** valid and a single reconstructed component.
- **Uniqueness:** distinct largest-component SMILES among valid samples,
  divided by the number of valid samples.

Validity and valid-and-connected use all generated samples as their denominator.
The same [ADiT evaluator](../scripts/evaluate.py) and pinned library versions are
used for GEOM and QM9. See the [comparison protocol](comparison.md) and
[10K results](../results/README.md). Viewer filters affect display only.

## Running the models

Run commands from the repository root after the [installation steps](../README.md#getting-started). The reference environment uses PyTorch 2.14.0+cu130 and Julia 1.11; Julia dependencies are pinned in [julia/Manifest.toml](../julia/Manifest.toml).

### Prepare data

~~~bash
julia --project=julia julia/scripts/export_qm9_source.jl --output=data/qm9-source
python -m mol_lpgm.prepare --source-dir data/qm9-source --output data/qm9
~~~

For GEOM:

~~~bash
pip install -e '.[geom]'
julia --project=julia julia/scripts/download_geom_source.jl --output=data/geom-source
python -m mol_lpgm.geom_prepare --geom-pt data/geom-source/geom.pt \
  --train-smiles data/geom-source/train_smiles.pt \
  --val-smiles data/geom-source/val_smiles.pt \
  --test-smiles data/geom-source/test_smiles.pt --output data/geom
~~~

### Train

Launch the models in separate terminals, one per GPU:

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_pipeline.py \
  --config configs/qm9-fragment.toml --corpus data/qm9 --run-dir runs/qm9-fragment
~~~

~~~bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_pipeline.py \
  --config configs/qm9-bm.toml --corpus data/qm9 --run-dir runs/qm9-bm
~~~

For GEOM, substitute [configs/geom-fragment.toml](../configs/geom-fragment.toml) or [configs/geom-bm.toml](../configs/geom-bm.toml) and the GEOM corpus path. Each run writes loss plots every 1,000 updates and checkpoints containing model, optimizer and EMA state, process parameters and calibrated weights.

### Sample and evaluate

Final GEOM and QM9 checkpoints are available from [MurrellLab/LPGM](https://huggingface.co/MurrellLab/LPGM). Download URLs and SHA-256 checksums are in [checkpoints.json](../checkpoints.json). Each file contains raw and EMA weights; use `--ema` for the EMA samples. Published files omit optimizer state and are intended for sampling.

Create matched conditioning batches using the same dataset, seed and sample count:

~~~bash
julia --project=julia julia/scripts/write_sampling_batches.jl \
  --domain=qm9 --corpus-dir=data/qm9 --out-dir=batches/fragment --samples=1024
julia --project=julia julia/scripts/write_sampling_batches.jl \
  --domain=qm9 --corpus-dir=data/qm9 --out-dir=batches/bm --samples=1024 \
  --atom-bm-profile=configs/processes/bm.toml
python scripts/sample_panel.py --checkpoint checkpoints/qm9/fragment.pt \
  --ema --template-root batches/fragment --output-dir samples/fragment \
  --viewer-dir artifacts/fragment --samples 1024 --steps 2000 --time-schedule sqrt
python scripts/evaluate.py --input-dir samples/fragment --output-dir artifacts/evaluation/fragment
~~~

Repeat the last two commands with the BM checkpoint and batches. For GEOM, use the geom-drugs domain, the corresponding corpus, and the downloaded [GEOM checkpoints](../checkpoints.json). Publish both panels with the instructions in [viewer/README.md](../viewer/README.md).

### Implementation checks

~~~bash
pytest
julia --project=julia julia/test/runtests.jl
node --test viewer/test.mjs
~~~

The main implementations are [conditional processes](../julia/src/process.jl), [neural architecture](../src/mol_lpgm/model.py), [structured objective](../src/mol_lpgm/structured.py), [BM objective](../src/mol_lpgm/cartesian_loss.py), and [sampling](../src/mol_lpgm/sampler.py).
