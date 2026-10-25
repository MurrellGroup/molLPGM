# Models and training

[Overview](../README.md) · [Mathematics](mathematics.md) · [Viewer](../viewer/README.md)

The fragment ("Fragment Noise") and atom-level ("Brownian Atoms") models share: i) a core architecture, ii) fragment conditioning, where each atom is informed of the fragment it belongs to, iii) a discrete atom-label process, iv) data preparation and v) optimization budget and schedule. They differ in their coordinate processes (and corresponding coordinate loss) and prediction heads. GEOM and QM9 use the same implementation, differing only QM9 using a smaller model.

## Conditioning and architecture

### Methods summary

Both Fragment Noise and Brownian Atoms use a time-conditioned atom transformer. The key layer is adapted from Invariant Point Attention (IPA) [Jumper et al. (2021), Supplementary Methods §1.8.2 and Algorithm 22](https://doi.org/10.1038/s41586-021-03819-2), modified to only preserve translation invariance (since, unlike the backbone amino acid "frames", atoms to not have an intrinsic observable orientation). Thus each atom defines a frame whose origin is its current position, but whose rotation is the identity. As with the original IPA, attention combines scalar query–key products, and squared distances between learned query/key points (projected from their associated atom's centroid). While the original IPA optionally allows dense pair features, here we use a simple trainable per-head same/different-fragment bias. This simplification allows us to exploit the factorization used in **FlashIPA** ([Liu et al., 2025, §2.4 and Algorithm 1](https://proceedings.neurips.cc/paper_files/paper/2025/hash/07ea874e9e4f71ec6680a3574a485a36-Abstract-Conference.html)), where squared point distances are expanded into augmented query–key dot products. We use FlexAttention to handle this and the fragment bias without materializing a dense attention matrix.

We use Fourier time conditioning with **AdaLN-Zero** residual blocks ([Peebles and Xie, 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)) and **SwiGLU** feedforward layers ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)). Fragment-size embeddings are summed into the starting atom features. The model is also permutation equivariant, without any atom-order positional encoding. The GEOM model uses 16 layers of width 768 and QM9 uses 12 layers of width 512, both using eight attention heads per layer, four query/key points and eight value points per head, and feedforward width four times the model width.

After the transformer "trunk", the Fragment Noise model predicts centroid and angular vectors from mean-pooled fragment features and per-atom vectors from atom features. Brownian Atoms predicts one Cartesian endpoint displacement per atom. Both predict endpoint element logits from atom features. As the model is translationally equivariant, training and sampling operate in the subspace where we remove each molecule’s overall translation by subtracting the mean of its atomic coordinates, fixing the global translation without changing relative positions.

### Architecture details

#### Inputs and time conditioning

Let $`x_i\in\mathbb R^3`$ be atom $`i`$'s current position, $`a_i`$ its noisy element class, $`f_i`$ its fragment membership, and $`n_f`$ the number of atoms in fragment $`f`$. Time $`t`$ is shared by all atoms in a molecule. The initial hidden feature is

```math
h_i^{(0)}=E_{\mathrm{element}}(a_i)
          +E_{\mathrm{size}}\!\left(\min(n_{f_i},128)\right).
```

The element vocabulary has 13 elements plus a mask class. Atom count and fragment membership are supplied conditioning; size is derived from membership. Clean coordinates, clean element labels, bonds, template coordinates and latent rotation frames are not network inputs. Molecule identifiers and padding masks delimit independent examples. Atom indices and the numerical values of fragment IDs are not embedded.

Time is encoded with 64 logarithmically spaced frequencies $`\omega_k\in[1,1000]`$:

```math
\phi(t)=\big[\sin(2\pi\omega_k t),\cos(2\pi\omega_k t)\big]_{k=1}^{64},
\qquad
 e_t=\mathrm{LN}\!\left(W_2\,\mathrm{SiLU}(W_1\phi(t)+b_1)+b_2\right).
```

The time MLP has dimensions $`128\to d\to d`$; its final LayerNorm has no learned affine parameters. The same time embedding conditions every layer through separate layer-specific projections. Coordinates enter through point attention, not through an absolute-coordinate embedding.

#### Translation-only point attention

The following describes one layer and one head; parameters are independent across layers. Let $`d_h=d/H`$ and suppress the head index where unambiguous. Bias-free linear maps of the layer's normalized, time-modulated features produce scalar queries, keys and values $`q_i,k_i,v_i\in\mathbb R^{d_h}`$ and point offsets $`\delta q_{ip},\delta k_{ip},\delta v_{ir}\in\mathbb R^3`$. Scalar queries and keys are separately RMS-normalized over their head channels, with learned scales. Values are not RMS-normalized.

There are $`P_q=4`$ query/key points and $`P_v=8`$ value points per head. The attention frame is simply $`T_i(z)=x_i+z`$, giving

```math
Q_{ip}=x_i+\delta q_{ip},\qquad
K_{ip}=x_i+\delta k_{ip},\qquad
V_{ir}=x_i+\delta v_{ir}.
```

Let $`\mathcal M(i)`$ contain the valid atoms in atom $`i`$'s molecule. Within this set, the implemented score and attention weights are

```math
\ell_{ij}^{h}
=\frac{(q_i^h)^{\mathsf T}k_j^h}{\sqrt{3d_h}}
-\frac{\alpha_h}{2}\sum_{p=1}^{P_q}\|Q_{ip}^h-K_{jp}^h\|^2
+b^h_{\mathbf{1}[f_i=f_j]},
\qquad
\alpha_h=\mathrm{softplus}(\theta_h)\sqrt{\frac{2}{9P_q}},
\qquad
 A_{ij}^{h}=\frac{\exp(\ell_{ij}^{h})}{\sum_{k\in\mathcal M(i)}\exp(\ell_{ik}^{h})}.
```

Each head has a learned two-entry bias table $`(b_0^h,b_1^h)`$: different fragment and same fragment, respectively. Both entries initially equal zero. Each layer has its own table. This is the explicit pair conditioning; it neither masks out different fragments nor encodes atom ordering. Atoms in different molecules cannot attend to one another, and padding is excluded.

The scalar output and query-relative value-point outputs are

```math
o_i^h=\sum_j A_{ij}^h v_j^h,
\qquad
U_{ir}^h=\sum_j A_{ij}^h V_{jr}^h-x_i.
```

A bias-free output projection combines all scalar outputs, the three Cartesian components of each $`U_{ir}^h`$, and their norms $`\sqrt{\|U_{ir}^h\|^2+10^{-8}}`$ into a $`d`$-dimensional feature. Thus this projection has input width $`H(d_h+4P_v)`$. Unlike the full AlphaFold IPA module, there is no aggregated pair-value output. Scalar Q/K normalization and the fragment-only pair bias are additional choices of this implementation; the equations above specify the actual scaling rather than assuming an unmodified IPA layer.

#### Symmetries

For a common translation $`x_i\mapsto x_i+a`$, all learned point positions translate by $`a`$. Query–key distances are unchanged, and

```math
\sum_j A_{ij}(V_{jr}+a)-(x_i+a)
=\sum_j A_{ij}V_{jr}-x_i,
```

because attention weights sum to one. The input features are translation invariant, so induction through the blocks gives translation-invariant hidden features, logits and predicted displacement/angular/internal vectors. The corresponding unconstrained coordinate update $`x\mapsto x+\Delta_\theta(x)`$ is translation **equivariant**: translating $`x`$ translates the updated coordinates by the same amount. The production sampler additionally chooses the centered representative, as described below. These statements hold in exact arithmetic; floating-point evaluation has rounding error.

For a global rotation $`x_i\mapsto Rx_i`$, the learned point offsets do not automatically become $`R\delta q_{ip}`$, and the component-wise output projection does not enforce a vector transformation law. Therefore this is **not an SE(3)-equivariant network**. Random global rotations augment the training distribution, but do not create an exact architectural guarantee. “Rotationless IPA” means attention has no rotating local frames; Fragment Noise still contains shared fragment rotations in its generative dynamics.

Atom permutations carry coordinates, labels and fragment memberships together. Shared projections, attention and mean pooling then permute atom outputs in the same way, while fragment outputs remain associated with their corresponding atom sets. Arbitrary renaming of fragment IDs leaves atom outputs unchanged and only reindexes fragment outputs. These are permutation equivariance and fragment-renaming invariance, respectively. There is no RoPE, atom-index embedding or sequence-relative pair feature.

#### Transformer block

For each layer $`\ell`$, a SiLU followed by a linear map of $`e_t`$ produces six $`d`$-vectors: shifts $`\beta_A,\beta_F`$, scales $`\gamma_A,\gamma_F`$ and residual gates $`g_A,g_F`$. The two sublayers are

```math
\widetilde h_i=(1+\gamma_A)\odot\mathrm{LN}(h_i)+\beta_A,
\qquad
h_i'=h_i+g_A\odot\mathrm{IPA}(\widetilde h,x,\mathcal F)_i,
```

```math
\widetilde h_i'=(1+\gamma_F)\odot\mathrm{LN}(h_i')+\beta_F,
\qquad
h_i^{\mathrm{next}}=h_i'+g_F\odot\mathrm{SwiGLU}(\widetilde h_i'),
```

```math
\mathrm{SwiGLU}(z)=W_{\mathrm{down}}
\left[(W_{\mathrm{up}}z)\odot\mathrm{SiLU}(W_{\mathrm{gate}}z)\right].
```

The modulation map is zero-initialized, so each block initially implements the identity. LayerNorm inside the blocks has no learned affine parameters; a final LayerNorm after the stack does. SwiGLU uses bias-free maps and hidden width $`4d`$. Attention output and feedforward down-projection weights use Xavier initialization scaled by $`1/L`$, where $`L`$ is depth. Output heads are zero-initialized. There is no dropout. All layers have the same structure and receive the same $`x_t`$; there are no special late coordinate-refinement layers.

#### Output heads and fragment pooling

Let $`h_i`$ now denote the final normalized atom features. Pooling produces

```math
\bar h_f=\frac{1}{n_f}\sum_{i\in I_f}h_i.
```

This is an average of **learned hidden features**, not an average of endpoint coordinates or a clean fragment representation. Those features already incorporate current geometry and interactions with all atoms in the molecule.

| Output | Fragment Noise | Brownian Atoms |
| --- | --- | --- |
| Coordinates | $`\widehat d_f^c=W_c\bar h_f+b_c`$, $`\widehat\omega_f=W_R\bar h_f+b_R`$, $`\widehat d_i^q=W_qh_i+b_q`$ | $`\widehat d_i=W_x[h_i;\bar h_{f_i}]+b_x`$ |
| Endpoint element logits | $`W_a h_i+b_a`$ | Same form |

Fragment Noise uses the structured targets and projection in [Fragment generator targets and loss](mathematics.md#fragment-generator-targets-and-loss). Brownian Atoms predicts $`x_{i,1}-x_{i,t}`$ in Cartesian coordinates with a single head and the [BM loss](mathematics.md#atom-wise-brownian-flow-matching). Pooling lets every atom in the BM readout access a summary of its fragment, while the Fragment Noise heads must produce one centroid/angular prediction per fragment. Both methods also receive membership throughout the trunk, not only at readout.

#### Centering and coordinate updates

All centering is applied independently to each molecule, using equal weight for each atom, including hydrogen. Write

```math
\bar x=\frac1n\sum_i x_i,\qquad
(Px)_i=x_i-\bar x.
```

This is the **center of geometry**, not the physical mass-weighted center of mass. Both models use the centered endpoint $`Px_1`$ and source $`x_0=3P\epsilon`$ Å, with independent standard Gaussian $`\epsilon_i`$ before projection. Thus

```math
\sum_i x_{i,0}=0,\qquad
\mathrm{Cov}(x_{i,0},x_{j,0})=9(\delta_{ij}-1/n)I_3\ \text{Å}^2.
```

Centering removes the molecule's common translational mode, leaving all atom-pair differences unchanged. It is a choice of representative for a molecule's position, rather than an additional learned or physical attraction to the origin. Translation invariance of attention already holds before applying this projection.

For **Brownian Atoms**, conditional coordinates and Cartesian Gaussian noise lie in this centered subspace. Its training target is the centered endpoint difference $`x_1-x_t`$; the raw predicted displacement is compared directly with that target, without projecting the prediction in the loss. Sampling implements

```math
x_{t'}=P\left[x_t+\kappa(t,t')\widehat d_\theta(x_t,a_t,t,\mathcal F)
                   +\sqrt{v(t,t')}\,P\xi\right],
\qquad \xi_i\sim\mathcal N(0,I_3),
```

where $`\kappa`$ and $`v`$ are its bridge pull and innovation-variance coefficients. The outer projection removes any common displacement error made by the network. It is a molecule-level projection, with no fragment rigid-body operation.

For **Fragment Noise**, let $`c_f=n_f^{-1}\sum_{i\in I_f}x_i`$. The centroid prediction and centroid noise use

```math
(P_c z)_f=z_f-\frac{\sum_g n_g z_g}{\sum_g n_g},
\qquad \sum_f n_f(P_c z)_f=0.
```

The weights are atom counts, not element masses: this is exactly the condition for zero global atom centroid when fragments have unequal sizes. Internal coordinates and their noise have zero within-fragment mean. The structured prediction projection also removes the internal prediction's rigid-rotation component using current coordinates, as specified in the [gauge decomposition](mathematics.md#fragment-generator-targets-and-loss). This is distinct from global centering. Each fragment rotates about its current centroid, and the assembled molecule is recentered after every sampling step. The initial fragment centroids are the means of the **same sampled atom coordinates** $`x_0`$, rather than a separate source. No clean endpoint is needed for these inference-time operations.

#### Exact attention factorization and implementation

The squared-distance contribution can be written

```math
-\frac{\alpha_h}{2}\|Q-K\|^2
=\alpha_h Q^{\mathsf T}K
 -\frac{\alpha_h}{2}\|Q\|^2
 -\frac{\alpha_h}{2}\|K\|^2.
```

Consequently, with flattening over query/key points,

```math
\widetilde q_i=
[q_i;\mathrm{vec}(Q_i);(\|Q_{ip}\|^2)_p;\mathbf1_{P_q}],
```

```math
\widetilde k_j=
[k_j/\sqrt{3d_h};\alpha_h\mathrm{vec}(K_j);
 -\tfrac12\alpha_h\mathbf1_{P_q};
 (-\tfrac12\alpha_h\|K_{jp}\|^2)_p],
```

```math
\ell_{ij}^{h}=\widetilde q_i^{\mathsf T}\widetilde k_j
                 +b^h_{\mathbf1[f_i=f_j]}.
```

Scalar values and value points are likewise concatenated before attention. This is the point-distance lifting used by FlashIPA; it requires no approximation to the geometric score. The fragment relation is evaluated directly inside FlexAttention's score modification, so no low-rank approximation to a learned pair tensor is involved. The implementation uses an explicit attention scale of one because all scaling is already in the augmented keys.

Packed molecules provide the block-sparse mask. Within each molecule, all atom pairs are allowed. Fused attention avoids storing quadratic attention/pair tensors, but the arithmetic remains quadratic in each molecule's atom count. During training, a temporary bias buffer indexed by head, query and relation spreads gradient accumulation across queries, then sums back to the same two learned biases per head. This reduces atomic-add contention without changing scores, parameters or gradients. The backbone uses bfloat16 autocast; geometric reductions and structured solves use float32.

| Dataset | Width $`d`$ | Layers $`L`$ | Feedforward width | Heads $`H`$ | Fragment Noise parameters | Brownian Atoms parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GEOM-Drugs | 768 | 16 | 3072 | 8 | 216,378,263 | 216,375,953 |
| QM9 | 512 | 12 | 2048 | 8 | 73,591,607 | 73,590,065 |

The same code implements both sizes. The architecture is in [model.py](../src/mol_lpgm/model.py) and [attention.py](../src/mol_lpgm/attention.py); centering and structured projections are in [process.py](../src/mol_lpgm/process.py), [structured.py](../src/mol_lpgm/structured.py) and [sampler.py](../src/mol_lpgm/sampler.py). Conditional-process projections are in [geometry.jl](../julia/src/geometry.jl).

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

## Architecture references

- John Jumper, Richard Evans, Alexander Pritzel, et al. **Highly accurate protein structure prediction with AlphaFold.** *Nature* 596, 583–589 (2021). [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2). IPA: Supplementary Methods §1.8.2 and Algorithm 22.
- Andrew Liu, Axel Elaldi, Nicholas Franklin, Nathan Russell, Gurinder S. Atwal, Yih-En Ban, and Olivia Viessmann. **Flash Invariant Point Attention.** *Advances in Neural Information Processing Systems* 38 (2025). [Paper and bibliographic record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/07ea874e9e4f71ec6680a3574a485a36-Abstract-Conference.html). Point-distance factorization: §2.4 and Algorithm 1.
- William Peebles and Saining Xie. **Scalable Diffusion Models with Transformers.** *ICCV*, 4195–4205 (2023). [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html). AdaLN-Zero conditioning.
- Noam Shazeer. **GLU Variants Improve Transformer.** (2020). [arXiv:2002.05202](https://arxiv.org/abs/2002.05202). SwiGLU feedforward layers.
