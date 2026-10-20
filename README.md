# molLPGM

Molecular generation with fragment **[latent process generator matching (LPGM)](https://arxiv.org/abs/2605.20547)** and an **atom-wise Brownian flow-matching baseline (BM)**, for GEOM-Drugs and QM9.

The fragment process separates shared fragment translation and rotation from internal atomic motion. BM uses a single Cartesian Brownian bridge. Both networks receive the same fragment membership information, including learned same/different-fragment attention biases in every layer, and use the same backbone and training schedule. Neither uses positional encoding.

## Documentation

- [Mathematics](docs/mathematics.md): latent and observed processes, the pushforward generator, conditional targets, and the fragment and BM objectives.
- [Models and training](docs/models-and-training.md): conditioning, architecture, process parameters, datasets, optimization, evaluation, and commands for training and sampling.
- [Trajectory viewer](viewer/README.md): an independent combined viewer with playback, static frame sequences, and figure export.
- [Publication figures](figures/README.md): standalone plotting and measurement code, with the paired generated and conditional GEOM path data.
- [10K evaluation results](results/README.md): four unfiltered sample archives and ADiT metrics.
- [Comparison protocol](docs/comparison.md): conditioning, method-specific objectives, and evaluation conventions.

## Models

| Dataset | Width | Layers | Parameters per model | Checkpoints |
| --- | ---: | ---: | ---: | --- |
| GEOM-Drugs | 768 | 16 | 216.4M | [Fragment](https://huggingface.co/MurrellLab/LPGM/blob/main/geom/fragment.pt), [BM](https://huggingface.co/MurrellLab/LPGM/blob/main/geom/bm.pt) |
| QM9 | 512 | 12 | 73.6M | [Fragment](https://huggingface.co/MurrellLab/LPGM/blob/main/qm9/fragment.pt), [BM](https://huggingface.co/MurrellLab/LPGM/blob/main/qm9/bm.pt) |

All checkpoints contain final 800K raw and EMA weights. Use `--ema` when sampling from the EMA. Download URLs and checksums are in [checkpoints.json](checkpoints.json).

## Getting started

Training and sampling require Python 3.11+, Julia 1.11+, and a CUDA-enabled PyTorch installation with FlexAttention.

~~~bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[evaluation,test]'
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
~~~

Follow the [data preparation and training commands](docs/models-and-training.md#running-the-models). Python implements the neural models and learned samplers; Julia supplies the conditional training paths. Coordinates are in Å. Evaluation uses the same **ADiT validity, valid-and-connected and uniqueness** protocol for both datasets.
