# Interpreting the Latent Space of I-JEPA

### Through SVD Analysis and Classical Machine Learning Probes

A framework for comparing the geometric structure of self-supervised visual
representations. It asks one central question:

> **Does I-JEPA learn a more semantically structured latent manifold than
> supervised CNNs and contrastive learning?**

Four frozen encoders are compared as feature extractors using **SVD spectral
analysis** (effective rank, dimensional collapse, intrinsic dimension) and a
**linear SVM probe** (margin, support vectors, class confusion). The key
contribution is *connecting* the two: testing whether spectral richness is
associated with linear separability, with the MAE/I-JEPA pair providing an
architecture control.

---

## The encoders

| Name (registry key)     | Backbone     | Paradigm        | Role |
|-------------------------|--------------|-----------------|------|
| `resnet18_supervised`   | ResNet-18    | Supervised      | CNN baseline |
| `simclr_resnet18`       | ResNet-18    | Contrastive     | Contrastive baseline |
| `ijepa_vits16`          | ViT-S/16     | Predictive      | **Focus of the study** |
| `mae_vits16`            | ViT-S/16     | Reconstruction  | Architecture control vs I-JEPA |

I-JEPA and MAE share an **identical ViT-S/16 backbone** and differ only in their
self-supervised objective. This pairing isolates the effect of the *predictive
objective* from the effect of the *Transformer architecture* — the cleanest test
in the project.

---

## Methodology → code map

The pipeline mirrors the proposal section by section.

| Proposal section | What it does | Code |
|------------------|--------------|------|
| 2.5 Implementation | Uniform preprocessing: random projection → 384-d (`N(0,1/384)`, seed 42), L₂ normalize, mean-center (in SVD) | [src/features/preprocessing.py](src/features/preprocessing.py) |
| **Part A** — SVD | Singular spectrum, **RankMe** effective rank, PCA participation ratio, dimensional collapse | [src/analysis/svd.py](src/analysis/svd.py) |
| **Part A** — Two-NN | Non-linear intrinsic dimension; gap vs effective rank flags manifold curvature | [src/analysis/intrinsic_dim.py](src/analysis/intrinsic_dim.py) |
| **Part B** — SVM | Linear SVM probe: margin `1/‖w‖`, support-vector count, class confusion | [src/analysis/svm_probe.py](src/analysis/svm_probe.py) |
| Connecting SVD↔SVM | Rank vs margin association; architecture-controlled MAE/I-JEPA delta | [src/analysis/connect.py](src/analysis/connect.py) |
| **Part C** (optional) | ModelNet40 multi-view viewpoint consistency + t-SNE | [src/analysis/viewpoint.py](src/analysis/viewpoint.py) |
| **Part D** (optional) | Latent stability under Gaussian noise / crop / blur | [src/analysis/perturbation.py](src/analysis/perturbation.py) |

---

## Project layout

```
.
├── configs/
│   ├── default.yaml          # experiment config (override on CLI)
│   └── encoders.yaml         # encoder specs + checkpoint paths
├── src/
│   ├── data/                 # STL-10 / CIFAR-10, transforms, perturbations, ModelNet40
│   ├── encoders/             # frozen feature extractors (ResNet, ViT) + registry
│   ├── features/             # extraction, preprocessing, caching
│   ├── analysis/             # svd, intrinsic_dim, svm_probe, connect, viewpoint, perturbation
│   ├── viz/                  # spectra / scatter / confusion / t-SNE / sensitivity plots
│   └── utils/                # config, seeding, logging
├── scripts/                  # numbered pipeline stages + run_all.sh
├── tests/                    # unit tests for the core math (no weights/data needed)
├── weights/                  # pretrained checkpoints (gitignored)
├── features/cache/           # cached raw features (gitignored)
└── results/                  # JSON summaries, CSVs, figures (gitignored)
```

---

## Setup

This project uses **conda**.

```bash
# Create and activate an environment
conda create -n ijepa python=3.10 -y
conda activate ijepa

# Install dependencies (PyTorch, scikit-learn, timm, etc.)
pip install -r requirements.txt
# (or, for development) pip install -e ".[dev]"
```

> The unit tests run with just `torch`, `scikit-learn`, `scipy`, `numpy`.
> `timm` is only needed to actually build the ViT encoders (I-JEPA / MAE).

### Pretrained weights

The supervised ResNet-18 downloads automatically via torchvision. The other
three need manual download — drop them at the paths in
[configs/encoders.yaml](configs/encoders.yaml) (default `./weights/`):

| Encoder | Source |
|---------|--------|
| I-JEPA ViT-S/16  | https://github.com/facebookresearch/ijepa |
| MAE ViT-S/16     | https://github.com/facebookresearch/mae |
| SimCLR ResNet-18 | https://github.com/google-research/simclr (convert) or a PyTorch port |

Verify everything loads:

```bash
python scripts/00_download_weights.py
```

---

## Running the pipeline

End-to-end (STL-10 by default):

```bash
bash scripts/run_all.sh
```

Or stage by stage:

```bash
python scripts/01_extract_features.py     # extract + cache raw features
python scripts/02_svd_analysis.py         # Part A: SVD / RankMe / Two-NN
python scripts/03_svm_probe.py            # Part B: linear SVM probe
python scripts/04_connect_analysis.py     # connect spectral rank ↔ SVM margin
```

Optional extensions:

```bash
python scripts/05_viewpoint.py    viewpoint.enabled=true viewpoint.modelnet_root=/path/to/modelnet40v2png
python scripts/06_perturbation.py perturbation.enabled=true
```

### Configuration & overrides

Every script reads `configs/default.yaml` and accepts `key.sub=value` overrides
on the command line:

```bash
# Cross-validate on CIFAR-10 with a larger SVM C, on CPU
python scripts/03_svm_probe.py dataset.name=cifar10 svm.C=10 device=cpu

# Use a different config file entirely
python scripts/02_svd_analysis.py --config configs/default.yaml features.batch_size=128
```

Key knobs (see [configs/default.yaml](configs/default.yaml) for all):

- `dataset.name` — `stl10` (primary) or `cifar10` (secondary).
- `features.random_projection` — common dim 384, seed 42 (fixed per proposal).
- `svd.collapse_threshold` — explained-variance fraction below which a dim is "collapsed".
- `svm.C`, `svm.max_samples_fit` — SVM regularization and fit-set cap.

---

## Outputs

Results land under `results/<dataset>/`:

| File | Contents |
|------|----------|
| `svd/svd_summary.json`        | RankMe, PCA effective rank, Two-NN dim, collapse counts per encoder |
| `svd/singular_spectra.png`    | Overlaid singular-value spectra |
| `svm/svm_summary.json`        | Accuracy, mean margin, support-vector counts, hardest class pairs |
| `svm/confusion_<encoder>.png` | Per-encoder confusion matrix |
| `connect/rank_vs_margin.csv`  | Combined rank↔margin table |
| `connect/connect_report.json` | Correlations + architecture-controlled MAE/I-JEPA delta |
| `connect/rank_vs_margin.png`  | Effective-rank vs margin scatter |

---

## Hypotheses under test

- **Part A:** I-JEPA shows slower spectral decay (higher effective rank) than the
  supervised CNN, with a *different* decay profile than SimCLR. A large Two-NN vs
  effective-rank gap indicates a curved (non-linear) manifold.
- **Part B:** Larger margin / fewer support vectors ⇒ more linearly organized
  class structure.
- **Connection:** Higher effective rank *may* afford a linear classifier more
  room for a wide margin — but a low-rank manifold can still be highly separable.
  This is reported as a **structural correlation, not a causal claim**.
- **Part C:** I-JEPA achieves a lower viewpoint-consistency score (more
  invariant to viewpoint).
- **Part D:** I-JEPA representations shift less under input perturbation than CNN
  features.

> **Interpretive note (from the proposal):** effective rank is a *descriptor of
> geometry, not a quality metric*. Higher rank does not by itself mean a better
> representation. Findings on STL-10/CIFAR-10 are indicative, not definitive at
> ImageNet scale.

---

## Testing

The core math is unit-tested and runs without any weights or datasets:

```bash
pytest -q
```

Covered: random-projection determinism & scale, L₂/mean-center invariants,
RankMe behaviour (full-rank vs low-rank, monotonicity), Two-NN recovery of a
known subspace dimension, and SVM separability/margin/support-vector diagnostics.

---

## References

- Assran et al. (2023). *Self-supervised learning from images with a
  joint-embedding predictive architecture (I-JEPA).* CVPR.
- Garrido et al. (2023). *RankMe: Assessing the downstream performance of
  pretrained self-supervised representations by their rank.* ICML.
- Jing et al. (2022). *Understanding dimensional collapse in contrastive
  self-supervised learning.* ICLR.
- Chen et al. (2020). *A simple framework for contrastive learning of visual
  representations (SimCLR).* ICML.
- He et al. (2022). *Masked autoencoders are scalable vision learners (MAE).* CVPR.
- Facco et al. (2017). *Estimating the intrinsic dimension of datasets by a
  minimal neighborhood information (Two-NN).* Scientific Reports.
- Coates et al. (2011). *An analysis of single-layer networks in unsupervised
  feature learning (STL-10).* AISTATS.
```
