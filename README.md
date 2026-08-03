# LazyChemVis

Automated 2D visualizations of chemical spaces.

LazyChemVis builds a set of reference 2D maps for a chemical library of interest and then
projects new molecules onto those maps. It follows a **fit → transform** logic:

- **fit** learns the reference chemical space once, from a (potentially large) library.
- **transform** places any new set of molecules into that same space, cheaply and
  deterministically, without recomputing the reference embedding.

Four complementary projections are produced, each pairing a molecular representation with a
dimensionality reduction method:

| Projection | Molecular representation | Reduction method | Captures |
| --- | --- | --- | --- |
| **PCA** | 15 physicochemical RDKit descriptors | PCA | Global, interpretable property gradients |
| **TMAP** | ECFP4 (Morgan, radius 2, 2048 bits) | TMAP (LSH forest + minimum spanning tree layout) | Fine-grained structural neighbourhoods and scaffold branching |
| **t-SNE** | [CheMeleon](https://github.com/ersilia-os/eos9o72) foundation-model embeddings | openTSNE (PCA-50 → FFT-accelerated t-SNE) | Local similarity in a learned chemical representation |
| **UMAP** | [CLAMP](https://github.com/ersilia-os/eos3l5f) bioactivity-aware embeddings | UMAP | Bioactivity-driven structure of the space |

## Why surrogates

Two of the four representations (CheMeleon, CLAMP) are neural models served through the
[Ersilia Model Hub](https://github.com/ersilia-os/ersilia), and two of the four reduction
methods (TMAP, t-SNE) are non-parametric, meaning they cannot natively embed new points.
Requiring either at inference time would make projecting new molecules slow and dependent on
Docker and network access.

LazyChemVis therefore distils every projection into a lightweight **surrogate** at fit time,
so that `transform` needs nothing but RDKit:

- **PCA** — the fitted PCA is re-expressed exactly as a frozen linear PyTorch module. No
  approximation.
- **t-SNE / UMAP** — a multi-output XGBoost regressor learns the mapping from ECFP
  fingerprints directly to the reference coordinates. The expensive embeddings are never
  needed again.
- **TMAP** — reference fingerprints are stored in an [FPSim2](https://github.com/chembl/FPSim2)
  database; a new molecule inherits the coordinates of its nearest reference neighbour by
  Tanimoto similarity.

The t-SNE and UMAP surrogates are trained through an Optuna hyperparameter search followed by
k-fold cross-validation (R², RMSE, MAE and mean Euclidean coordinate error, reported as
mean ± standard deviation across folds), and only then refitted on the full dataset. All
validation metrics and diagnostic plots are written to disk alongside the models.

## Installation

### 1. Main environment

```bash
conda create -n lazychemvis python=3.10
conda activate lazychemvis
pip install git+https://github.com/ersilia-os/lazy-chemvis.git
```

This installs the `lazychemvis_fit` and `lazychemvis_transform` commands.

### 2. TMAP environment (required for `fit` only)

TMAP is distributed as a compiled conda package whose newest release (1.0.6) supports at most
Python 3.9, which is incompatible with the dependency set of the main environment. It
therefore lives in its own environment and is invoked as a subprocess. Create it once:

```bash
conda create -n tmap-env -c tmap -c conda-forge python=3.9 "tmap=1.0.6" numpy -y
```

`numpy` must be requested explicitly: the `tmap` package does not declare it as a dependency,
but the TMAP driver script imports it.

Then find the environment's path, which is what you pass to `--tmap_env`:

```bash
conda env list | grep tmap-env
# e.g. /home/user/anaconda3/envs/tmap-env
```

> `--tmap_env` expects a **path to the environment directory**, not the environment name.
> LazyChemVis calls `<tmap_env>/bin/python3` directly, so a bare name only resolves if it
> happens to be a valid relative path from your working directory.

**Platform support.** The `tmap` conda channel ships builds for `linux-64` and `osx-64` only;
there is no build for Apple Silicon (`osx-arm64`) or Windows. On those platforms the TMAP
projection cannot be fitted, and on `osx-64` the newest `tmap=1.0.6` build requires Python 3.8
rather than 3.9.

TMAP is only needed to *fit* a reference space. `transform` uses the FPSim2 surrogate and
does not require this environment.

### 3. Ersilia and Docker (required for `fit` only)

The CheMeleon and CLAMP featurizers are served with Ersilia, which requires a working Docker
installation. Follow the [Ersilia installation
instructions](https://ersilia.gitbook.io/ersilia-book/quick-start/installation) and make sure
Docker is running before fitting. As above, this is a fit-time requirement only.

## Usage

### Input format

Both commands take `--lib_input`, a **required** path to a CSV file with a header row and
SMILES in the **first column**. Any further columns are ignored.

```csv
smiles
CCOc1ccc2nc(S(N)(=O)=O)sc2c1
CC(=O)Nc1ccc(O)cc1
```

### Fitting a reference chemical space

```bash
lazychemvis_fit \
    --lib_input my_reference_library.csv \
    --dir_path my_reference_space \
    --tmap_env /home/user/anaconda3/envs/tmap-env
```

Options:

| Flag | Description |
| --- | --- |
| `--lib_input` | **Required.** Path to the CSV of reference SMILES |
| `--dir_path` | **Required.** Directory in which all featurizers, projectors and surrogates are written |
| `--tmap_env` | **Required.** Path to the TMAP conda environment directory |
| `--use_cache` | Reuse descriptor matrices already present in `--dir_path` instead of recomputing them |
| `--low_memory` | Reduced-footprint TMAP settings for very large libraries (fewer LSH permutations, smaller batches, lighter layout) |
| `--verbose` | Print iteration-level progress from t-SNE and UMAP |

Fitting runs the four pipelines in sequence (PCA → TMAP → t-SNE → UMAP), releasing memory
between steps. Each pipeline computes its representation, fits the projection, renders a
reference plot, and trains the surrogate.

### Projecting new molecules

```bash
lazychemvis_transform \
    --lib_input my_new_compounds.csv \
    --dir_path my_reference_space \
    --output_path my_results
```

Options:

| Flag | Description |
| --- | --- |
| `--lib_input` | **Required.** CSV of the molecules to project |
| `--dir_path` | **Required.** Directory of a previously fitted reference space |
| `--output_path` | **Required.** Directory for the output coordinates and plots |
| `--no_plots` | Write only the CSV, skipping the overlay figures |

Outputs written to `--output_path`:

- `coordinates.csv` — one row per input molecule, with columns `smiles`, `pca_x`, `pca_y`,
  `tmap_x`, `tmap_y`, `tsne_x`, `tsne_y`, `umap_x`, `umap_y`. All coordinates are scaled to
  the `[-1, 1]` range of the reference space.
- `pca_plot.png`, `tmap_plot.png`, `tsne_plot.png`, `umap_plot.png` — the new molecules
  overlaid on the grey reference landscape (unless `--no_plots` is passed).

### Fitted reference space layout

A fitted `--dir_path` contains one subdirectory per component:

```
my_reference_space/
├── rdkit_descriptor/   # descriptor calculator, imputer, variance filter, scaler, X.npy
├── pca/                # PCA model, axis scaler, reduced.npy, surrogate.pt, reference_space.png
├── ecfp/               # ECFP settings and fingerprint matrix
├── tmap/               # reduced.npy, edges.npz, fps.h5, ref_coords.npy, reference_space.png
├── CheMeleon/          # CheMeleon embeddings
├── tsne/               # t-SNE embedding, PCA-50 model, axis scaler, reduced.npy, reference_space.png
├── tsne_surrogate/     # XGBoost surrogate, metrics and validation_artifacts/
├── CLAMP/              # CLAMP embeddings and validity metadata
├── umap/               # UMAP model, scalers, reduced.npy, reference_space.png
└── umap_surrogate/     # XGBoost surrogate, metrics and validation_artifacts/
```

`reduced.npy` holds the reference coordinates of each projection and is what the plots and
surrogates are built from.

## Memory considerations

Fitting scales to libraries of over a million molecules, but the reference embeddings are the
memory bottleneck. Recommendations:

- Pass `--low_memory` to relax the TMAP layout settings for libraries above ~1M molecules.
- Use `--use_cache` when re-running a pipeline so descriptors are read from disk rather than
  recomputed.
- Featurizers write per-batch files and merge them into a pre-allocated array, so peak memory
  during merging is roughly the final matrix plus one batch.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)

## License

This repository is open-sourced under the GPL-3.0 license. See the [LICENSE](LICENSE) file for
details.
