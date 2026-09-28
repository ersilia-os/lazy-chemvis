# LazyChemVis

Automated 2D visualizations of chemical spaces.

LazyChemVis builds a set of reference 2D maps for a chemical library of interest and then
projects new molecules onto those maps. It follows a **fit → transform** logic:

- **fit** learns the reference chemical space once, from a (potentially large) library.
- **transform** places any new set of molecules into that same space, cheaply and
  deterministically, without recomputing the reference embedding.

Four complementary projections are produced, each pairing a molecular representation with a
dimensionality reduction method:

| Projection | Molecular representation | Reduction method |
| --- | --- | --- |
| **PCA** | 15 physicochemical RDKit descriptors | PCA |
| **TMAP** | ECFP4 (Morgan, radius 2, 2048 bits) | TMAP |
| **t-SNE** | [CheMeleon](https://github.com/ersilia-os/eos9o72) embeddings | t-SNE (openTSNE) |
| **UMAP** | [CLAMP](https://github.com/ersilia-os/eos3l5f) embeddings | UMAP |

At fit time, every projection is distilled into a lightweight surrogate, so `transform` does
not need Docker, the Ersilia models or the TMAP environment. The method is described in the
accompanying publication.

## Installation

### 1. Main environment

LazyChemVis installs in two sizes. The base install is enough to project molecules onto an
existing reference space; fitting a new one needs the `fit` extra.

```bash
conda create -n lazychemvis python=3.10
conda activate lazychemvis

# Projecting only (lazychemvis transform)
pip install "lazychemvis @ git+https://github.com/ersilia-os/lazy-chemvis.git" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Fitting and projecting (lazychemvis fit and lazychemvis transform)
pip install "lazychemvis[fit] @ git+https://github.com/ersilia-os/lazy-chemvis.git" \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

`--extra-index-url` installs the CPU-only build of PyTorch.

### 2. TMAP environment (required for `fit` only)

```bash
conda create -n tmap-env -c tmap -c conda-forge python=3.9 "tmap=1.0.6" numpy -y
conda env list | grep tmap-env
# e.g. /home/user/anaconda3/envs/tmap-env
```

`--tmap-env` expects the **path** to this environment directory, not its name. TMAP is
available on Linux and Intel macOS only.

### 3. Ersilia and Docker (required for `fit` only)

The CheMeleon and CLAMP featurizers are served with Ersilia, which requires Docker. Follow
the [Ersilia installation instructions](https://ersilia.gitbook.io/ersilia-book/quick-start/installation)
and make sure Docker is running before fitting.

## Usage

LazyChemVis installs one command, `lazychemvis`, with two subcommands:

| Command | Description |
| --- | --- |
| `lazychemvis fit` | Fit a reference chemical space from a library of molecules |
| `lazychemvis transform` | Project new molecules onto a fitted reference space |

Run `lazychemvis <command> --help` for the options of each.

### Input format

Both commands take `-i/--input`, a **required** path to a CSV file with a header row and
SMILES in the **first column**. Any further columns are ignored.

```csv
smiles
CCOc1ccc2nc(S(N)(=O)=O)sc2c1
CC(=O)Nc1ccc(O)cc1
```

### Fitting a reference chemical space

```bash
lazychemvis fit \
    -i my_reference_library.csv \
    -r my_reference_space \
    --tmap-env /home/user/anaconda3/envs/tmap-env
```

Options:

| Flag | Description |
| --- | --- |
| `-i`, `--input` | **Required.** Path to the CSV of reference SMILES |
| `-r`, `--reference` | **Required.** Directory in which the fitted reference space is written |
| `--tmap-env` | **Required.** Path to the TMAP conda environment directory |
| `--no-cache` | Recompute every featurizer output instead of reusing the ones already in the reference directory |
| `--low-memory` | Lighter TMAP settings for very large libraries (above ~1M molecules) |
| `--no-report` | Skip building the HTML report |
| `-v`, `--verbose` | Show the full log on the console |

When it finishes, `my_reference_space/report/report.html` shows the four reference maps and
a summary of surrogate quality. Re-running a fit into the same reference directory reuses the
featurizer outputs already there, so an interrupted fit resumes where it stopped.

### Projecting new molecules

```bash
lazychemvis transform \
    -i my_new_compounds.csv \
    -r my_reference_space \
    -o my_results
```

Options:

| Flag | Description |
| --- | --- |
| `-i`, `--input` | **Required.** CSV of the molecules to project |
| `-r`, `--reference` | **Required.** Directory of a previously fitted reference space |
| `-o`, `--output` | **Required.** Directory for the output coordinates, figures and report |
| `--no-plots` | Write only the CSV, skipping the figures and the report |
| `--no-report` | Write the figures but skip the HTML report |
| `-v`, `--verbose` | Show the full log on the console |

Outputs written to the output directory:

- `coordinates.csv` — one row per input molecule, in input order, with columns `smiles`,
  `pca_x`, `pca_y`, `tmap_x`, `tmap_y`, `tsne_x`, `tsne_y`, `umap_x`, `umap_y`. Unparseable
  SMILES keep their row; their TMAP coordinates are NaN, so filter on `tmap_x` before using
  the other columns.
- `report/report.html` — the HTML report, with the figures also saved as PDF in
  `report/pdf/`. Keep the `report/` folder together when sharing it.
- `lazychemvis.log` — the full log of the run.

## License

This repository is open-sourced under the GPL-3.0 license. See the [LICENSE](LICENSE) file for
details.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit organization fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for antimicrobial drug discovery.

![Ersilia Logo](assets/Ersilia_Brand.png)
