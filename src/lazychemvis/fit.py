"""
End-to-end pipeline with memory management.

This script includes explicit memory cleanup between pipeline steps to handle
large datasets (>1M molecules) within memory constraints.

Heavy dependencies (umap, openTSNE, ersilia, etc.) are imported lazily inside
each step method so that starting the TMAP pipeline does not trigger the UMAP /
numba import chain (and vice-versa).
"""

import gc
from rich.panel import Panel

from .helpers.logger import get_logger, console, spinner, echo
from .helpers.libraries import load_lib_input
from .helpers.validation import validate_smiles

logger = get_logger(__name__)


class Pipeline(object):
    """
    Complete processing pipeline for computing chemical space projections.
    """

    def __init__(self, lib_input: str, dir_path: str, tmap_env: str,
                 use_cache: bool = False, low_memory: bool = False, verbose: bool = False):
        """
        Initialize the pipeline.

        Parameters
        ----------
        lib_input : str
            Path to a CSV file with a header row and SMILES in the first column.
        dir_path : str
            Directory in which all trained models and outputs will be saved.
        tmap_env : str
            Path to the TMAP conda environment.
        use_cache : bool, default=False
            If True, load precomputed descriptors from disk instead of recomputing.
        low_memory : bool, default=False
            If True, use memory-efficient settings for large datasets (>1M molecules).
        verbose : bool, default=False
            If True, print iteration progress from t-SNE and UMAP to stdout.
        """
        self.lib_input = lib_input
        self.dir_path = dir_path
        self.tmap_env = tmap_env
        self.use_cache = use_cache
        self.low_memory = low_memory
        self.verbose = verbose

    def _pca_step(self, smiles_list):
        """Execute the descriptor → PCA → surrogate → plot sequence."""
        from .featurizers.rdkit_descriptor import RDKitDescriptor
        from .projectors.pca import PCAProjector
        from .surrogates.pca import PCASurrogate
        from .plots.scatter import ScatterPlot

        console.print(Panel.fit("PCA Pipeline", style="bold cyan"))

        def featurize():
            featurizer = RDKitDescriptor(dir_path=self.dir_path)
            featurizer.fit(smiles_list, use_cache=self.use_cache)
            featurizer.save()
            del featurizer
            gc.collect()

        def project():
            pca_proj = PCAProjector(dir_path=self.dir_path)
            pca_proj.fit()
            pca_proj.save()
            del pca_proj
            gc.collect()

        def plot():
            scatter = ScatterPlot(projection_name="pca", dir_path=self.dir_path)
            scatter.plot_reference()

        def train_surrogate():
            pca_surrogate = PCASurrogate(dir_path=self.dir_path)
            pca_surrogate.fit()
            pca_surrogate.save()
            del pca_surrogate
            gc.collect()

        spinner("RDKit Featurization", featurize)
        spinner("PCA Projection", project)
        spinner("Plotting", plot)
        spinner("PCA Surrogate Training", train_surrogate)
        echo("PCA pipeline complete")

    def _tmap_step(self, smiles_list):
        """Execute the ECFP → TMAP → plot sequence with optional low-memory mode."""
        from .featurizers.ecfp import ECFPFeaturizer
        from .projectors.tmap_projector import TMAPProjector
        from .plots.scatter import ScatterPlot
        from .surrogates.tmap import TMAPSurrogate

        console.print(Panel.fit("TMAP Pipeline", style="bold cyan"))

        def featurize():
            featurizer = ECFPFeaturizer(dir_path=self.dir_path)
            featurizer.fit(smiles_list, use_cache=self.use_cache)
            featurizer.save()
            del featurizer
            gc.collect()

        def project():
            tmap_proj = TMAPProjector(
                dir_path=self.dir_path,
                low_memory=self.low_memory,
                n_permutations=64 if self.low_memory else 128,
                batch_size=5000 if self.low_memory else 10000
            )
            tmap_proj.fit(self.tmap_env)

        def plot():
            scatter = ScatterPlot(projection_name="tmap", dir_path=self.dir_path)
            scatter.plot_reference()

        def train_surrogate():
            tmap_surrogate = TMAPSurrogate(dir_path=self.dir_path)
            tmap_surrogate.fit(smiles_list)
            tmap_surrogate.save()

        spinner("ECFP Featurization", featurize)
        spinner("TMAP Projection", project)
        spinner("Plotting", plot)
        spinner("TMAP Surrogate Training", train_surrogate)
        echo("TMAP pipeline complete")

    def _tsne_step(self, smiles_list):
        """Execute the CheMeleon → t-SNE → surrogate → plot sequence with memory management."""
        from .featurizers.chemeleon import CheMeleonFeaturizer
        from .projectors.tsne_projector import TSNEProjector
        from .surrogates.tsne import TSNESurrogate
        from .plots.scatter import ScatterPlot

        console.print(Panel.fit("t-SNE Pipeline", style="bold cyan"))

        def featurize():
            featurizer = CheMeleonFeaturizer(dir_path=self.dir_path)
            featurizer.fit(smiles_list=smiles_list)
            featurizer.save()
            if hasattr(featurizer, 'cleanup'):
                featurizer.cleanup()
            del featurizer
            gc.collect()

        def project():
            tsne_proj = TSNEProjector(dir_path=self.dir_path, verbose=self.verbose)
            tsne_proj.fit()
            tsne_proj.save()
            tsne_proj.cleanup()
            del tsne_proj
            gc.collect()

        def plot():
            scatter = ScatterPlot(projection_name="tsne", dir_path=self.dir_path)
            scatter.plot_reference()

        def train_surrogate():
            tsne_surrogate = TSNESurrogate(dir_path=self.dir_path)
            tsne_surrogate.fit()
            tsne_surrogate.save()
            del tsne_surrogate
            gc.collect()

        spinner("CheMeleon Featurization", featurize)
        spinner("t-SNE Projection", project)
        spinner("Plotting", plot)
        spinner("t-SNE Surrogate Training", train_surrogate)
        echo("t-SNE pipeline complete")

    def _umap_step(self, smiles_list):
        """Execute the CLAMP → UMAP → surrogate → plot sequence with memory management."""
        from .featurizers.clamp import CLAMPFeaturizer
        from .projectors.umap_projector import UMAPProjector
        from .surrogates.umap import UMAPSurrogate
        from .plots.scatter import ScatterPlot

        console.print(Panel.fit("UMAP Pipeline", style="bold cyan"))

        def featurize():
            featurizer = CLAMPFeaturizer(dir_path=self.dir_path)
            featurizer.fit(smiles_list=smiles_list)
            featurizer.save()
            if hasattr(featurizer, 'cleanup'):
                featurizer.cleanup()
            del featurizer
            gc.collect()

        def project():
            umap_proj = UMAPProjector(dir_path=self.dir_path, verbose=self.verbose)
            umap_proj.fit()
            umap_proj.save()
            if hasattr(umap_proj, 'cleanup'):
                umap_proj.cleanup()
            del umap_proj
            gc.collect()

        def plot():
            scatter = ScatterPlot(projection_name="umap", dir_path=self.dir_path)
            scatter.plot_reference()

        def train_surrogate():
            umap_surrogate = UMAPSurrogate(dir_path=self.dir_path)
            umap_surrogate.fit()
            umap_surrogate.save()
            del umap_surrogate
            gc.collect()

        spinner("CLAMP Featurization", featurize)
        spinner("UMAP Projection", project)
        spinner("Plotting", plot)
        spinner("UMAP Surrogate Training", train_surrogate)
        echo("UMAP pipeline complete")

    def run(self):
        """Run the full pipeline with memory management."""
        # Fail fast on a bad TMAP environment: the TMAP step runs after PCA and
        # ECFP featurization, so without this check a mistyped --tmap_env is only
        # discovered hours into a large fit.
        from .projectors.tmap_projector import verify_tmap_env
        verify_tmap_env(self.tmap_env)

        smiles_list = load_lib_input(self.lib_input)

        # Validate once, up front: every featurizer downstream must produce a
        # matrix with exactly these rows, in this order, for the projectors and
        # surrogates to index across them safely.
        smiles_list, _ = validate_smiles(smiles_list)

        n_mols = len(smiles_list)
        console.print(Panel.fit(f"Pipeline Starting — {n_mols:,} molecules", style="bold cyan"))

        if n_mols > 1_000_000:
            logger.warning(
                f"Large dataset detected (>1M molecules). "
                f"Memory cleanup will be performed between steps. "
                f"Consider --low_memory for TMAP if you encounter OOM errors."
            )

        # Run pipeline steps
        self._pca_step(smiles_list)
        self._tmap_step(smiles_list)
        self._tsne_step(smiles_list)
        self._umap_step(smiles_list)

        echo("Full pipeline complete")


def main():
    """Command-line entry point for running the pipeline."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lib_input", type=str, required=True,
                        help="Path to the input library: a CSV file with a header row and "
                             "SMILES in the first column")
    parser.add_argument("--dir_path", type=str, required=True,
                        help="Directory to save trained featurizers and projectors")
    parser.add_argument("--tmap_env", type=str, required=True,
                        help="Path to the TMAP conda environment directory "
                             "(not the environment name); see 'conda env list'")
    parser.add_argument("--use_cache", action='store_true',
                        help='Load precomputed descriptors instead of recomputing them.')
    parser.add_argument("--low_memory", action='store_true',
                        help='Use memory-efficient mode for large datasets (>1M molecules).')
    parser.add_argument("--verbose", action='store_true',
                        help='Print iteration-level progress from t-SNE and UMAP.')
    args = parser.parse_args()

    pipe = Pipeline(
        args.lib_input,
        args.dir_path,
        args.tmap_env,
        args.use_cache,
        args.low_memory,
        args.verbose
    )
    pipe.run()


if __name__ == "__main__":
    main()
