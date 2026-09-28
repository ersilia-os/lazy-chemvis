"""
End-to-end pipeline with memory management.

This script includes explicit memory cleanup between pipeline steps to handle
large datasets (>1M molecules) within memory constraints.

Heavy dependencies (umap, openTSNE, ersilia, etc.) are imported lazily inside
each step method so that starting the TMAP pipeline does not trigger the UMAP /
numba import chain (and vice-versa).
"""

import gc
import importlib.util
import json
import os

from .helpers import logger as log_manager
from .helpers.logger import get_logger
from .helpers.libraries import load_lib_input
from .helpers.tracker import PipelineTracker
from .helpers.validation import validate_smiles

logger = get_logger(__name__)

# Written next to the artifacts so the report can describe the run that produced them.
RUN_MANIFEST = "run.json"

# Fit-only dependencies, installed with the ``[fit]`` extra: module name → distribution.
FIT_DEPENDENCIES = {
    "ersilia": "ersilia",
    "umap": "umap-learn",
    "openTSNE": "opentsne",
    "optuna": "optuna",
}

FIT_INSTALL_HINT = (
    'pip install "lazychemvis[fit] @ git+https://github.com/ersilia-os/lazy-chemvis.git" '
    "--extra-index-url https://download.pytorch.org/whl/cpu"
)


def check_fit_dependencies():
    """
    Fail fast when the fit-only dependencies are not installed.

    The base install is enough for ``lazychemvis transform``; fitting also needs
    the packages of the ``[fit]`` extra. They are imported lazily, step by step, so
    without this check a transform-only install would fail with a bare
    ImportError partway through a fit.

    Raises
    ------
    ImportError
        Listing every missing distribution and the command that installs them.
    """
    missing = [
        dist for module, dist in FIT_DEPENDENCIES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise ImportError(
            f"Fitting needs packages that are not installed: {', '.join(missing)}.\n"
            f"This looks like a transform-only install. Install the fit extra with:\n"
            f"    {FIT_INSTALL_HINT}"
        )


class Pipeline(object):
    """
    Complete processing pipeline for computing chemical space projections.
    """

    def __init__(self, lib_input: str, dir_path: str, tmap_env: str,
                 no_cache: bool = False, low_memory: bool = False,
                 verbose: bool = False, no_report: bool = False):
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
        no_cache : bool, default=False
            If True, recompute every featurizer output. By default, outputs already
            in ``dir_path`` are reused when they were computed from the same library
            with the same settings, and discarded otherwise.
        low_memory : bool, default=False
            If True, use memory-efficient settings for large datasets (>1M molecules).
        verbose : bool, default=False
            If True, show the full DEBUG log stream on the console.
        no_report : bool, default=False
            If True, skip building the HTML report.
        """
        self.lib_input = lib_input
        self.dir_path = dir_path
        self.tmap_env = tmap_env
        self.no_cache = no_cache
        self.low_memory = low_memory
        self.verbose = verbose
        self.no_report = no_report
        self.tracker = PipelineTracker()
        self.manifest = {}
        self.cache_reused = {}

    def _record_reuse(self, featurizer):
        """Note whether a featurizer reused its cached output, and say so on the console."""
        reused = bool(getattr(featurizer, "_from_cache", False))
        self.cache_reused[featurizer.featurizer_name] = reused
        if reused:
            self.tracker.substep(f"{featurizer.featurizer_name}: reused cached output")

    def _pca_step(self, smiles_list):
        """Execute the descriptor → PCA → surrogate → plot sequence."""
        from .featurizers.rdkit_descriptor import RDKitDescriptor
        from .projectors.pca import PCAProjector
        from .surrogates.pca import PCASurrogate
        from .report.plots import ReferenceLandscapePlot

        self.tracker.start("pca", "RDKit descriptors → PCA")

        self.tracker.substep("RDKit featurization")
        featurizer = RDKitDescriptor(dir_path=self.dir_path)
        featurizer.fit(smiles_list, reuse=not self.no_cache)
        featurizer.save()
        self._record_reuse(featurizer)
        del featurizer
        gc.collect()

        self.tracker.substep("PCA projection")
        pca_proj = PCAProjector(dir_path=self.dir_path)
        pca_proj.fit()
        pca_proj.save()
        explained = float(sum(pca_proj.reducer.explained_variance_ratio_))
        self.manifest.setdefault("pca", {})["explained_variance"] = explained
        del pca_proj
        gc.collect()

        self.tracker.substep("Plotting reference landscape")
        ReferenceLandscapePlot(projection_name="pca", path=self.dir_path).save()

        self.tracker.substep("PCA surrogate")
        pca_surrogate = PCASurrogate(dir_path=self.dir_path)
        pca_surrogate.fit()
        pca_surrogate.save()
        del pca_surrogate
        gc.collect()

        self.tracker.complete("pca", rows=[("explained variance", f"{explained:.1%}")])

    def _tmap_step(self, smiles_list):
        """Execute the ECFP → TMAP → plot sequence with optional low-memory mode."""
        from .featurizers.ecfp import ECFPFeaturizer
        from .projectors.tmap_projector import TMAPProjector
        from .report.plots import ReferenceLandscapePlot
        from .surrogates.tmap import TMAPSurrogate

        self.tracker.start("tmap", "ECFP → TMAP")

        self.tracker.substep("ECFP featurization")
        featurizer = ECFPFeaturizer(dir_path=self.dir_path)
        featurizer.fit(smiles_list, reuse=not self.no_cache)
        featurizer.save()
        self._record_reuse(featurizer)
        self.manifest.setdefault("tmap", {}).update(
            {"radius": featurizer.radius, "n_bits": featurizer.n_bits}
        )
        del featurizer
        gc.collect()

        self.tracker.substep("TMAP projection")
        tmap_proj = TMAPProjector(
            dir_path=self.dir_path,
            low_memory=self.low_memory,
            n_permutations=64 if self.low_memory else 128,
            batch_size=5000 if self.low_memory else 10000
        )
        tmap_proj.fit(self.tmap_env)
        self.manifest["tmap"].update({
            "k": tmap_proj.k, "kc": tmap_proj.kc,
            "n_permutations": tmap_proj.n_permutations,
            "low_memory": self.low_memory,
        })

        self.tracker.substep("Plotting reference landscape")
        ReferenceLandscapePlot(projection_name="tmap", path=self.dir_path).save()

        self.tracker.substep("TMAP surrogate (FPSim2 database)")
        tmap_surrogate = TMAPSurrogate(dir_path=self.dir_path)
        tmap_surrogate.fit(smiles_list)
        tmap_surrogate.save()

        self.tracker.complete("tmap")

    def _tsne_step(self, smiles_list):
        """Execute the CheMeleon → t-SNE → surrogate → plot sequence with memory management."""
        from .featurizers.chemeleon import CheMeleonFeaturizer
        # Ersilia resets loguru's handlers at import time; re-assert ours or the rest
        # of the run would be silent and nothing more would reach the log file.
        log_manager.configure()
        from .projectors.tsne_projector import TSNEProjector
        from .surrogates.tsne import TSNESurrogate
        from .report.plots import ReferenceLandscapePlot

        self.tracker.start("tsne", "CheMeleon → t-SNE")

        self.tracker.substep("CheMeleon featurization")
        featurizer = CheMeleonFeaturizer(dir_path=self.dir_path)
        featurizer.fit(smiles_list=smiles_list, reuse=not self.no_cache)
        featurizer.save()
        self._record_reuse(featurizer)
        self.manifest.setdefault("tsne", {})["model_id"] = featurizer._model_id
        if hasattr(featurizer, 'cleanup'):
            featurizer.cleanup()
        del featurizer
        gc.collect()
        log_manager.configure()

        self.tracker.substep("t-SNE projection")
        tsne_proj = TSNEProjector(dir_path=self.dir_path, verbose=self.verbose)
        tsne_proj.fit()
        self.manifest["tsne"]["perplexity"] = tsne_proj.perplexity
        tsne_proj.cleanup()
        del tsne_proj
        gc.collect()

        self.tracker.substep("Plotting reference landscape")
        ReferenceLandscapePlot(projection_name="tsne", path=self.dir_path).save()

        self.tracker.substep("t-SNE surrogate (Optuna → CV → production)")
        tsne_surrogate = TSNESurrogate(dir_path=self.dir_path)
        tsne_surrogate.fit()
        rows = self._metric_rows(tsne_surrogate.metrics)
        del tsne_surrogate
        gc.collect()

        self.tracker.complete("tsne", rows=rows)

    def _umap_step(self, smiles_list):
        """Execute the CLAMP → UMAP → surrogate → plot sequence with memory management."""
        from .featurizers.clamp import CLAMPFeaturizer
        # See the note in _tsne_step: Ersilia wipes loguru's handlers on import.
        log_manager.configure()
        from .projectors.umap_projector import UMAPProjector
        from .surrogates.umap import UMAPSurrogate
        from .report.plots import ReferenceLandscapePlot

        self.tracker.start("umap", "CLAMP → UMAP")

        self.tracker.substep("CLAMP featurization")
        featurizer = CLAMPFeaturizer(dir_path=self.dir_path)
        featurizer.fit(smiles_list=smiles_list, reuse=not self.no_cache)
        featurizer.save()
        self._record_reuse(featurizer)
        self.manifest.setdefault("umap", {})["model_id"] = featurizer._model_id
        if hasattr(featurizer, 'cleanup'):
            featurizer.cleanup()
        del featurizer
        gc.collect()
        log_manager.configure()

        self.tracker.substep("UMAP projection")
        umap_proj = UMAPProjector(dir_path=self.dir_path, verbose=self.verbose)
        umap_proj.fit()
        self.manifest["umap"].update({
            "n_neighbors": umap_proj.n_neighbors,
            "min_dist": umap_proj.min_dist,
            "metric": umap_proj.metric,
        })
        if hasattr(umap_proj, 'cleanup'):
            umap_proj.cleanup()
        del umap_proj
        gc.collect()

        self.tracker.substep("Plotting reference landscape")
        ReferenceLandscapePlot(projection_name="umap", path=self.dir_path).save()

        self.tracker.substep("UMAP surrogate (Optuna → CV → production)")
        umap_surrogate = UMAPSurrogate(dir_path=self.dir_path)
        umap_surrogate.fit()
        rows = self._metric_rows(umap_surrogate.metrics)
        del umap_surrogate
        gc.collect()

        self.tracker.complete("umap", rows=rows)

    @staticmethod
    def _metric_rows(metrics):
        """Format a surrogate's CV metrics as tracker detail rows."""
        if not metrics:
            return None
        return [
            ("R²", f"{metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}"),
            ("Euclidean error", f"{metrics['euclidean_mean']:.4f} ± "
                               f"{metrics['euclidean_std']:.4f}"),
        ]

    def _write_manifest(self, n_input, n_valid):
        """Persist the run configuration for the report."""
        from rdkit import Chem

        try:
            from importlib.metadata import version
            pkg_version = version("lazychemvis")
        except Exception:
            pkg_version = "unknown"

        self.manifest.update({
            "mode": "fit",
            "lib_input": os.path.abspath(self.lib_input),
            "n_input": n_input,
            "n_valid": n_valid,
            "n_invalid": n_input - n_valid,
            "low_memory": self.low_memory,
            "no_cache": self.no_cache,
            "cache_reused": dict(self.cache_reused),
            "timings": dict(self.tracker.timings),
            "versions": {
                "lazychemvis": pkg_version,
                "rdkit": Chem.rdBase.rdkitVersion,
            },
        })
        path = os.path.join(self.dir_path, RUN_MANIFEST)
        with open(path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        logger.debug(f"Saved run manifest: {path}")

    def run(self):
        """Run the full pipeline with memory management."""
        # Attach the log file before anything can fail, so even an early error is recorded.
        log_path = log_manager.attach_file_sink(self.dir_path)
        logger.info(f"Starting fit — log file: {log_path}")

        check_fit_dependencies()

        # Fail fast on a bad TMAP environment: the TMAP step runs after PCA and
        # ECFP featurization, so without this check a mistyped --tmap-env is only
        # discovered hours into a large fit.
        from .projectors.tmap_projector import verify_tmap_env
        verify_tmap_env(self.tmap_env)

        smiles_list = load_lib_input(self.lib_input)
        n_input = len(smiles_list)

        # Validate once, up front: every featurizer downstream must produce a
        # matrix with exactly these rows, in this order, for the projectors and
        # surrogates to index across them safely.
        smiles_list, _ = validate_smiles(smiles_list)
        n_mols = len(smiles_list)

        self.tracker.begin(
            "LazyChemVis — fitting reference space",
            f"{n_mols:,} molecules"
            + (f" ({n_input - n_mols:,} invalid dropped)" if n_input != n_mols else ""),
        )

        if n_mols > 1_000_000:
            logger.warning(
                "Large dataset detected (>1M molecules). Memory cleanup will be "
                "performed between steps. Consider --low-memory for TMAP if you "
                "encounter OOM errors."
            )

        # Run pipeline steps
        self._pca_step(smiles_list)
        self._tmap_step(smiles_list)
        self._tsne_step(smiles_list)
        self._umap_step(smiles_list)

        self._write_manifest(n_input, n_mols)

        extra = []
        if not self.no_report:
            from .report.report import FitReporter

            self.tracker.start("report", "HTML report")
            report_path = FitReporter(path=self.dir_path).run()
            self.tracker.complete("report")
            extra.append(("report", report_path))
        if log_path:
            extra.append(("log", log_path))

        self.tracker.finish(extra_rows=extra)
        logger.info("Fit pipeline complete.")
