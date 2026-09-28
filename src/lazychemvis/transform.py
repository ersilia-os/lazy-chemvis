"""
Project new molecules onto a previously fitted reference space.

Every projection is served by a surrogate (an exact linear transform for PCA, an
XGBoost regressor for t-SNE and UMAP, a fingerprint nearest-neighbour lookup for
TMAP), so neither the Ersilia models nor the non-parametric reducers — and hence
neither Docker nor the TMAP environment — are required here.
"""

import gc
import json
import os

import pandas as pd

from .helpers import logger as log_manager
from .helpers.libraries import load_lib_input
from .helpers.logger import get_logger
from .helpers.tracker import PipelineTracker
from .artifacts.pca import PCAArtifact
from .artifacts.tmap import TMAPArtifact
from .artifacts.tsne import TSNEArtifact
from .artifacts.umap import UMAPArtifact
from .featurizers.ecfp import ECFPFeaturizer

logger = get_logger(__name__)

RUN_MANIFEST = "run.json"


class Pipeline(object):
    def __init__(self, lib_input: str, dir_path: str, output_path: str,
                 no_plots: bool = False, no_report: bool = False):
        """
        Parameters
        ----------
        lib_input : str
            CSV of molecules to project (header row, SMILES in the first column).
        dir_path : str
            Directory of a previously fitted reference space.
        output_path : str
            Directory for the coordinates, figures and report.
        no_plots : bool, default=False
            Skip the figures (and therefore the report), writing only the CSV.
        no_report : bool, default=False
            Skip the HTML report but still write the figures.
        """
        self.lib_input = lib_input
        self.dir_path = os.path.abspath(dir_path)
        self.output_path = os.path.abspath(output_path)
        self.no_plots = no_plots
        self.no_report = no_report
        self.tracker = PipelineTracker()

    def _pca_step(self, smiles_list):
        pca_artifact = PCAArtifact(dir_name=self.dir_path)
        X_reduced = pca_artifact.transform(smiles_list)
        del pca_artifact
        return pd.DataFrame(X_reduced, columns=["pca_x", "pca_y"])

    def _tmap_step(self, smiles_list):
        tmap_artifact = TMAPArtifact(dir_name=self.dir_path)
        X_reduced = tmap_artifact.transform(smiles_list)
        del tmap_artifact
        return pd.DataFrame(X_reduced, columns=["tmap_x", "tmap_y"])

    def _tsne_step(self, smiles_list, X_ecfp=None):
        tsne_artifact = TSNEArtifact(dir_name=self.dir_path)
        X_reduced = tsne_artifact.transform(smiles_list, X_ecfp=X_ecfp)
        del tsne_artifact
        return pd.DataFrame(X_reduced, columns=["tsne_x", "tsne_y"])

    def _umap_step(self, smiles_list, X_ecfp=None):
        umap_artifact = UMAPArtifact(dir_name=self.dir_path)
        X_reduced = umap_artifact.transform(smiles_list, X_ecfp=X_ecfp)
        del umap_artifact
        return pd.DataFrame(X_reduced, columns=["umap_x", "umap_y"])

    def _write_manifest(self, n_input):
        """Persist the run configuration for the report."""
        try:
            from importlib.metadata import version
            pkg_version = version("lazychemvis")
        except Exception:
            pkg_version = "unknown"

        manifest = {
            "mode": "transform",
            "lib_input": os.path.abspath(self.lib_input),
            "reference_path": self.dir_path,
            "n_input": n_input,
            "timings": dict(self.tracker.timings),
            "versions": {"lazychemvis": pkg_version},
        }
        with open(os.path.join(self.output_path, RUN_MANIFEST), "w") as f:
            json.dump(manifest, f, indent=2)

    def run(self):
        os.makedirs(self.output_path, exist_ok=True)
        log_path = log_manager.attach_file_sink(self.output_path)
        logger.info(f"Starting transform — log file: {log_path}")

        smiles_list = load_lib_input(self.lib_input)
        n_input = len(smiles_list)

        self.tracker.begin(
            "LazyChemVis — projecting molecules",
            f"{n_input:,} molecules onto {os.path.basename(self.dir_path)}",
        )

        # Molecules are NOT filtered here: the output must stay aligned row-for-row
        # with the input CSV, so unparseable molecules keep their row. TMAP gives them
        # NaN coordinates; PCA (imputed descriptors) and the t-SNE/UMAP surrogates
        # (all-zero fingerprint) give them a placeholder coordinate instead.
        df_combined = pd.DataFrame({'smiles': smiles_list})

        self.tracker.start("pca", "RDKit descriptors → frozen PCA")
        df_pca = self._pca_step(smiles_list)
        df_combined = pd.concat([df_combined, df_pca], axis=1)
        del df_pca
        gc.collect()
        self.tracker.complete("pca")

        self.tracker.start("tmap", "ECFP → nearest reference neighbour")
        df_tmap = self._tmap_step(smiles_list)
        df_combined = pd.concat([df_combined, df_tmap], axis=1)
        del df_tmap
        gc.collect()
        self.tracker.complete("tmap")

        # Compute ECFP once and share it between the two XGBoost surrogates.
        ecfp_featurizer = ECFPFeaturizer.load(dir_path=self.dir_path, load_X=False)
        X_ecfp = ecfp_featurizer.transform(smiles_list)
        del ecfp_featurizer

        self.tracker.start("tsne", "ECFP → t-SNE surrogate")
        df_tsne = self._tsne_step(smiles_list, X_ecfp=X_ecfp)
        df_combined = pd.concat([df_combined, df_tsne], axis=1)
        del df_tsne
        gc.collect()
        self.tracker.complete("tsne")

        self.tracker.start("umap", "ECFP → UMAP surrogate")
        df_umap = self._umap_step(smiles_list, X_ecfp=X_ecfp)
        df_combined = pd.concat([df_combined, df_umap], axis=1)
        del df_umap, X_ecfp
        gc.collect()
        self.tracker.complete("umap")

        # The CSV is written before the figures, so a failure while plotting cannot
        # cost the coordinates — which are the actual deliverable.
        output_df = os.path.join(self.output_path, "coordinates.csv")
        df_combined.to_csv(output_df, index=False)
        logger.info(f"Wrote coordinates: {output_df}")
        del df_combined
        gc.collect()

        self._write_manifest(n_input)

        extra = [("coordinates", output_df)]
        if not self.no_plots:
            from .report.report import TransformReporter

            self.tracker.start("report", "figures and HTML report")
            reporter = TransformReporter(
                path=self.output_path, reference_path=self.dir_path,
                make_plots=True,
            )
            report_path = reporter.run() if not self.no_report else None
            self.tracker.complete("report")
            if report_path:
                extra.append(("report", report_path))
        if log_path:
            extra.append(("log", log_path))

        self.tracker.finish(extra_rows=extra)
        logger.info("Transform pipeline complete.")
