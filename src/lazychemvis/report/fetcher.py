"""
Read a finished run.

One place that knows where a run's artifacts live, so no figure or HTML fragment
touches paths directly. Everything is best-effort: a report must be buildable from
a partial run (e.g. ``--no-report`` was used earlier, or a step was skipped), so a
missing artifact yields None rather than raising.

Reference coordinates are cached, because several figures shade the same
projection and re-reading a million-row array four times is pure waste.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from ..helpers.logger import get_logger

logger = get_logger(__name__)

PROJECTIONS = ("pca", "tmap", "tsne", "umap")

#: Which surrogate directory holds the CV artifacts for a projection. PCA and TMAP
#: have no cross-validated surrogate by construction, so they are absent here.
SURROGATE_DIRS = {"tsne": "tsne_surrogate", "umap": "umap_surrogate"}

PROJECTION_LABELS = {
    "pca": "PCA",
    "tmap": "TMAP",
    "tsne": "t-SNE",
    "umap": "UMAP",
}

PROJECTION_DESCRIPTIONS = {
    "pca": "RDKit physicochemical descriptors → PCA",
    "tmap": "ECFP4 fingerprints → TMAP (LSH forest + MST layout)",
    "tsne": "CheMeleon embeddings → PCA-50 → FFT t-SNE",
    "umap": "CLAMP bioactivity embeddings → UMAP",
}


class ResultsFetcher(object):
    """
    Accessor for the artifacts of a fitted reference space or a transform run.

    Parameters
    ----------
    path : str
        A run's output directory (``--reference`` for fit, ``--output`` for
        transform).
    reference_path : str, optional
        For a transform run, the fitted reference space directory, which is where the
        reference coordinates live. Defaults to ``path``.
    """

    def __init__(self, path, reference_path=None):
        self.path = os.path.abspath(path)
        self.reference_path = os.path.abspath(reference_path or path)
        self._coords_cache = {}

    # ------------------------------------------------------------------
    # Run manifest
    # ------------------------------------------------------------------

    def manifest(self):
        """Return the run manifest dict written by the pipeline, or {}."""
        for base in (self.path, self.reference_path):
            path = os.path.join(base, "run.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        return json.load(f)
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Could not read run manifest {path}: {e}")
        return {}

    # ------------------------------------------------------------------
    # Reference coordinates
    # ------------------------------------------------------------------

    def reference_coords(self, projection):
        """
        Return the reference coordinates for a projection as an (n, 2) array, or None.

        Cached: several figures shade the same landscape.
        """
        if projection in self._coords_cache:
            return self._coords_cache[projection]
        path = os.path.join(self.reference_path, projection, "reduced.npy")
        coords = None
        if os.path.exists(path):
            try:
                coords = np.load(path).astype(np.float32)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Could not read {path}: {e}")
        else:
            logger.debug(f"No reference coordinates for {projection} at {path}")
        self._coords_cache[projection] = coords
        return coords

    def has_projection(self, projection):
        """True if the reference coordinates for a projection exist."""
        return self.reference_coords(projection) is not None

    def available_projections(self):
        """Projections whose reference coordinates are present, in canonical order."""
        return [p for p in PROJECTIONS if self.has_projection(p)]

    # ------------------------------------------------------------------
    # Surrogate metrics
    # ------------------------------------------------------------------

    def metrics(self, projection):
        """
        Return a surrogate's cross-validation metrics dict, or None.

        Prefers ``metrics.json``; falls back to the joblib file written by older runs.
        """
        sub = SURROGATE_DIRS.get(projection)
        if sub is None:
            return None
        base = os.path.join(self.reference_path, sub)

        json_path = os.path.join(base, "metrics.json")
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    return json.load(f)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Could not read {json_path}: {e}")

        joblib_path = os.path.join(base, "metrics.joblib")
        if os.path.exists(joblib_path):
            try:
                import joblib

                return joblib.load(joblib_path)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Could not read {joblib_path}: {e}")
        return None

    def fold_predictions(self, projection):
        """
        Return the concatenated per-fold prediction frames for a surrogate, or None.

        Columns: true_x, true_y, pred_x, pred_y, euclidean_error, fold.
        """
        sub = SURROGATE_DIRS.get(projection)
        if sub is None:
            return None
        pattern = os.path.join(
            self.reference_path, sub, "validation_artifacts", "fold_*_predictions.csv"
        )
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        frames = []
        for f in files:
            try:
                df = pd.read_csv(f)
                stem = os.path.basename(f)
                df["fold"] = int(stem.split("_")[1])
                frames.append(df)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Could not read {f}: {e}")
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def validation_figures(self, projection):
        """
        Return the per-fold validation figure stems already on disk for a surrogate.

        These are produced during surrogate training, not by the report, so the report
        copies them in rather than redrawing them.
        """
        sub = SURROGATE_DIRS.get(projection)
        if sub is None:
            return []
        val_dir = os.path.join(self.reference_path, sub, "validation_artifacts")
        if not os.path.isdir(val_dir):
            return []
        return sorted(
            os.path.join(val_dir, f)
            for f in os.listdir(val_dir)
            if f.endswith(".png")
        )

    # ------------------------------------------------------------------
    # Featurizer metadata
    # ------------------------------------------------------------------

    def featurizer_meta(self, name):
        """Return a featurizer's ``featurizer.json`` contents, or {}."""
        path = os.path.join(self.reference_path, name, "featurizer.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:  # pragma: no cover
            return {}

    def n_reference_molecules(self):
        """Number of molecules in the reference space, or None."""
        for projection in PROJECTIONS:
            coords = self.reference_coords(projection)
            if coords is not None:
                return int(coords.shape[0])
        return None

    def tmap_edge_count(self):
        """Number of edges in the TMAP spanning tree, or None."""
        path = os.path.join(self.reference_path, "tmap", "edges.npz")
        if not os.path.exists(path):
            return None
        try:
            with np.load(path) as data:
                return int(len(data["s"]))
        except Exception:  # pragma: no cover
            return None

    def timings(self):
        """Per-step wall-clock times from the manifest, or {}."""
        return self.manifest().get("timings", {})

    # ------------------------------------------------------------------
    # Transform outputs
    # ------------------------------------------------------------------

    def coordinates(self):
        """Return the transform run's ``coordinates.csv`` as a DataFrame, or None."""
        path = os.path.join(self.path, "coordinates.csv")
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not read {path}: {e}")
            return None

    def projected_coords(self, projection):
        """Return the (n, 2) projected coordinates for a projection, or None."""
        df = self.coordinates()
        if df is None:
            return None
        cols = [f"{projection}_x", f"{projection}_y"]
        if not all(c in df.columns for c in cols):
            return None
        return df[cols].to_numpy(dtype=np.float32)

    def unplaceable_counts(self):
        """
        Return ``{projection: n_unplaced}`` for a transform run.

        A molecule is unplaced when its coordinates are NaN. Only the TMAP artifact
        produces NaN (for SMILES RDKit cannot parse); PCA, t-SNE and UMAP return a
        placeholder coordinate instead, so their counts are always zero.
        """
        df = self.coordinates()
        if df is None:
            return {}
        counts = {}
        for projection in PROJECTIONS:
            cols = [f"{projection}_x", f"{projection}_y"]
            if all(c in df.columns for c in cols):
                counts[projection] = int(df[cols].isna().any(axis=1).sum())
        return counts
