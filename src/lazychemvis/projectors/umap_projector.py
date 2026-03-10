"""
UMAP projection module for CLAMP embeddings.

This module provides the UMAPProjector class with comprehensive logging,
memory management, and performance tracking for large-scale datasets.
"""

import os
import shutil
import joblib
import numpy as np
import gc
import time
import umap
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rich.panel import Panel

from ..featurizers.clamp import CLAMPFeaturizer
from ..helpers.logger import get_logger, console

logger = get_logger(__name__)


class UMAPProjector(object):
    """
    Perform UMAP projection on CLAMP descriptor features and scale the output.

    This class:
      - Loads a previously fitted CLAMP featurizer
      - Fits a UMAP model to the descriptor matrix
      - Transforms the data into a 2D UMAP space
      - Scales the resulting coordinates to [-1, 1]
      - Provides memory cleanup and performance tracking
    """

    def __init__(self, dir_path: str, n_neighbors: int = 150, min_dist: float = 0.4,
                 metric: str = 'cosine', low_memory: bool = True):
        """
        Create a UMAPProjector.

        Parameters
        ----------
        dir_path : str
            Directory where the featurizer is stored and results will be saved.
        n_neighbors : int, default=150
            The size of local neighborhood used for manifold approximation.
        min_dist : float, default=0.4
            The effective minimum distance between embedded points.
        metric : str, default='cosine'
            The metric to use to compute distances in high dimensional space.
        low_memory : bool, default=True
            Use memory-efficient implementation (recommended for >1M molecules).
        """
        self.projector_name = "umap"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.dir_path = os.path.abspath(dir_path)
        self.n_dim = 2
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.low_memory = low_memory

        self.feature_scaler = None
        self.reducer = None
        self.scaler = None
        self.X = None

        # Performance tracking
        self.timing = {}

    def fit(self, X=None):
        """
        Fit UMAP on the stored CLAMP descriptor matrix with performance tracking.
        """
        console.print(Panel.fit("Fitting UMAP Projector", style="bold cyan"))

        total_start = time.time()

        # Load data if not provided
        if X is None:
            logger.info("[1/4] Loading CLAMP embeddings...")
            load_start = time.time()

            featurizer = CLAMPFeaturizer.load(dir_path=self.dir_path)
            X = featurizer.X

            self.timing['load_featurizer'] = time.time() - load_start
            logger.info(
                f"Loaded: {X.shape[0]:,} molecules × {X.shape[1]} features "
                f"({self.timing['load_featurizer']:.2f}s)"
            )

            del featurizer
            gc.collect()
        else:
            logger.info(f"[1/4] Using provided data: {X.shape[0]:,} molecules × {X.shape[1]} features")

        if X is None:
            raise ValueError("Featurizer matrix X is empty. Run featurizer.fit() first.")

        n_invalid = np.sum(~np.isfinite(X))
        if n_invalid > 0:
            logger.warning(f"{n_invalid:,} invalid values detected — will be replaced with 0.")

        # 2. Feature scaling
        logger.info("[2/4] Scaling features with StandardScaler...")
        scale_start = time.time()

        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled)

        self.timing['feature_scaling'] = time.time() - scale_start
        logger.debug(
            f"Scaled data range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}] "
            f"({self.timing['feature_scaling']:.2f}s)"
        )

        del X
        gc.collect()
        logger.debug("Memory freed: original embeddings released.")

        # 3. UMAP embedding
        logger.info(
            f"[3/4] Running UMAP (n_neighbors={self.n_neighbors}, min_dist={self.min_dist}, "
            f"metric={self.metric}, low_memory={self.low_memory})..."
        )

        umap_start = time.time()

        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_dim,
            min_dist=self.min_dist,
            low_memory=self.low_memory,
            metric=self.metric,
            random_state=42,
            verbose=True
        )

        X_embedded = reducer.fit_transform(X_scaled)
        self.reducer = reducer

        self.timing['umap_embedding'] = time.time() - umap_start
        logger.info(
            f"UMAP embedding complete. "
            f"({self.timing['umap_embedding']:.2f}s / {self.timing['umap_embedding']/60:.2f} min)"
        )

        del X_scaled
        gc.collect()
        logger.debug("Memory freed: scaled features released.")

        # 4. Scale coordinates to [-1, 1]
        logger.info("[4/4] Scaling coordinates to [-1, 1] range...")
        coord_start = time.time()

        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.X = scaler.fit_transform(X_embedded)
        self.scaler = scaler

        self.timing['coordinate_scaling'] = time.time() - coord_start

        logger.debug(
            f"Coordinate range: X=[{self.X[:, 0].min():.3f}, {self.X[:, 0].max():.3f}], "
            f"Y=[{self.X[:, 1].min():.3f}, {self.X[:, 1].max():.3f}]"
        )

        del X_embedded
        gc.collect()

        self.timing['total'] = time.time() - total_start
        logger.success(
            f"UMAP projection complete. Total time: "
            f"{self.timing['total']:.2f}s ({self.timing['total']/60:.2f} min)"
        )

        for step, duration in self.timing.items():
            if step != 'total':
                pct = (duration / self.timing['total']) * 100
                logger.debug(f"  {step}: {duration:.2f}s ({pct:.1f}%)")

        logger.info("Saving projector to disk...")
        self.save()

    def save(self):
        """
        Save the UMAP model, scalers, and projected coordinates to disk.
        """
        proj_path = os.path.join(self.dir_path, self.projector_name)

        if os.path.exists(proj_path):
            logger.debug(f"Removing existing projection at {proj_path}")
            shutil.rmtree(proj_path)

        os.makedirs(proj_path)
        logger.info(f"Saving UMAP projector to: {proj_path}")

        joblib.dump(self.reducer, os.path.join(proj_path, "orig.pkl"))
        logger.debug("Saved: orig.pkl")

        joblib.dump(self.scaler, os.path.join(proj_path, "axis_scaler.pkl"))
        logger.debug("Saved: axis_scaler.pkl")

        joblib.dump(self.feature_scaler, os.path.join(proj_path, "feature_scaler.pkl"))
        logger.debug("Saved: feature_scaler.pkl")

        np.save(os.path.join(proj_path, "reduced.npy"), self.X)
        logger.debug(f"Saved: reduced.npy ({self.X.shape[0]:,} points)")

        if self.timing:
            joblib.dump(self.timing, os.path.join(proj_path, "timing.pkl"))
            logger.debug("Saved: timing.pkl")

        logger.success("All UMAP components saved successfully.")

    def cleanup(self):
        """
        Free memory by clearing large objects.

        Call this after saving to free memory before training surrogate.
        Keeps only the essential scalers for later use.
        """
        logger.info("Cleaning up UMAP projector memory...")

        if self.reducer is not None:
            del self.reducer
            self.reducer = None
            logger.debug("Released: UMAP reducer object.")

        if self.X is not None:
            size_mb = self.X.nbytes / (1024 * 1024)
            del self.X
            self.X = None
            logger.debug(f"Released: coordinate array ({size_mb:.2f} MB).")

        gc.collect()
        logger.success("UMAP projector memory cleaned up. Scalers retained.")

    @classmethod
    def load(cls, dir_path: str):
        """
        Load a previously saved UMAP projection.
        """
        proj_folder = os.path.join(dir_path, "umap")
        if not os.path.exists(proj_folder):
            raise FileNotFoundError(f"Projector folder {proj_folder} not found.")

        logger.info(f"Loading UMAP projector from: {proj_folder}")

        projector = cls(dir_path=dir_path)

        projector.reducer = joblib.load(os.path.join(proj_folder, "orig.pkl"))
        logger.debug(
            f"Loaded: orig.pkl (n_neighbors={projector.reducer.n_neighbors}, "
            f"min_dist={projector.reducer.min_dist})"
        )

        projector.feature_scaler = joblib.load(os.path.join(proj_folder, "feature_scaler.pkl"))
        logger.debug("Loaded: feature_scaler.pkl")

        projector.scaler = joblib.load(os.path.join(proj_folder, "axis_scaler.pkl"))
        logger.debug("Loaded: axis_scaler.pkl")

        projector.X = np.load(os.path.join(proj_folder, "reduced.npy"))
        logger.debug(f"Loaded: reduced.npy ({projector.X.shape[0]:,} points)")

        timing_path = os.path.join(proj_folder, "timing.pkl")
        if os.path.exists(timing_path):
            projector.timing = joblib.load(timing_path)
            if 'total' in projector.timing:
                logger.debug(
                    f"Original fit time: {projector.timing['total']:.2f}s "
                    f"({projector.timing['total']/60:.2f} min)"
                )

        logger.success("UMAP projector loaded successfully.")
        return projector
