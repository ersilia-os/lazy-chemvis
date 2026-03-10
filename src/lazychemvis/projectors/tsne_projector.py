import os
import shutil
import joblib
import numpy as np
import gc
from openTSNE import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from rich.panel import Panel

from ..helpers.logger import get_logger, console

logger = get_logger(__name__)


class TSNEProjector(object):
    """
    Perform t-SNE projection using openTSNE for high-performance embedding.
    """
    def __init__(self, dir_path: str, perplexity: int = 100):
        self.projector_name = "tsne"
        self.dir_path = os.path.abspath(dir_path)
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        self.perplexity = perplexity

        # Internal State
        self.pca = None
        self.scaler = None
        self.embedding = None  # This is the TSNEEmbedding object
        self.X = None          # This stores the final 2D coordinates

    def fit(self, X=None):
        """Fit PCA and openTSNE on the descriptor matrix."""
        console.print(Panel.fit("Fitting t-SNE Projector", style="bold cyan"))

        # Load data if not provided
        if X is None:
            logger.info("[1/4] Loading CheMeleon embeddings...")
            from ..featurizers.chemeleon import CheMeleonFeaturizer
            featurizer = CheMeleonFeaturizer.load(dir_path=self.dir_path)
            X = featurizer.X
            logger.info(f"Loaded: {X.shape[0]:,} molecules × {X.shape[1]} features")

            del featurizer
            gc.collect()
        else:
            logger.info(f"[1/4] Using provided data: {X.shape[0]:,} molecules × {X.shape[1]} features")

        if X is None:
            raise ValueError("Data matrix X is empty.")

        # 1. PCA dimensionality reduction
        logger.info(f"[2/4] Performing PCA: {X.shape[1]} → 50 dimensions...")
        self.pca = PCA(n_components=50, random_state=42)
        X_pca = self.pca.fit_transform(X)
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA complete: {explained_var * 100:.2f}% variance retained.")

        del X
        gc.collect()
        logger.debug("Memory freed: original embeddings released.")

        # 2. t-SNE embedding
        logger.info(
            f"[3/4] Running t-SNE (perplexity={self.perplexity}, metric=cosine, "
            f"init=pca, negative_gradient_method=fft)..."
        )

        reducer = TSNE(
            perplexity=self.perplexity,
            metric="cosine",
            initialization="pca",
            n_jobs=-1,
            random_state=42,
            verbose=True,
            negative_gradient_method="fft"
        )

        self.embedding = reducer.fit(X_pca)
        logger.info("t-SNE embedding complete.")

        del X_pca
        gc.collect()
        logger.debug("Memory freed: PCA-reduced data released.")

        # 3. Scale coordinates to [-1, 1]
        logger.info("[4/4] Scaling coordinates to [-1, 1] range...")
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.X = self.scaler.fit_transform(self.embedding)

        logger.debug(
            f"Coordinate range: X=[{self.X[:, 0].min():.3f}, {self.X[:, 0].max():.3f}], "
            f"Y=[{self.X[:, 1].min():.3f}, {self.X[:, 1].max():.3f}]"
        )

        logger.success("t-SNE projection complete.")
        logger.info("Saving projector to disk...")
        self.save()

    def save(self):
        """Save all t-SNE components to disk."""
        proj_path = os.path.join(self.dir_path, self.projector_name)

        if os.path.exists(proj_path):
            logger.debug(f"Removing existing projection at {proj_path}")
            shutil.rmtree(proj_path)

        os.makedirs(proj_path)
        logger.info(f"Saving t-SNE projector to: {proj_path}")

        joblib.dump(self.embedding, os.path.join(proj_path, "embedding.pkl"))
        logger.debug("Saved: embedding.pkl")

        joblib.dump(self.pca, os.path.join(proj_path, "pca.pkl"))
        logger.debug("Saved: pca.pkl")

        joblib.dump(self.scaler, os.path.join(proj_path, "axis_scaler.pkl"))
        logger.debug("Saved: axis_scaler.pkl")

        joblib.dump(self.perplexity, os.path.join(proj_path, "perplexity.pkl"))
        logger.debug("Saved: perplexity.pkl")

        np.save(os.path.join(proj_path, "reduced.npy"), self.X)
        logger.debug(f"Saved: reduced.npy ({self.X.shape[0]:,} points)")

        logger.success("All t-SNE components saved successfully.")

    def cleanup(self):
        """
        Free memory by clearing large objects.

        Call this after saving to free memory before training surrogate.
        Keeps only the essential scaler for later use.
        """
        logger.info("Cleaning up t-SNE projector memory...")

        if self.embedding is not None:
            del self.embedding
            self.embedding = None
            logger.debug("Released: t-SNE embedding object.")

        if self.pca is not None:
            del self.pca
            self.pca = None
            logger.debug("Released: PCA model.")

        if self.X is not None:
            size_mb = self.X.nbytes / (1024 * 1024)
            del self.X
            self.X = None
            logger.debug(f"Released: coordinate array ({size_mb:.2f} MB).")

        gc.collect()
        logger.success("t-SNE projector memory cleaned up. Scaler retained.")

    @classmethod
    def load(cls, dir_path: str):
        """Load a previously saved t-SNE projection."""
        proj_folder = os.path.join(dir_path, "tsne")
        if not os.path.exists(proj_folder):
            raise FileNotFoundError(f"Projector folder {proj_folder} not found.")

        logger.info(f"Loading t-SNE projector from: {proj_folder}")

        perp = joblib.load(os.path.join(proj_folder, "perplexity.pkl"))
        logger.debug(f"Perplexity: {perp}")

        projector = cls(dir_path=dir_path, perplexity=perp)

        projector.embedding = joblib.load(os.path.join(proj_folder, "embedding.pkl"))
        logger.debug("Loaded: embedding.pkl")

        projector.pca = joblib.load(os.path.join(proj_folder, "pca.pkl"))
        logger.debug("Loaded: pca.pkl")

        projector.scaler = joblib.load(os.path.join(proj_folder, "axis_scaler.pkl"))
        logger.debug("Loaded: axis_scaler.pkl")

        projector.X = np.load(os.path.join(proj_folder, "reduced.npy"))
        logger.debug(f"Loaded: reduced.npy ({projector.X.shape[0]:,} points)")

        logger.success("t-SNE projector loaded successfully.")
        return projector
