"""
UMAP projection module for CheMeleon embeddings.

This module provides the UMAPProjector class, which applies a fitted CheMeleon
featurizer, performs UMAP dimensionality reduction to 2D, and scales the 
resulting coordinates to the range [-1, 1].
"""

import os
import shutil
import joblib
import numpy as np
import umap
from sklearn.preprocessing import MinMaxScaler

# Assuming CheMeleonFeaturizer is in the same project structure
from ..featurizers.mole import MolEFeaturizer


class UMAPProjector(object):
    """
    Perform UMAP projection on CheMeleon descriptor features and scale the output.

    This class:
      - Loads a previously fitted CheMeleonFeaturizer.
      - Fits a UMAP model to the descriptor matrix.
      - Transforms the data into a 2D UMAP space.
      - Scales the resulting coordinates to [-1, 1] with MinMaxScaler.
      - Saves and loads all components from disk.
    """

    def __init__(self, dir_path: str, n_neighbors: int = 15, min_dist: float = 0.1, metric: str = 'euclidean'):
        """
        Create a UMAPProjector.

        Parameters
        ----------
        dir_path : str
            Directory where the featurizer is stored and results will be saved.
        n_neighbors : int, default=15
            The size of local neighborhood used for manifold approximation.
        min_dist : float, default=0.1
            The effective minimum distance between embedded points.
        metric : str, default='euclidean'
            The metric to use to compute distances in high dimensional space.
        """
        self.projector_name = "umap"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.dir_path = os.path.abspath(dir_path)
        self.n_dim = 2
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric

    def fit(self):
        """
        Fit UMAP on the stored CheMeleon descriptor matrix.
        """
        # 1. Load the featurizer
        featurizer = MolEFeaturizer.load(dir_path=self.dir_path)
        X = featurizer.X
        
        if X is None:
            raise ValueError("Featurizer matrix X is empty. Run featurizer.fit() first.")

        # 2. Fit UMAP
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_dim,
            metric=self.metric,
            random_state=42  # For reproducibility
        )
        
        X_embedded = reducer.fit_transform(X)
        self.reducer = reducer

        # 3. Scale to [-1, 1]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.X = scaler.fit_transform(X_embedded)
        self.scaler = scaler

    def save(self):
        """
        Save the UMAP model, scaler, and projected coordinates to disk.
        """
        proj_path = os.path.join(self.dir_path, self.projector_name)
        if os.path.exists(proj_path):
            shutil.rmtree(proj_path)
        os.makedirs(proj_path)
        
        joblib.dump(self.reducer, os.path.join(proj_path, "orig.pkl"))
        joblib.dump(self.scaler, os.path.join(proj_path, "axis_scaler.pkl"))
        np.save(os.path.join(proj_path, "reduced.npy"), self.X)

    @classmethod
    def load(cls, dir_path: str):
        """
        Load a previously saved UMAPProjector from disk.
        """
        projector = cls(dir_path=dir_path)
        proj_folder = os.path.join(dir_path, "umap")
        
        projector.reducer = joblib.load(os.path.join(proj_folder, "orig.pkl"))
        projector.scaler = joblib.load(os.path.join(proj_folder, "axis_scaler.pkl"))
        projector.X = np.load(os.path.join(proj_folder, "reduced.npy"))
        
        return projector