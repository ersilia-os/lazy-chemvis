"""
PCA projection module.

This module provides the PCAProjector class, which applies a fitted RDKitDescriptor
featurizer, performs PCA dimensionality reduction to 2D, and scales the resulting
coordinates to the range [-1, 1]. The trained PCA model, scaler, and projected
coordinates can be saved and loaded from disk.
"""

import os
import shutil
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from ..featurizers.rdkit_descriptor import RDKitDescriptor


class PCAProjector(object):
    """
    Perform PCA projection on RDKit descriptor features and scale the output.

    This class:
      - Loads a previously fitted RDKitDescriptor featurizer.
      - Fits a PCA model to the descriptor matrix.
      - Transforms the data into a 2D PCA space.
      - Scales the resulting coordinates to [-1, 1] with MinMaxScaler.
      - Saves and loads all required PCA components from disk.
    """

    def __init__(self, dir_path: str):
        """
        Create a PCAProjector.

        Parameters
        ----------
        dir_path : str
            Directory where the RDKit descriptor featurizer is stored and where all
            PCA-related files (model, scaler, projections) will be saved.
        """
        self.projector_name = "pca"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.dir_path = os.path.abspath(dir_path)
        self.n_dim = 2

    def fit(self):
        """
        Fit PCA on the stored RDKit descriptor matrix and scale the 2D projection.

        This method:
          - Loads the RDKitDescriptor object from `dir_path`.
          - Fits PCA with `n_dim` components (default: 2).
          - Generates the PCA projection.
          - Fits a MinMaxScaler on the PCA projection.
          - Stores the PCA model, scaler, and scaled projection.

        Returns
        -------
        None
        """
        featurizer = RDKitDescriptor.load(dir_path=self.dir_path)
        X = featurizer.X
        reducer = PCA(n_components=self.n_dim)
        reducer.fit(X)
        X = reducer.transform(X)
        self.reducer = reducer
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
        self.scaler = scaler
        X = scaler.transform(X)
        self.X = X

    def save(self):
        """
        Save the PCA model, scaler, and projected coordinates to disk.

        Files saved in {dir_path}/pca/:
          - 'orig.pkl'        : PCA reducer (sklearn PCA object)
          - 'axis_scaler.pkl' : MinMaxScaler used on PCA coordinates
          - 'reduced.npy'     : Scaled 2D PCA coordinates

        Returns
        -------
        None
        """
        proj_path = os.path.join(self.dir_path, self.projector_name)
        if os.path.exists(proj_path):
            shutil.rmtree(proj_path)
        os.makedirs(proj_path)
        joblib.dump(self.reducer, os.path.join(proj_path, "orig.pkl"))
        joblib.dump(self.scaler, os.path.join(proj_path, "axis_scaler.pkl"))
        numpy_path = os.path.join(proj_path, "reduced.npy")
        np.save(numpy_path, self.X)

    @classmethod
    def load(cls, dir_path: str):
        """
        Load a previously saved PCAProjector from disk.

        Parameters
        ----------
        dir_path : str
            Directory containing the 'pca' subfolder with the saved PCA model.

        Returns
        -------
        PCAProjector
            A PCAProjector instance with loaded PCA model, scaler, and coordinates.
        """
        projector = cls(dir_path=dir_path)
        projector.reducer = joblib.load(os.path.join(dir_path, "pca", "orig.pkl"))
        projector.scaler = joblib.load(os.path.join(dir_path, "pca", "axis_scaler.pkl"))
        numpy_path = os.path.join(dir_path, "pca", "reduced.npy")
        projector.X = np.load(numpy_path)
        return projector
