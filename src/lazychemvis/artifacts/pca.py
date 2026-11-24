import os
from typing import List

import joblib
import numpy as np

from ..featurizers.rdkit_descriptor import RDKitDescriptor
from ..helpers.logger import get_logger
from ..surrogates.pca import SURROGATE_FILE, project

logger = get_logger(__name__)


class PCAArtifact(object):
    """
    Wrapper for applying a stored PCA surrogate model to new molecules.

    This class loads:
      - a fitted RDKitDescriptor featurizer
      - the fitted PCA mean and components
      - a MinMaxScaler for final coordinate scaling

    It provides a `transform()` method that takes SMILES strings and returns
    2D PCA coordinates consistent with the pretrained chemical space.
    """

    def __init__(self, dir_name: str):
        """
        Initialize the PCAArtifact by loading its components from disk.

        Parameters
        ----------
        dir_name : str
            Path to the directory containing:
              - the RDKit descriptor featurizer
              - the "pca" folder with:
                    - surrogate.npz (PCA mean and components), or, in reference
                      spaces fitted before it existed, orig.pkl (sklearn PCA)
                    - axis_scaler.pkl (MinMaxScaler)
        """
        self.artifact_name = "pca"
        self.dir_name = os.path.abspath(dir_name)
        self.featurizer = RDKitDescriptor.load(dir_path=self.dir_name, load_X=False)
        proj_path = os.path.join(self.dir_name, self.artifact_name)
        self.mean, self.components = self._load_pca(proj_path)
        self.scaler = joblib.load(os.path.join(proj_path, "axis_scaler.pkl"))

    @staticmethod
    def _load_pca(proj_path):
        """
        Return the fitted PCA mean and components.

        Reference spaces fitted before ``surrogate.npz`` existed stored the same
        two arrays in a PyTorch ``surrogate.pt``. Those spaces also hold the fitted
        sklearn PCA as ``orig.pkl``, which carries them too, so they keep working
        without PyTorch.
        """
        npz_path = os.path.join(proj_path, SURROGATE_FILE)
        if os.path.exists(npz_path):
            with np.load(npz_path) as data:
                return data["mean"], data["components"]

        orig_path = os.path.join(proj_path, "orig.pkl")
        if os.path.exists(orig_path):
            logger.debug(
                f"No {SURROGATE_FILE}; reading the fitted PCA from {orig_path}"
            )
            pca = joblib.load(orig_path)
            return (
                np.asarray(pca.mean_, dtype=np.float64),
                np.asarray(pca.components_, dtype=np.float64),
            )

        raise FileNotFoundError(
            f"PCA artifact not found in {proj_path}: expected {SURROGATE_FILE} or "
            f"orig.pkl. Reference spaces holding only surrogate.pt were made by an "
            f"older LazyChemVis; refit the space with `lazychemvis fit`."
        )

    def transform(self, smiles_list: List[str]):
        """
        Compute 2D PCA coordinates for a set of SMILES strings.

        Steps:
          1. Compute RDKit descriptors with the stored featurizer.
          2. Project them onto the fitted principal components.
          3. Scale the resulting coordinates to [-1, 1] using the stored MinMaxScaler.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings to transform.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Scaled PCA coordinates.
        """
        X = self.featurizer.transform(smiles_list)
        X = project(X, self.mean, self.components)
        return self.scaler.transform(X)
