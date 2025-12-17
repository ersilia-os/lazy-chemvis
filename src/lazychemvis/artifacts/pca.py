import os
from typing import List
import torch
import joblib
import torch.nn as nn

from ..featurizers.rdkit_descriptor import RDKitDescriptor


class PCAArtifact(object):
    """
    Wrapper for applying a stored PCA surrogate model to new molecules.

    This class loads:
      - a fitted RDKitDescriptor featurizer
      - a PyTorch PCA surrogate (fixed PCA transform)
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
                    - surrogate.pt  (PyTorch PCA model)
                    - axis_scaler.pkl (MinMaxScaler)
        """
        self.artifact_name = "pca"
        self.dir_name = os.path.abspath(dir_name)
        self.featurizer = RDKitDescriptor.load(dir_path=self.dir_name)
        file_path = os.path.join(dir_name, self.artifact_name, "surrogate.pt")
        self.model = self._load_pca_surrogate(file_path)
        self.scaler = joblib.load(
            os.path.join(dir_name, self.artifact_name, "axis_scaler.pkl")
        )

    @staticmethod
    def _load_pca_surrogate(path, map_location=None):
        """
        Load a fixed PCA model stored as a PyTorch checkpoint.

        The saved file contains:
          - PCA mean vector
          - PCA components
          - state_dict for a simple linear PCA transform module

        Parameters
        ----------
        path : str
            Path to the surrogate.pt checkpoint.
        map_location : optional
            Optional device mapping for torch.load.

        Returns
        -------
        nn.Module
            A PyTorch module implementing the PCA transform.
        """
        ckpt = torch.load(path, map_location=map_location)
        n_features = ckpt["n_features"]
        n_components = ckpt["n_components"]

        class _PCALoader(nn.Module):
            """
            Internal helper module representing a non-trainable PCA transform.
            """
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.zeros(n_features))
                self.register_buffer(
                    "components", torch.zeros(n_components, n_features)
                )

            def forward(self, x):
                """
                Apply PCA projection: (x - mean) dot components.T
                """
                return (x - self.mean) @ self.components.T

        model = _PCALoader()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def transform(self, smiles_list: List[str]):
        """
        Compute 2D PCA coordinates for a set of SMILES strings.

        Steps:
          1. Compute RDKit descriptors with the stored featurizer.
          2. Apply the PCA surrogate model (PyTorch).
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
        # Compute standardized RDKit descriptors
        X = self.featurizer.transform(smiles_list)

        # Apply fixed PCA transformation
        X = self.model(torch.tensor(X, dtype=torch.float32)).numpy()

        # Scale PCA axes to [-1, 1]
        X = self.scaler.transform(X)

        return X
