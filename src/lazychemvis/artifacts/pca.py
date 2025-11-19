import os
from typing import List
import torch
import joblib
import torch
import torch.nn as nn

from ..featurizers.rdkit_descriptor import RDKitDescriptor


class PCAArtifact(object):
    def __init__(self, dir_name: str):
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
        ckpt = torch.load(path, map_location=map_location)
        n_features = ckpt["n_features"]
        n_components = ckpt["n_components"]
        class _PCALoader(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.zeros(n_features))
                self.register_buffer("components", torch.zeros(n_components, n_features))

            def forward(self, x):
                return (x - self.mean) @ self.components.T
        model = _PCALoader()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def transform(self, smiles_list: List[str]):
        X = self.featurizer.transform(smiles_list)
        X = self.model(torch.tensor(X, dtype=torch.float32)).numpy()
        X = self.scaler.transform(X)
        return X
