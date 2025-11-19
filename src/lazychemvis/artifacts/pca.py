import os
from typing import List
import torch
import joblib

from ..descriptors.rdkit_descriptor import RDKitDescriptor


class PCAArtifact(object):
    def __init__(self, dir_name: str):
        self.artifact_name = "pca_projector"
        self.dir_name = os.path.abspath(dir_name)
        self.featurizer = RDKitDescriptor.load(dir_path=self.dir_name)
        file_path = os.path.join(dir_name, "pca_projector", "pca_surrogate.pt")
        self.model = torch.load(file_path)
        self.scaler = joblib.load(
            os.path.join(dir_name, "pca_projector", "pca_axis_scaler.pkl")
        )

    def transform(self, smiles_list: List[str]):
        X = self.featurizer.transform(smiles_list)
        X_red = torch.tensor(X, dtype=torch.float32).numpy()
        X_red = self.scaler.transform(X_red)
        return X_red
