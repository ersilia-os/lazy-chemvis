import os
import shutil
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from ..featurizers.rdkit_descriptor import RDKitDescriptor
from typing import List


class PCAProjector(object):

    def __init__(self, dir_path: str):
        self.projector_name = "pca_projector"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.dir_path = os.path.abspath(dir_path)
        self.n_dim = 2

    def fit(self, smiles_list: List[str]):
        featurizer = RDKitDescriptor.load(dir_path=self.dir_path)
        X = featurizer.transform(smiles_list)
        reducer = PCA(n_components=self.n_dim)
        reducer.fit(X)
        X_reduced = reducer.transform(X)
        self.reducer = reducer
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X_reduced)
        self.scaler = scaler
    
    def save(self):
        proj_path = os.path.join(self.dir_path, self.projector_name)
        if os.path.exists(proj_path):
            shutil.rmtree(proj_path)
        os.makedirs(proj_path)
        joblib.dump(self.reducer, os.path.join(proj_path, "pca_orig.pkl"))
        joblib.dump(self.scaler, os.path.join(proj_path, "pca_axis_scaler.pkl"))

    @classmethod
    def load(cls, dir_path: str):
        projector = cls(dir_path=dir_path)
        projector.reducer = joblib.load(os.path.join(
            dir_path, "pca_projector", "pca_orig.pkl"
        ))
        projector.scaler = joblib.load(os.path.join(
            dir_path, "pca_projector", "pca_axis_scaler.pkl"
        ))
        return projector

    