import os
import shutil
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from ..featurizers.rdkit_descriptor import RDKitDescriptor


class PCAProjector(object):
    def __init__(self, dir_path: str):
        self.projector_name = "pca"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.dir_path = os.path.abspath(dir_path)
        self.n_dim = 2

    def fit(self):
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
        projector = cls(dir_path=dir_path)
        projector.reducer = joblib.load(
            os.path.join(dir_path, "pca", "orig.pkl")
        )
        projector.scaler = joblib.load(
            os.path.join(dir_path, "pca", "axis_scaler.pkl")
        )
        numpy_path = os.path.join(dir_path, "pca", "reduced.npy")
        projector.X = np.load(numpy_path)
        return projector
