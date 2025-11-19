import os

import torch
import torch.nn as nn

from .projectors.pca import PCAProjector


class PCAFixed(nn.Module):
    def __init__(self, pca):
        super().__init__()
        self.register_buffer("mean", torch.tensor(pca.mean_, dtype=torch.float32))
        self.register_buffer("components", torch.tensor(pca.components_, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) @ self.components.t()


class PCASurrogate(object):

    def __init__(self, dir_path: str):
        self.surrogate_name = "pca_projector"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.dir_path = os.path.abspath(dir_path)
        self.n_dim = 2

    def fit(self):
        pca = PCAProjector.load(dir_path=self.dir_path)
        pca_torch = PCAFixed(pca.reducer)
        self.model = pca_torch 

    def save(self):
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        file_path = os.path.join(proj_path, "pca_surrogate.pt")
        if os.path.exists(file_path):
            os.remove(file_path)
        torch.save(self.model.state_dict(), file_path)
