import os
import torch
import torch.nn as nn

from ..projectors.pca import PCAProjector


class PCAFixed(nn.Module):
    """Non-trainable PCA implemented as a PyTorch module."""
    def __init__(self, n_features, n_components):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("components", torch.zeros(n_components, n_features))

    def forward(self, x):
        return (x - self.mean) @ self.components.T

    @classmethod
    def from_sklearn(cls, pca):
        """Construct from fitted sklearn PCA."""
        model = cls(
            n_features=pca.mean_.shape[0],
            n_components=pca.components_.shape[0],
        )
        model.mean.copy_(torch.tensor(pca.mean_, dtype=torch.float32))
        model.components.copy_(torch.tensor(pca.components_, dtype=torch.float32))
        return model

    @staticmethod
    def load(path, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        model = PCAFixed(
            n_features=ckpt["n_features"],
            n_components=ckpt["n_components"]
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "n_features": self.mean.shape[0],
            "n_components": self.components.shape[0],
        }, path)


class PCASurrogate(object):
    def __init__(self, dir_path: str):
        self.surrogate_name = "pca"
        self.dir_path = os.path.abspath(dir_path)
        os.makedirs(self.dir_path, exist_ok=True)
        self.n_dim = 2

    def fit(self):
        pca = PCAProjector.load(dir_path=self.dir_path)
        self.model = PCAFixed.from_sklearn(pca.reducer)

    def save(self):
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        os.makedirs(proj_path, exist_ok=True)

        file_path = os.path.join(proj_path, "surrogate.pt")
        if os.path.exists(file_path):
            os.remove(file_path)

        self.model.save(file_path)

    def load(self):
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        file_path = os.path.join(proj_path, "surrogate.pt")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No PCA surrogate found at: {file_path}")

        self.model = PCAFixed.load(file_path)
        return self.model
