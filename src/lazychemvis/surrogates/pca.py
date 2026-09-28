import os

import torch
import torch.nn as nn

from ..helpers.logger import get_logger
from ..projectors.pca import PCAProjector

logger = get_logger(__name__)


class PCAFixed(nn.Module):
    """Non-trainable PCA implemented as a PyTorch module."""

    def __init__(self, n_features, n_components):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("components", torch.zeros(n_components, n_features))

    def forward(self, x):
        """
        Project ``x`` onto the principal components.

        Parameters
        ----------
        x : torch.Tensor of shape (n_samples, n_features)
            Preprocessed descriptors.

        Returns
        -------
        torch.Tensor of shape (n_samples, n_components)
            ``(x - mean) @ components.T``, as sklearn's ``PCA.transform``.
        """
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
        """
        Load a module saved with :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the ``surrogate.pt`` checkpoint.
        map_location : optional
            Passed through to :func:`torch.load`.

        Returns
        -------
        PCAFixed
            The module, in evaluation mode.
        """
        ckpt = torch.load(path, map_location=map_location)
        model = PCAFixed(
            n_features=ckpt["n_features"], n_components=ckpt["n_components"]
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def save(self, path):
        """
        Save the mean, components and shape to a checkpoint.

        Parameters
        ----------
        path : str
            Destination file, conventionally ``surrogate.pt``.
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "n_features": self.mean.shape[0],
                "n_components": self.components.shape[0],
            },
            path,
        )


class PCASurrogate(object):
    """
    Exact surrogate for the PCA projection: the fitted PCA as a frozen linear module.

    Parameters
    ----------
    dir_path : str
        Reference space directory holding the fitted PCA projector.
    """

    def __init__(self, dir_path: str):
        self.surrogate_name = "pca"
        self.dir_path = os.path.abspath(dir_path)
        os.makedirs(self.dir_path, exist_ok=True)
        self.n_dim = 2

    def fit(self):
        """Re-express the fitted sklearn PCA as a :class:`PCAFixed` module."""
        logger.info("Fitting PCA surrogate model...")
        pca = PCAProjector.load(dir_path=self.dir_path)
        self.model = PCAFixed.from_sklearn(pca.reducer)
        del pca
        logger.success("PCA surrogate model ready.")

    def save(self):
        """Write the module to ``<dir_path>/pca/surrogate.pt``."""
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        os.makedirs(proj_path, exist_ok=True)

        file_path = os.path.join(proj_path, "surrogate.pt")
        if os.path.exists(file_path):
            os.remove(file_path)

        self.model.save(file_path)
        logger.debug(f"Saved: {file_path}")

    def load(self):
        """
        Load the module written by :meth:`save`.

        Returns
        -------
        PCAFixed
            The loaded module, also stored as ``self.model``.
        """
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        file_path = os.path.join(proj_path, "surrogate.pt")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No PCA surrogate found at: {file_path}")

        self.model = PCAFixed.load(file_path)
        logger.debug(f"Loaded: {file_path}")
        return self.model
