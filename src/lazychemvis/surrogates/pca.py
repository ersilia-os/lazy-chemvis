import os

import numpy as np

from ..helpers.logger import get_logger
from ..projectors.pca import PCAProjector

logger = get_logger(__name__)

SURROGATE_FILE = "surrogate.npz"


def project(X, mean, components):
    """
    Apply a fitted PCA: ``(X - mean) @ components.T``.

    This is exactly what sklearn's ``PCA.transform`` computes (without whitening,
    which the projector does not use), so the surrogate adds no error.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Preprocessed descriptors.
    mean : numpy.ndarray of shape (n_features,)
        The fitted ``PCA.mean_``.
    components : numpy.ndarray of shape (n_components, n_features)
        The fitted ``PCA.components_``.

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_components)
    """
    return (np.asarray(X, dtype=np.float64) - mean) @ components.T


class PCASurrogate(object):
    """
    Exact surrogate for the PCA projection: the fitted mean and components.

    Stored as a plain ``.npz`` so that transforming new molecules needs nothing
    beyond numpy.

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
        """Read the mean and components out of the fitted sklearn PCA."""
        logger.info("Fitting PCA surrogate model...")
        pca = PCAProjector.load(dir_path=self.dir_path)
        self.mean = np.asarray(pca.reducer.mean_, dtype=np.float64)
        self.components = np.asarray(pca.reducer.components_, dtype=np.float64)
        del pca
        logger.success("PCA surrogate model ready.")

    def save(self):
        """Write the mean and components to ``<dir_path>/pca/surrogate.npz``."""
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        os.makedirs(proj_path, exist_ok=True)
        file_path = os.path.join(proj_path, SURROGATE_FILE)
        np.savez(file_path, mean=self.mean, components=self.components)
        logger.debug(f"Saved: {file_path}")

    def load(self):
        """
        Load the mean and components written by :meth:`save`.

        Returns
        -------
        PCASurrogate
            self, with ``mean`` and ``components`` set.
        """
        file_path = os.path.join(self.dir_path, self.surrogate_name, SURROGATE_FILE)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No PCA surrogate found at: {file_path}")
        with np.load(file_path) as data:
            self.mean = data["mean"]
            self.components = data["components"]
        logger.debug(f"Loaded: {file_path}")
        return self

    def transform(self, X):
        """Project preprocessed descriptors onto the principal components."""
        return project(X, self.mean, self.components)
