import os

import numpy as np
from FPSim2.io import create_db_file

from ..featurizers.ecfp import ECFPFeaturizer

ARTIFACT_NAME = "tmap"


class TMAPSurrogate:
    """
    Surrogate for the TMAP projection: a fingerprint database of the reference set.

    TMAP cannot embed new points, so a new molecule later takes the coordinates of
    its most similar reference molecule (see
    :class:`~lazychemvis.artifacts.tmap.TMAPArtifact`). This stores what that lookup
    needs: an FPSim2 database of the reference fingerprints and their coordinates.

    Parameters
    ----------
    dir_path : str
        Reference space directory holding the ECFP settings and TMAP coordinates.
    """

    def __init__(self, dir_path: str):
        self.dir_path = os.path.abspath(dir_path)

    def fit(self, smiles_list):
        """
        Collect the reference molecules, coordinates and fingerprint settings.

        Parameters
        ----------
        smiles_list : list of str
            The validated reference SMILES, in the order of the TMAP coordinates.
        """
        featurizer = ECFPFeaturizer.load(dir_path=self.dir_path, load_X=False)
        coords_path = os.path.join(self.dir_path, ARTIFACT_NAME, "reduced.npy")
        self.ref_coords = np.load(coords_path)
        self.smiles_list = smiles_list
        self.radius = featurizer.radius
        self.n_bits = featurizer.n_bits

    def save(self):
        """Write ``fps.h5`` (FPSim2 database) and ``ref_coords.npy`` into ``tmap/``."""
        proj_path = os.path.join(self.dir_path, ARTIFACT_NAME)
        os.makedirs(proj_path, exist_ok=True)

        mols = [(smi, i) for i, smi in enumerate(self.smiles_list)]
        create_db_file(
            mols,
            os.path.join(proj_path, "fps.h5"),
            "smiles",
            "Morgan",
            {"radius": self.radius, "fpSize": self.n_bits},
        )
        np.save(os.path.join(proj_path, "ref_coords.npy"), self.ref_coords)
