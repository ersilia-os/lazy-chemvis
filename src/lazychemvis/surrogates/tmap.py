import os
import numpy as np
from FPSim2.io import create_db_file

from ..featurizers.ecfp import ECFPFeaturizer

ARTIFACT_NAME = "tmap"


class TMAPSurrogate:
    def __init__(self, dir_path: str):
        self.dir_path = os.path.abspath(dir_path)

    def fit(self, smiles_list):
        featurizer = ECFPFeaturizer.load(dir_path=self.dir_path, load_X=False)
        coords_path = os.path.join(self.dir_path, ARTIFACT_NAME, "reduced.npy")
        self.ref_coords = np.load(coords_path)
        self.smiles_list = smiles_list
        self.radius = featurizer.radius
        self.n_bits = featurizer.n_bits

    def save(self):
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
