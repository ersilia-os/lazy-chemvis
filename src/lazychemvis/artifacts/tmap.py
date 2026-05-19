import os
import numpy as np
from typing import List
from FPSim2 import FPSim2Engine


class TMAPArtifact:
    def __init__(self, dir_name: str):
        self.dir_name = os.path.abspath(dir_name)
        proj_path = os.path.join(self.dir_name, "tmap")

        self.engine = FPSim2Engine(
            os.path.join(proj_path, "fps.h5"), in_memory_fps=True
        )
        self.ref_coords = np.load(os.path.join(proj_path, "ref_coords.npy"))

    def transform(self, smiles_list: List[str]):
        indices = []
        for smi in smiles_list:
            results = self.engine.similarity(smi, 0.0, n_workers=1)
            indices.append(int(results[0]["mol_id"]))
        return self.ref_coords[indices]
