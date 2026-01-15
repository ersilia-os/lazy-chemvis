import os
import joblib
import numpy as np
from typing import List

# Import your specific ECFP featurizer
from ..featurizers.ecfp import ECFPFeaturizer

class TMAPArtifact(object):
    """
    Wrapper for applying a stored TMAP surrogate to new molecules.
    Uses Nearest Neighbor 'snapping' to place new compounds.
    """
    def __init__(self, dir_name: str):
        """
        Parameters
        ----------
        dir_name : str
            Base directory containing 'ecfp' and 'tmap' folders.
        """
        self.artifact_name = "tmap"
        self.dir_name = os.path.abspath(dir_name)
        
        # 1. Load the ECFP featurizer (logic for bit radius/bits)
        self.featurizer = ECFPFeaturizer.load(dir_path=self.dir_name)
        
        # 2. Load the Surrogate components
        proj_path = os.path.join(self.dir_name, self.artifact_name)
        
        self.knn_index = joblib.load(os.path.join(proj_path, "knn_index.joblib"))
        self.ref_coords = np.load(os.path.join(proj_path, "ref_coords.npy"))
        
        # 3. Load the Scaler
        self.scaler = joblib.load(os.path.join(proj_path, "axis_scaler.pkl"))

    def transform(self, smiles_list: List[str]):
        """
        Project new SMILES into the TMAP tree.
        """
        # Step 1: Compute binary fingerprints
        # Returns (n_samples, 2048) int8 array
        X = self.featurizer.transform(smiles_list)
        
        # Step 2: Query the BallTree for the 1-Nearest Neighbor
        # This returns distances and indices of the closest reference molecules
        # Using Jaccard metric (Tanimoto) as defined in the Surrogate
        _, indices = self.knn_index.query(X, k=1)
        
        # Step 3: Map to coordinates
        # We flatten indices to get a 1D array of row positions in ref_coords
        indices = indices.flatten()
        X_projected = self.ref_coords[indices]
        
        # Step 4: Scale to [-1, 1]
        X_scaled = self.scaler.transform(X_projected)
        
        return X_scaled