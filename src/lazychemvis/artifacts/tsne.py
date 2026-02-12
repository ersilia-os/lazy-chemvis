import os
import joblib
import numpy as np
from typing import List

# Import your specific ECFP featurizer for the input
from ..featurizers.ecfp import ECFPFeaturizer

class TSNEArtifact(object):
    """
    Wrapper for applying a stored t-SNE surrogate to new molecules.
    Uses an XGBoost Regressor to predict coordinates in the CheMeleon-t-SNE space.
    """
    def __init__(self, dir_name: str):
        """
        Parameters
        ----------
        dir_name : str
            Base directory containing 'ecfp' and 'tsne_surrogate' folders.
        """
        self.artifact_name = "tsne_surrogate"
        self.dir_name = os.path.abspath(dir_name)
        
        # 1. Load the ECFP featurizer 
        # Crucial: This ensures the radius and bit-length match what the surrogate learned
        self.featurizer = ECFPFeaturizer.load(dir_path=self.dir_name)
        
        # 2. Load the Surrogate components
        proj_path = os.path.join(self.dir_name, self.artifact_name)
        
        # The trained MultiOutput XGBoost model (learned ECFP -> t-SNE map)
        self.model = joblib.load(os.path.join(proj_path, "xgb_model.joblib"))
        
        # 3. Load the Scaler
        # Used to transform the raw XGBoost predictions into the [-1, 1] coordinate space
        self.scaler = joblib.load(os.path.join(proj_path, "axis_scaler.pkl"))

    def transform(self, smiles_list: List[str]):
        """
        Project new SMILES into the t-SNE landscape using the XGBoost surrogate.
        """
        # Step 1: Compute binary fingerprints locally
        # Result shape: (n_samples, n_bits)
        X_ecfp = self.featurizer.transform(smiles_list)
        
        # Step 2: Predict coordinates using the XGBoost model
        # This replaces the need for PCA and openTSNE.transform()
        # Result shape: (n_samples, 2)
        X_projected = self.model.predict(X_ecfp)
        
        # Step 3: Ensure the coordinates are scaled to [-1, 1]
        X_scaled = self.scaler.transform(X_projected)
        
        return X_scaled