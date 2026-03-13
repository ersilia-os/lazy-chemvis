import os
import joblib
import numpy as np
from typing import List

# Import your specific ECFP featurizer
from ..featurizers.ecfp import ECFPFeaturizer

class UMAPArtifact(object):
    """
    Wrapper for applying a stored UMAP surrogate to new molecules.
    Uses an XGBoost Regressor to predict coordinates in the CLAMP-UMAP space.
    """
    def __init__(self, dir_name: str):
        """
        Parameters
        ----------
        dir_name : str
            Base directory containing 'ecfp' and 'umap_surrogate' folders.
        """
        self.artifact_name = "umap_surrogate"
        self.dir_name = os.path.abspath(dir_name)
        
        # 1. Load the ECFP featurizer 
        self.featurizer = ECFPFeaturizer.load(dir_path=self.dir_name, load_X=False)
        
        # 2. Load the Surrogate components
        proj_path = os.path.join(self.dir_name, self.artifact_name)
        
        # Load the trained MultiOutput XGBoost model
        self.model = joblib.load(os.path.join(proj_path, "xgb_model.joblib"))
        

    def transform(self, smiles_list: List[str]):
        """
        Project new SMILES into the UMAP landscape using the XGBoost surrogate.
        """
        # Step 1: Compute binary fingerprints 

        X = self.featurizer.transform(smiles_list)
        
        # Step 2: Predict coordinates using the XGBoost model

        X_projected = self.model.predict(X)
    
        
        return X_projected