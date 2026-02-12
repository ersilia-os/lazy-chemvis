import os
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# We use ECFP for the input (X) because it's fast and independent of the API
from ..featurizers.ecfp import ECFPFeaturizer
# We still need the coordinates (Y) from the original CLAMP-based UMAP
from ..projectors.umap_projector import UMAPProjector 

class UMAPSurrogate(object):
    """
    Shortcut Surrogate: Maps ECFP bits directly to CLAMP-UMAP coordinates.
    Bypasses the need for the CLAMP API during inference.
    """
    def __init__(self, dir_path: str):
        self.surrogate_name = "umap_surrogate"
        self.dir_path = os.path.abspath(dir_path)
        
    def fit(self):
        """
        Learns to replicate the CLAMP landscape using ECFP inputs.
        """
        # 1. Load ECFP fingerprints 
        ecfp_feat = ECFPFeaturizer.load(dir_path=self.dir_path)
        X = ecfp_feat.X 
        
        # 2. Load the  coordinates generated previously using CLAMP + UMAP
        projector = UMAPProjector.load(dir_path=self.dir_path)
        y_coords = projector.X 
        
        if X is None or y_coords is None:
            raise ValueError("Reference ECFP matrix or UMAP coordinates missing.")

        print(f"Training Shortcut: ECFP ({X.shape[1]} bits) -> CLAMP-UMAP (2D)")
        
        # 3. Train the XGBoost Regressor
        # We use 'hist' for speed with 1.3M molecules
        base_xgb = XGBRegressor(
            n_estimators=300, 
            max_depth=9,      # Slightly deeper to capture CLAMP's complexity
            learning_rate=0.05,
            tree_method='hist',
            device='cpu',    
            random_state=42
        )
        
        self.model = MultiOutputRegressor(base_xgb)
        self.model.fit(X, y_coords)
        
        # We also need the original axis scaler to keep coordinates in [-1, 1]
        self.axis_scaler = projector.scaler

    def save(self):
        """Save the surrogate and the axis scaler."""
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        os.makedirs(proj_path, exist_ok=True)

        joblib.dump(self.model, os.path.join(proj_path, "xgb_model.joblib"))
        joblib.dump(self.axis_scaler, os.path.join(proj_path, "axis_scaler.pkl"))

    def load(self):
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        self.model = joblib.load(os.path.join(proj_path, "xgb_model.joblib"))
        self.axis_scaler = joblib.load(os.path.join(proj_path, "axis_scaler.pkl"))
        return self