import os
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# We use ECFP for the surrogate input (X) to avoid API dependency
from ..featurizers.ecfp import ECFPFeaturizer
# We pull the target coordinates (Y) from the CheMeleon-based t-SNE
from ..projectors.tsne_projector import TSNEProjector 

class TSNESurrogate(object):
    """
    Shortcut Surrogate for t-SNE: 
    Maps local ECFP bits directly to CheMeleon-t-SNE coordinates.
    """
    def __init__(self, dir_path: str):
        self.surrogate_name = "tsne_surrogate"
        self.dir_path = os.path.abspath(dir_path)
        
    def fit(self):
        """
        Trains an XGBoost model to mimic the t-SNE layout using ECFP bits.
        """
        # 1. Load ECFP fingerprints (X - Input)
        ecfp_feat = ECFPFeaturizer.load(dir_path=self.dir_path)
        X = ecfp_feat.X 
        
        # 2. Load the t-SNE coordinates (Y - Target)
        # These coordinates were created using CheMeleon fingerprints + openTSNE
        tsne_projector = TSNEProjector.load(dir_path=self.dir_path)
        y_coords = tsne_projector.X 
        
        if X is None or y_coords is None:
            raise ValueError("Reference ECFP matrix or t-SNE coordinates missing.")

        print(f"Training t-SNE Shortcut: ECFP -> CheMeleon-t-SNE (2D)")

        # 3. Train the XGBoost Surrogate
        # t-SNE clusters are often very tight and complex, so we keep depth around 9-10
        base_xgb = XGBRegressor(
            n_estimators=300,
            max_depth=10, 
            learning_rate=0.05,
            tree_method='hist',
            device='cpu', 
            n_jobs=-1,
            random_state=42
        )
        
        self.model = MultiOutputRegressor(base_xgb)
        self.model.fit(X, y_coords)
        
        # Keep the axis_scaler to ensure target consistency
        self.axis_scaler = tsne_projector.scaler

    def save(self):
        """Save the surrogate regressor and the axis scaler."""
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        os.makedirs(proj_path, exist_ok=True)

        joblib.dump(self.model, os.path.join(proj_path, "xgb_model.joblib"))
        joblib.dump(self.axis_scaler, os.path.join(proj_path, "axis_scaler.pkl"))

    def load(self):
        """Loads the components for the Artifact."""
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        self.model = joblib.load(os.path.join(proj_path, "xgb_model.joblib"))
        self.axis_scaler = joblib.load(os.path.join(proj_path, "axis_scaler.pkl"))
        return self