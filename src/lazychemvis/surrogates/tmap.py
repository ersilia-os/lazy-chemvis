import os
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Importing your specific featurizer and projector
from ..featurizers.ecfp import ECFPFeaturizer
from ..projectors.tmap_projector import TMAPProjector 

class TMAPSurrogate(object):
    """
    Surrogate for TMAP that enables fast coordinate assignment via 
    Nearest Neighbor lookup using ECFP bit vectors.
    """
    def __init__(self, dir_path: str):
        self.surrogate_name = "tmap"
        self.dir_path = os.path.abspath(dir_path)
        
    def fit(self):
        """
        Builds the BallTree index using  ECFP fingerprints.
        """
        # 1. Load the ECFP reference matrix (X)
        featurizer = ECFPFeaturizer.load(dir_path=self.dir_path)
        X = featurizer.X 
        
        # 2. Load the 2D coordinates (Y) from the original TMAP run
        tmap_projector = TMAPProjector.load(dir_path=self.dir_path)
        y_coords = tmap_projector.X 
        
        if X is None or y_coords is None:
            raise ValueError("Reference ECFP matrix or TMAP coordinates missing.")

        # 3. Build a NearestNeighbors index for fast nearest-neighbor lookups
        # Metric 'jaccard' is the standard for chemical Tanimoto similarity.
        # Brute-force is used because BallTree degrades to O(N) linear scan on
        # 2048-dimensional binary data (curse of dimensionality), while brute-force
        # runs in parallel across all CPU cores via n_jobs=-1.
        print(f"Building TMAP KNN Index for {X.shape[0]} compounds...")
        self.knn_index = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='jaccard', n_jobs=-1)
        self.knn_index.fit(X)
        self.coords = y_coords

    def save(self):
        """Save the search index and the coordinate lookup table."""
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        os.makedirs(proj_path, exist_ok=True)

        # We save the BallTree index and the reference coordinates
        joblib.dump(self.knn_index, os.path.join(proj_path, "knn_index.joblib"))
        np.save(os.path.join(proj_path, "ref_coords.npy"), self.coords)
        
        # We also need to save the scaler from the original projector
        # to ensure new points are scaled the same way
        if hasattr(self, 'scaler'):
             joblib.dump(self.scaler, os.path.join(proj_path, "axis_scaler.pkl"))

    def load(self):
        """Loads the index and coordinates for validation."""
        proj_path = os.path.join(self.dir_path, self.surrogate_name)
        self.knn_index = joblib.load(os.path.join(proj_path, "knn_index.joblib"))
        self.coords = np.load(os.path.join(proj_path, "ref_coords.npy"))
        return self