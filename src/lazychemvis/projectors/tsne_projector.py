import os
import shutil
import joblib
import numpy as np
from openTSNE import TSNE 
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

class TSNEProjector(object):
    """
    Perform t-SNE projection using openTSNE for high-performance embedding.
    """
    def __init__(self, dir_path: str, perplexity: int = 100):
        self.projector_name = "tsne"
        self.dir_path = os.path.abspath(dir_path)
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
  
        self.perplexity = perplexity

        # Internal State
        self.pca = None
        self.scaler = None
        self.embedding = None # This is the TSNEEmbedding object
        self.X = None         # This stores the final 2D coordinates

    def fit(self, X=None):
        """Fit PCA and openTSNE on the descriptor matrix."""
        if X is None:
            # Assumes your Featurizer logic here
            from ..featurizers.chemeleon import CheMeleonFeaturizer
            featurizer = CheMeleonFeaturizer.load(dir_path=self.dir_path)
            X = featurizer.X
        
        if X is None:
            raise ValueError("Data matrix X is empty.")

        # 1. PCA step 
        self.pca = PCA(n_components=50, random_state=42)
        X_pca = self.pca.fit_transform(X)

        # 2. Perform openTSNE 
        reducer = TSNE(
            perplexity=self.perplexity,
            metric="cosine",
            initialization="pca",
            n_jobs=-1,
            random_state=42,
            verbose=True,
            negative_gradient_method="fft"
        )
        
        # This returns a TSNEEmbedding object, which is required for .transform()
        self.embedding = reducer.fit(X_pca)

        # 3. Scale coordinates to [-1, 1]
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.X = self.scaler.fit_transform(self.embedding)
        
        self.save()


    def save(self):
        proj_path = os.path.join(self.dir_path, self.projector_name)
        if os.path.exists(proj_path):
            shutil.rmtree(proj_path)
        os.makedirs(proj_path)
        
        # Save the 'Logic' components
        joblib.dump(self.embedding, os.path.join(proj_path, "embedding.pkl"))
        joblib.dump(self.pca, os.path.join(proj_path, "pca.pkl"))
        joblib.dump(self.scaler, os.path.join(proj_path, "axis_scaler.pkl"))
        joblib.dump(self.perplexity, os.path.join(proj_path, "perplexity.pkl"))
        
        # Save the 'Data' 
        np.save(os.path.join(proj_path, "reduced.npy"), self.X)

    @classmethod
    def load(cls, dir_path: str):
        proj_folder = os.path.join(dir_path, "tsne")
        if not os.path.exists(proj_folder):
            raise FileNotFoundError(f"Projector folder {proj_folder} not found.")

        # Load metadata to get perplexity
        perp = joblib.load(os.path.join(proj_folder, "perplexity.pkl"))
        projector = cls(dir_path=dir_path, perplexity=perp)
        
        # Load the pipeline components
        projector.embedding = joblib.load(os.path.join(proj_folder, "embedding.pkl"))
        projector.pca = joblib.load(os.path.join(proj_folder, "pca.pkl"))
        projector.scaler = joblib.load(os.path.join(proj_folder, "axis_scaler.pkl"))
        
        # Load the 2D coordinates
        projector.X = np.load(os.path.join(proj_folder, "reduced.npy"))
        
        return projector