"""
ECFP (Morgan fingerprint) featurizer.

This module provides the ECFPFeaturizer class, which computes binary Morgan
fingerprints from SMILES and applies preprocessing steps consisting of
variance filtering and robust scaling. The fitted transformers and training
matrix can be saved and reloaded reproducibly.
"""

import os
import json
import shutil
import joblib
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

RDLogger.DisableLog("rdApp.*")


class ECFPFeaturizer(object):
    """
    Featurizer that computes extended-connectivity fingerprints (ECFP/Morgan)
    and applies preprocessing (variance filtering → scaling). This maintains
    consistency across datasets and prepares the features for dimensionality
    reduction (e.g., TMAP).
    """

    def __init__(self, dir_path: str, radius: int = 2, n_bits: int = 2048):
        """
        Initialize an ECFP/Morgan fingerprint featurizer.

        Parameters
        ----------
        dir_path : str
            Output directory where featurizer parameters and matrices will be saved.
        radius : int, default=2
            Morgan fingerprint radius (ECFP4 uses radius 2).
        n_bits : int, default=2048
            Length of the fingerprint bit vector.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.featurizer_name = "ecfp"
        self.radius = radius
        self.n_bits = n_bits
        self.dir_path = os.path.abspath(dir_path)

    def _compute_fp(self, smiles):
        """Compute the Morgan fingerprint vector for a single SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=self.radius, nBits=self.n_bits
        )
        return np.array(fp, dtype=float)

    def fit(self, smiles_list):
        """
        Fit the preprocessing pipeline on a list of SMILES.

        Steps:
        - compute Morgan fingerprints
        - remove invalid molecules
        - apply VarianceThreshold to remove constant bits
        - apply RobustScaler to reduce skewness

        Parameters
        ----------
        smiles_list : list of str
            Molecules used to fit the fingerprint preprocessing.

        Returns
        -------
        ECFPFeaturizer
            The fitted featurizer (self).
        """
        feature_filter = VarianceThreshold(threshold=0.0)
        scaler = RobustScaler()

        R = []
        for smi in tqdm(smiles_list, desc="Fitting ECFP descriptors"):
            fp = self._compute_fp(smi)
            if fp is not None:
                R.append(fp)

        X = np.array(R, dtype=float)

        # Fit preprocessing
        feature_filter.fit(X)
        X = feature_filter.transform(X)

        scaler.fit(X)
        X = scaler.transform(X)

        self.feature_filter = feature_filter
        self.scaler = scaler
        self.X = X

        return self

    def transform(self, smiles_list):
        """
        Transform SMILES into processed ECFP vectors using the fitted pipeline.

        Parameters
        ----------
        smiles_list : list of str
            Molecules to featurize.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_molecules, n_processed_bits).
        """
        R = []
        for smi in tqdm(smiles_list, desc="Featurizing with ECFP"):
            fp = self._compute_fp(smi)
            if fp is None:
                fp = np.zeros(self.n_bits, dtype=float)
            R.append(fp)

        X = np.array(R, dtype=float)
        X = self.feature_filter.transform(X)
        X = self.scaler.transform(X)

        return X

    def save(self):
        """
        Save the fitted featurizer (feature filter, scaler, X) to disk.
        """
        desc_path = os.path.join(self.dir_path, self.featurizer_name)
        if os.path.exists(desc_path):
            shutil.rmtree(desc_path)
        os.makedirs(desc_path)

        metadata = {
            "featurizer": self.featurizer_name,
            "radius": self.radius,
            "n_bits": self.n_bits,
            "rdkit_version": Chem.rdBase.rdkitVersion,
        }

        with open(os.path.join(desc_path, "featurizer.json"), "w") as f:
            json.dump(metadata, f)

        joblib.dump(self.feature_filter, os.path.join(desc_path, "feature_filter.pkl"))
        joblib.dump(self.scaler, os.path.join(desc_path, "scaler.pkl"))

        np.save(os.path.join(desc_path, "X.npy"), self.X)

    @classmethod
    def load(cls, dir_path: str):
        """
        Load a previously saved ECFPFeaturizer.

        Returns
        -------
        ECFPFeaturizer
            Featurizer with restored preprocessing and parameters.
        """
        desc_path = os.path.join(dir_path, "ecfp")
        with open(os.path.join(desc_path, "featurizer.json"), "r") as f:
            metadata = json.load(f)

        obj = cls(
            dir_path,
            radius=metadata["radius"],
            n_bits=metadata["n_bits"],
        )

        obj.feature_filter = joblib.load(os.path.join(desc_path, "feature_filter.pkl"))
        obj.scaler = joblib.load(os.path.join(desc_path, "scaler.pkl"))
        obj.X = np.load(os.path.join(desc_path, "X.npy"))

        return obj
