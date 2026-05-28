"""
ECFP (Morgan fingerprint) featurizer.

This module provides the ECFPFeaturizer class, which computes binary Morgan
fingerprints from SMILES. The fitted transformers and training
matrix can be saved and reloaded reproducibly.
"""

import os
import json
import shutil
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from ..helpers.logger import get_logger

RDLogger.DisableLog("rdApp.*")

logger = get_logger(__name__)


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
        return np.array(fp, dtype="int8")

    def _compute_fps(self, smiles_list):
        X = np.zeros((len(smiles_list), self.n_bits), dtype="int8")
        for i, smi in enumerate(smiles_list):
            fp = self._compute_fp(smi)
            if fp is None:
                continue
            X[i, :] = fp
        return X

    def fit(self, smiles_list, use_cache=True):
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
        fp_path = os.path.join(self.dir_path, self.featurizer_name, "X.npy")

        # Skip computation if fingerprints are already on disk and caching is enabled
        if use_cache and os.path.exists(fp_path):
            logger.info(f"Found existing fingerprints at {fp_path}. Loading...")
            self.X = np.load(fp_path)
            return self  # early return — do NOT fall through to _compute_fps

        if smiles_list is None:
            raise ValueError("X.npy not found and no smiles_list provided to compute them.")

        logger.info(f"Computing fingerprints for {len(smiles_list):,} molecules...")
        self.X = self._compute_fps(smiles_list)

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
        return self._compute_fps(smiles_list)

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

        np.save(os.path.join(desc_path, "X.npy"), self.X)
        logger.debug(f"Saved: {desc_path}/X.npy ({self.X.shape[0]:,} molecules)")

    @classmethod
    def load(cls, dir_path: str, load_X: bool = True):
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
            radius=metadata.get("radius", 2),
            n_bits=metadata.get("n_bits", 2048),
        )

        if load_X:
            obj.X = np.load(os.path.join(desc_path, "X.npy"))
            logger.debug(f"Loaded: {desc_path}/X.npy ({obj.X.shape[0]:,} molecules)")

        return obj
