"""
ECFP (Morgan fingerprint) featurizer.

This module provides the ECFPFeaturizer class, which computes binary Morgan
fingerprints from SMILES. The fingerprint settings and the reference matrix can
be saved and reloaded reproducibly.
"""

import os
import json
import shutil
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from ..helpers.cache import cache_key, invalidate, mismatch_reason, write_key
from ..helpers.logger import get_logger

RDLogger.DisableLog("rdApp.*")

logger = get_logger(__name__)


class ECFPFeaturizer(object):
    """
    Featurizer that computes binary extended-connectivity fingerprints
    (ECFP/Morgan). The fingerprints are used unprocessed: as input to TMAP, and as
    the features of the t-SNE and UMAP surrogates.
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
        """
        Compute fingerprints for a list of SMILES, one row per input molecule.

        Unparseable molecules are left as all-zero rows so that the row count
        always matches the input. At fit time this cannot happen, because the
        pipeline validates SMILES before featurizing; at transform time an
        all-zero fingerprint is reported rather than silently ignored.
        """
        X = np.zeros((len(smiles_list), self.n_bits), dtype="int8")
        n_invalid = 0
        for i, smi in enumerate(smiles_list):
            fp = self._compute_fp(smi)
            if fp is None:
                n_invalid += 1
                continue
            X[i, :] = fp

        if n_invalid:
            logger.warning(
                f"{n_invalid:,} of {len(smiles_list):,} molecules could not be parsed "
                f"by RDKit — their fingerprints are all-zero, so their coordinates are "
                f"not meaningful."
            )

        return X

    def _cache_key(self, smiles_list):
        """Key identifying this featurizer's output for ``smiles_list``."""
        return cache_key(
            smiles_list,
            featurizer=self.featurizer_name,
            radius=self.radius,
            n_bits=self.n_bits,
            rdkit_version=Chem.rdBase.rdkitVersion,
        )

    def fit(self, smiles_list, reuse=True):
        """
        Compute the reference fingerprint matrix for a list of SMILES.

        Parameters
        ----------
        smiles_list : list of str
            Molecules used as the reference set.
        reuse : bool, default=True
            If True and ``dir_path`` already holds fingerprints computed from the
            same library with the same radius and length, load them instead of
            recomputing. Fingerprints from a different library are discarded.

        Returns
        -------
        ECFPFeaturizer
            The fitted featurizer (self).
        """
        desc_path = os.path.join(self.dir_path, self.featurizer_name)
        self._key = self._cache_key(smiles_list)
        self._from_cache = False

        reason = mismatch_reason(desc_path, self._key) if reuse else "--no-cache was given"
        if reason is None:
            logger.info(f"Reusing cached fingerprints at {desc_path}")
            self.X = np.load(os.path.join(desc_path, "X.npy"))
            self._from_cache = True
            return self
        invalidate([desc_path], "ECFP fingerprints", reason)

        logger.info(f"Computing fingerprints for {len(smiles_list):,} molecules...")
        self.X = self._compute_fps(smiles_list)

        return self

    def transform(self, smiles_list):
        """
        Compute fingerprints for SMILES with the stored radius and length.

        Parameters
        ----------
        smiles_list : list of str
            Molecules to featurize.

        Returns
        -------
        numpy.ndarray
            Binary array of shape (n_molecules, n_bits).
        """
        return self._compute_fps(smiles_list)

    def save(self):
        """
        Save the featurizer settings and fingerprint matrix to disk.

        The cache key is written last. Nothing is rewritten when :meth:`fit`
        reused a cached output.
        """
        if getattr(self, "_from_cache", False):
            return
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
        if getattr(self, "_key", None) is not None:
            write_key(desc_path, self._key)

    @classmethod
    def load(cls, dir_path: str, load_X: bool = True):
        """
        Load a previously saved ECFPFeaturizer.

        Returns
        -------
        ECFPFeaturizer
            Featurizer with its radius and length restored.
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
