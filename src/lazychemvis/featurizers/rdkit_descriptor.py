import os
import json
import shutil
import joblib
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

RDLogger.DisableLog("rdApp.*")

DESCRIPTORS = [
    "MolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "FractionCSP3",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "HeavyAtomCount",
    "Chi0v",
    "Chi1v",
    "Chi2v",
    "Chi3v",
    "Kappa1",
    "Kappa2",
]


class RDKitDescriptor(object):
    """
    RDKit descriptor featurizer that computes a fixed set of molecular
    descriptors and applies preprocessing steps (imputation, variance
    filtering, and robust scaling). The fitted transformations can be
    saved and reused to ensure consistent descriptor processing across
    datasets.
    """
    def __init__(self, dir_path: str):
        """
        Initialize the RDKitDescriptor featurizer.

        Parameters
        ----------
        dir_path : str
            Directory where the featurizer parameters and metadata will be saved.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.featurizer_name = "rdkit_descriptor"
        descriptor_names = sorted(DESCRIPTORS)
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            descriptor_names
        )
        self.features = [n.lower() for n in descriptor_names]
        self.dir_path = os.path.abspath(dir_path)

    def fit(self, smiles_list):
    
        """
        Fit the descriptor preprocessing pipeline on a list of SMILES.

        This performs:
        - RDKit descriptor calculation
        - removal of molecules with invalid descriptors
        - missing value imputation (SimpleImputer)
        - zero-variance feature filtering (VarianceThreshold)
        - robust scaling (RobustScaler)

        Parameters
        ----------
        smiles_list : list of str
            List of SMILES strings used to fit the preprocessing pipeline.

        Returns
        -------
        RDKitDescriptor
            The fitted descriptor object (self).
        """
        imputer = SimpleImputer()
        feature_filter = VarianceThreshold(threshold=0.0)
        scaler = RobustScaler()
        R = []
        for smiles in tqdm(smiles_list, desc="Fitting RDKit descriptors"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            desc_values = np.array(self.calculator.CalcDescriptors(mol), dtype=float)
            if not np.all(np.isfinite(desc_values)):
                continue
            R += [desc_values]
        X = np.array(R)
        X = np.clip(X, -1e5, 1e5)
        imputer.fit(X)
        feature_filter.fit(X)
        X = feature_filter.transform(X)
        scaler.fit(X)
        X = scaler.transform(X)
        self.imputer = imputer
        self.feature_filter = feature_filter
        self.scaler = scaler
        self.X = X
        return self

    def transform(self, smiles_list):
        """
        Transform a list of SMILES into preprocessed descriptor vectors.

        Applies the previously fitted preprocessing steps:
        imputation → variance filtering → clipping → scaling.

        Parameters
        ----------
        smiles_list : list of str
            SMILES to featurize.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_molecules, n_features_after_filtering).
        """
        R = []
        n_desc = len(self.features)
        for smiles in tqdm(smiles_list, desc="Featurizing with RDKit descriptors"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError("Invalid molecule")
                desc_values = np.array(
                    self.calculator.CalcDescriptors(mol), dtype=float
                )
                desc_values[~np.isfinite(desc_values)] = np.nan
            except Exception:
                desc_values = np.array([np.nan] * n_desc, dtype=float)
            R += [desc_values]
        X = np.array(R)
        X = self.imputer.transform(X)
        X = self.feature_filter.transform(X)
        X = np.clip(X, -1e5, 1e5)
        X = self.scaler.transform(X)
        return X

    def save(self):
        """
        Save the fitted descriptor preprocessing pipeline to disk.

        This stores:
        - RDKit version metadata
        - imputer, feature filter, and scaler objects
        - the fitted descriptor matrix X
        """
        dir_path = self.dir_path
        desc_path = os.path.join(dir_path, self.featurizer_name)
        if os.path.exists(desc_path):
            shutil.rmtree(desc_path)
        os.makedirs(desc_path)
        metadata = {
            "featurizer": self.featurizer_name,
            "rdkit_version": Chem.rdBase.rdkitVersion,
        }
        with open(os.path.join(desc_path, "featurizer.json"), "w") as f:
            json.dump(metadata, f)
        joblib.dump(self.imputer, os.path.join(desc_path, "imputer.pkl"))
        joblib.dump(self.feature_filter, os.path.join(desc_path, "feature_filter.pkl"))
        joblib.dump(self.scaler, os.path.join(desc_path, "scaler.pkl"))
        numpy_path = os.path.join(desc_path, "X.npy")
        np.save(numpy_path, self.X)

    @classmethod
    def load(cls, dir_path: str):
        """
        Load a previously saved RDKitDescriptor featurizer.

        Checks RDKit version compatibility and restores the imputer,
        variance filter, scaler, and training descriptor matrix.

        Parameters
        ----------
        dir_path : str
            Directory that contains the saved featurizer.

        Returns
        -------
        RDKitDescriptor
            The loaded featurizer object.
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")
        desc_path = os.path.join(dir_path, "rdkit_descriptor")
        obj = cls(dir_path)
        with open(os.path.join(desc_path, "featurizer.json"), "r") as f:
            metadata = json.load(f)
            rdkit_version = metadata.get("rdkit_version")
            if rdkit_version:
                print(f"Loaded RDKit version: {rdkit_version}")
            current_rdkit_version = Chem.rdBase.rdkitVersion
            if current_rdkit_version != rdkit_version:
                raise ValueError(
                    f"RDKit version mismatch: got {current_rdkit_version}, expected {rdkit_version}"
                )
        obj.imputer = joblib.load(os.path.join(desc_path, "imputer.pkl"))
        obj.feature_filter = joblib.load(os.path.join(desc_path, "feature_filter.pkl"))
        obj.scaler = joblib.load(os.path.join(desc_path, "scaler.pkl"))
        obj.X = np.load(os.path.join(desc_path, "X.npy"))
        return obj
