import os
import json
import shutil
import joblib
import numpy as np

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

from ..helpers.logger import get_logger, console

RDLogger.DisableLog("rdApp.*")

logger = get_logger(__name__)

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

    def fit(self, smiles_list, use_cache=True):

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
        fp_path = os.path.join(self.dir_path,self.featurizer_name, "X.npy")

        # 1. Skip computation if fingerprints are already on disk and caching is enabled
        if use_cache and os.path.exists(fp_path):
            logger.info(f"Found existing descriptors at {fp_path}. Loading...")
            X = np.load(fp_path)
            self.X = X
            return self
        else:
            if smiles_list is None:
                raise ValueError("X.npy not found and no smiles_list provided to compute them.")
            
        imputer = SimpleImputer()
        feature_filter = VarianceThreshold(threshold=0.0)
        scaler = RobustScaler()

        # One row per input molecule, always. Unparseable molecules and
        # non-finite descriptors become NaN and are handled by the imputer,
        # exactly as in transform() — dropping rows here would silently
        # desynchronise this matrix from the other featurizers' matrices.
        R = []
        n_desc = len(self.features)
        for smiles in smiles_list:
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

        if X.shape[0] != len(smiles_list):
            raise RuntimeError(
                f"Descriptor matrix has {X.shape[0]:,} rows for "
                f"{len(smiles_list):,} input molecules."
            )

    # 1. Clip Raw Values (Handling super-large numbers before stats)
        X = np.clip(X, -1e5, 1e5)
        
        # 2. Impute (Fit AND Transform)
        imputer.fit(X)
        X = imputer.transform(X) # <--- THIS WAS MISSING
        
        # 3. Filter (Fit AND Transform)
        feature_filter.fit(X)
        X = feature_filter.transform(X)
        
        # 4. Scale (Fit AND Transform)
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
        for smiles in smiles_list:
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
        
        # MUST MATCH FIT ORDER EXACTLY:
        # 1. Clip
        X = np.clip(X, -1e5, 1e5) 
        
        # 2. Impute
        X = self.imputer.transform(X)
        
        # 3. Filter
        X = self.feature_filter.transform(X)
        
        # 4. Scale
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
    def load(cls, dir_path: str, load_X: bool = True):
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
                logger.debug(f"Saved RDKit version: {rdkit_version}")
            current_rdkit_version = Chem.rdBase.rdkitVersion
            if rdkit_version and current_rdkit_version != rdkit_version:
                # A warning, not an error: descriptor values are stable across most
                # RDKit releases, and refusing to load would make every published
                # reference space unusable on any other version. Routed through the
                # Rich console because loguru output is suppressed package-wide.
                logger.warning(
                    f"RDKit version mismatch: got {current_rdkit_version}, "
                    f"reference space was fitted with {rdkit_version}."
                )
                console.print(
                    f"  [bold yellow]![/bold yellow] RDKit version mismatch: this "
                    f"reference space was fitted with [bold]{rdkit_version}[/bold] but "
                    f"[bold]{current_rdkit_version}[/bold] is installed.\n"
                    f"    Descriptor values may differ slightly; install "
                    f"rdkit=={rdkit_version} for an exact reproduction.",
                    style="yellow",
                )
        obj.imputer = joblib.load(os.path.join(desc_path, "imputer.pkl"))
        obj.feature_filter = joblib.load(os.path.join(desc_path, "feature_filter.pkl"))
        obj.scaler = joblib.load(os.path.join(desc_path, "scaler.pkl"))
        if load_X:
            obj.X = np.load(os.path.join(desc_path, "X.npy"))
        return obj
