import os
import json
import shutil
import joblib
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

RDLogger.DisableLog("rdApp.*")


class RDKitDescriptor(object):
    def __init__(self, dir_path: str = None):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.featurizer_name = "rdkit_descriptor"
        descriptor_names = sorted([desc_name for desc_name, _ in Descriptors._descList])
        self.calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            descriptor_names
        )
        self.features = [n.lower() for n in descriptor_names]

    def fit(self, smiles_list):
        imputer = SimpleImputer()
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
        X = np.clip(np.vstack(X), -1e5, 1e5)
        imputer.fit(X)
        scaler.fit(X)
        self.imputer = imputer
        self.scaler = scaler
        return self

    def transform(self, smiles_list):
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
        X = np.clip(np.vstack(X), -1e5, 1e5)
        X = self.scaler.transform(X)
        return X

    def save(self):
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
        joblib.dump(self.scaler, os.path.join(desc_path, "scaler.pkl"))

    @classmethod
    def load(cls, dir_path: str):
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory {dir_path} does not exist.")
        desc_path = os.path.join(dir_path, "rdkit_descriptor")
        obj = cls()
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
        obj.scaler = joblib.load(os.path.join(desc_path, "scaler.pkl"))
        return obj
