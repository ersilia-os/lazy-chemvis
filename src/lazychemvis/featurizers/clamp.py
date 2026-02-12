import os
import json
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from ersilia.api import Model

RDLogger.DisableLog("rdApp.*")

class CLAMPFeaturizer(object):

    def __init__(self,dir_path:str, model_id:str='eos3l5f'):

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        self.featurizer_name = 'CLAMP'
        self._model_id = model_id
        self._model_instance = None
        self.dir_path = os.path.abspath(dir_path)

    @property
    def model(self):
        """Lazy loader for Ersilia model"""
        if self._model_instance is None:
            print(f"Initializing and serving model:{self._model_id}")
            self._model_instance = Model(model_id=self._model_id)
        #    self._model_instance.fetch()
            self._model_instance.serve()

        return self._model_instance
    

    def _compute_fps(self, smiles_list):
        """
        Runs the model on  a list of SMILES
        """
        total_smiles = len(smiles_list)
        batch_size = 2000

        desc_path = os.path.join(self.dir_path, self.featurizer_name)
        temp_dir = os.path.join(desc_path,"tmp_batches")

        for i in tqdm(range(0,total_smiles, batch_size), desc = "Processing Batches"):
            batch_idx = i // batch_size
            batch_file = os.path.join(temp_dir, f"batch_{batch_idx}.npy")
            
            if os.path.exists(batch_file):
                continue
                
            # SLICE the list: from i to i + batch_size
            current_batch_smiles = smiles_list[i : i + batch_size]
            
            try:
                df_batch = self.model.run(current_batch_smiles)
                
                # Keep only numeric fingerprint columns
                numeric_df = df_batch.select_dtypes(include=[np.number])
                X_batch = numeric_df.to_numpy(dtype=np.float32)
                
                np.save(batch_file, X_batch)
            except Exception as e:
                print(f"Error at batch {batch_idx}: {e}")
    
    def fit(self, smiles_list):
        """
        Fit the featurizer by computing descriptors for the reference set.
        """
        desc_path = os.path.join(self.dir_path, self.featurizer_name)
        temp_dir = os.path.join(desc_path,"tmp_batches")
        x_path = os.path.join(desc_path, "X.npy")

        # --- NEW CHECK: Global Skip ---
        if os.path.exists(x_path):
            print(f"[*] {self.featurizer_name} descriptors already calculated. Loading from disk...")
            self.X = np.load(x_path)
            return self

        self._compute_fps(smiles_list)

        #Merge files
        all_files = sorted(
            [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.npy')],
            key= lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
        )
        if not all_files:
            raise ValueError("No batch files found. Ensure that _compute_fps ran successfully.")
        else:
            self.X = np.concatenate([np.load(f) for f in all_files], axis=0)

        return self

    def transform(self, smiles_list):
        """
        Transform SMILES into ECFP.
        """
        X = self._compute_fps(smiles_list)
        return X

    def save(self):
        """
        Save the fitted featurizer metadata and training matrix to disk.
        """
        desc_path = os.path.join(self.dir_path, self.featurizer_name)
        if os.path.exists(desc_path):
            shutil.rmtree(desc_path)
        os.makedirs(desc_path)

        metadata = {
            "featurizer": self.featurizer_name,
            "model_id": self._model_id,
            "dir_path": self.dir_path
        }

        with open(os.path.join(desc_path, "featurizer.json"), "w") as f:
            json.dump(metadata, f)

        if self.X is not None:
            np.save(os.path.join(desc_path, "X.npy"), self.X)

    @classmethod
    def load(cls, dir_path: str):
        """
        Load a previously saved CLAMPFeaturizer.
        """
        desc_path = os.path.join(dir_path, "CLAMP")
        with open(os.path.join(desc_path, "featurizer.json"), "r") as f:
            metadata = json.load(f)

        obj = cls(
            dir_path=metadata["dir_path"],
            model_id=metadata["model_id"]
        )

        x_path = os.path.join(desc_path, "X.npy")
        if os.path.exists(x_path):
            obj.X = np.load(x_path)

        return obj

    def cleanup(self):
        "Shut down served container"
        if self._model_instance:
            self._model_instance.close()