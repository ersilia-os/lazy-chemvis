import os
import gc
import json
import shutil
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from ersilia.api import Model

from ..helpers.logger import get_logger

RDLogger.DisableLog("rdApp.*")

logger = get_logger(__name__)


class CheMeleonFeaturizer(object):

    def __init__(self, dir_path: str, model_id: str = 'eos9o72'):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.featurizer_name = 'CheMeleon'
        self._model_id = model_id
        self._model_instance = None
        self.dir_path = os.path.abspath(dir_path)

    @property
    def model(self):
        """Lazy loader for Ersilia model."""
        if self._model_instance is None:
            logger.info(f"Initializing and serving model: {self._model_id}")
            self._model_instance = Model(model_id=self._model_id)
            self._model_instance.serve()
        return self._model_instance

    def _compute_fps(self, smiles_list):
        """Run the model on a list of SMILES and save per-batch .npy files."""
        total_smiles = len(smiles_list)
        batch_size = 2000

        desc_path = os.path.join(self.dir_path, self.featurizer_name)
        temp_dir = os.path.join(desc_path, "tmp_batches")
        os.makedirs(temp_dir, exist_ok=True)

        for i in tqdm(range(0, total_smiles, batch_size), desc="Processing Batches"):
            batch_idx = i // batch_size
            batch_file = os.path.join(temp_dir, f"batch_{batch_idx}.npy")

            if os.path.exists(batch_file):
                continue

            current_batch_smiles = smiles_list[i: i + batch_size]

            for attempt in range(3):
                try:
                    df_batch = self.model.run(current_batch_smiles)
                    numeric_df = df_batch.select_dtypes(include=[np.number])
                    X_batch = numeric_df.to_numpy(dtype=np.float32)

                    if X_batch.shape[0] != len(current_batch_smiles):
                        raise RuntimeError(
                            f"model returned {X_batch.shape[0]} rows for "
                            f"{len(current_batch_smiles)} molecules"
                        )

                    np.save(batch_file, X_batch)
                    del df_batch, numeric_df, X_batch
                    break
                except Exception as e:
                    if attempt < 2:
                        wait = 5 * (attempt + 1)
                        logger.warning(
                            f"Batch {batch_idx} failed ({e}); retrying in {wait}s "
                            f"(attempt {attempt + 1}/3)."
                        )
                        time.sleep(wait)
                        continue
                    # Fatal: skipping the batch would leave X.npy short and
                    # silently misaligned with the other featurizers, surfacing
                    # much later as an unexplained dimension mismatch.
                    logger.error(f"Error at batch {batch_idx}: {e}")
                    raise RuntimeError(
                        f"CheMeleon featurization failed at batch {batch_idx} "
                        f"(molecules {i:,}–{i + len(current_batch_smiles):,}) "
                        f"after 3 attempts: {e}\n"
                        "Successfully computed batches are cached on disk, so "
                        "re-running resumes from this point."
                    ) from e

    def fit(self, smiles_list):
        """Fit the featurizer by computing descriptors for the reference set."""
        desc_path = os.path.join(self.dir_path, self.featurizer_name)
        temp_dir = os.path.join(desc_path, "tmp_batches")
        x_path = os.path.join(desc_path, "X.npy")

        if os.path.exists(x_path):
            logger.info(f"{self.featurizer_name} descriptors already calculated. Loading from disk...")
            self.X = np.load(x_path)
            return self

        self._compute_fps(smiles_list)

        # Collect batch file paths in sorted order
        all_files = sorted(
            [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.npy')],
            key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
        )
        if not all_files:
            raise ValueError("No batch files found. Ensure that _compute_fps ran successfully.")

        # Determine total shape without loading data (avoids doubling peak memory)
        shapes = [np.load(f, mmap_mode='r').shape for f in all_files]
        n_total = sum(s[0] for s in shapes)
        n_feat = shapes[0][1]
        logger.info(f"Merging {len(all_files)} batch files → {n_total:,} molecules × {n_feat} features")

        # Pre-allocate and fill incrementally — peak memory = final array + one batch
        self.X = np.empty((n_total, n_feat), dtype=np.float32)
        row = 0
        for f, shape in zip(all_files, shapes):
            batch = np.load(f)
            n = shape[0]
            self.X[row: row + n] = batch
            row += n
            del batch
            gc.collect()

        if n_total != len(smiles_list):
            raise RuntimeError(
                f"CheMeleon produced {n_total:,} rows for {len(smiles_list):,} input "
                f"molecules. The reference matrices must stay row-aligned; delete "
                f"{temp_dir} and re-run."
            )

        logger.success(f"Merge complete: {n_total:,} molecules, {n_feat} features.")

        # Persist merged matrix
        np.save(x_path, self.X)
        logger.debug(f"Saved: {x_path}")

        # Clean up temp batch files now that X.npy is on disk
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary batch files at {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temp directory: {e}")

        return self

    def transform(self, smiles_list):
        """Transform SMILES into CheMeleon embeddings."""
        return self._compute_fps(smiles_list)

    def save(self):
        """Save the fitted featurizer metadata and training matrix to disk."""
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
            logger.debug("Saved: X.npy")

        logger.success("CheMeleon featurizer saved successfully.")

    @classmethod
    def load(cls, dir_path: str):
        """Load a previously saved CheMeleonFeaturizer."""
        desc_path = os.path.join(dir_path, "CheMeleon")
        with open(os.path.join(desc_path, "featurizer.json"), "r") as f:
            metadata = json.load(f)

        obj = cls(dir_path=metadata["dir_path"], model_id=metadata["model_id"])

        x_path = os.path.join(desc_path, "X.npy")
        if os.path.exists(x_path):
            obj.X = np.load(x_path)
            logger.debug(f"Loaded: X.npy ({obj.X.shape[0]:,} molecules)")

        return obj

    def cleanup(self):
        """Shut down served container and remove any leftover temp files."""
        if self._model_instance:
            try:
                self._model_instance.close()
                logger.info(f"Closed Ersilia model: {self._model_id}")
            except Exception as e:
                logger.warning(f"Error closing model: {e}")
            finally:
                self._model_instance = None

        desc_path = os.path.join(self.dir_path, self.featurizer_name)
        temp_dir = os.path.join(desc_path, "tmp_batches")
        x_path = os.path.join(desc_path, "X.npy")

        if os.path.exists(x_path) and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary batch files at {temp_dir}")
            except Exception as e:
                logger.warning(f"Could not remove temp directory: {e}")
