import os
import gc
import json
import shutil
import time
import pandas as pd
import numpy as np

from rdkit import RDLogger

from ..helpers.cache import cache_key, invalidate, mismatch_reason, write_key
from ..helpers.console import quiet
from ..helpers.ersilia_model import close_model, serve_model
from ..helpers.live import LiveProgressBar
from ..helpers.logger import get_logger

RDLogger.DisableLog("rdApp.*")

logger = get_logger(__name__)


class CLAMPFeaturizer(object):

    def __init__(self, dir_path: str, model_id: str = 'eos3l5f'):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.featurizer_name = 'CLAMP'
        self._model_id = model_id
        self._model_instance = None
        self.dir_path = os.path.abspath(dir_path)
        self.batch_size = 2000
        self._from_cache = False

    @property
    def model(self):
        """Lazy loader for Ersilia model. Fetches the model if it is missing."""
        if self._model_instance is None:
            logger.info(f"Initializing and serving model: {self._model_id}")
            self._model_instance = serve_model(self._model_id)
        return self._model_instance

    def _cache_key(self, smiles_list):
        """Key identifying this featurizer's output (and its batches) for ``smiles_list``."""
        return cache_key(
            smiles_list,
            featurizer=self.featurizer_name,
            model_id=self._model_id,
            batch_size=self.batch_size,
        )

    def _compute_fps(self, smiles_list):
        """Run the model on a list of SMILES and save per-batch .npy/.csv files."""
        total_smiles = len(smiles_list)
        batch_size = self.batch_size
        desc_path = os.path.join(self.dir_path, self.featurizer_name)
        temp_dir = os.path.join(desc_path, "tmp_batches")

        # Batches are only resumable if they came from this exact library and batch
        # size; batches left behind by a different fit would be merged in silently.
        key = self._cache_key(smiles_list)
        reason = mismatch_reason(temp_dir, key)
        if reason is not None:
            invalidate([temp_dir], f"{self.featurizer_name} batches", reason)
        os.makedirs(temp_dir, exist_ok=True)
        write_key(temp_dir, key)

        # Fetch and serve up front, but only if there is anything left to
        # compute: a missing model or a stopped Docker daemon should surface as
        # itself rather than as three retried "batch failed" warnings.
        if any(
            not os.path.exists(os.path.join(temp_dir, f"batch_{i // batch_size}.npy"))
            for i in range(0, total_smiles, batch_size)
        ):
            _ = self.model

        starts = list(range(0, total_smiles, batch_size))
        progress = LiveProgressBar(f"{self.featurizer_name} batches", total=len(starts))

        with progress.live() as bar:
            for i in starts:
                bar.advance()
                batch_idx = i // batch_size
                batch_file = os.path.join(temp_dir, f"batch_{batch_idx}.npy")
                info_file = os.path.join(temp_dir, f"info_{batch_idx}.csv")

                if os.path.exists(batch_file):
                    bar.set_note(f"batch {batch_idx} cached")
                    continue

                current_batch_smiles = smiles_list[i: i + batch_size]
                bar.set_note(f"molecules {i:,}–{i + len(current_batch_smiles):,}")
                self._run_batch(batch_idx, i, current_batch_smiles, batch_file, info_file)

    def _run_batch(self, batch_idx, i, current_batch_smiles, batch_file, info_file):
        """Run one batch through the model, with retries, and persist it."""
        for attempt in range(3):
            try:
                with quiet(logger, label=f"ersilia batch {batch_idx}"):
                    df_batch = self.model.run(current_batch_smiles)

                returned = df_batch['input'].tolist()
                batch_info = pd.DataFrame({
                    'original_index': [i + idx for idx in range(len(returned))],
                    'smiles': returned,
                })

                numeric_df = df_batch.select_dtypes(include=[np.number])
                X_batch = numeric_df.to_numpy(dtype=np.float32)

                if X_batch.shape[0] != len(current_batch_smiles):
                    raise RuntimeError(
                        f"model returned {X_batch.shape[0]} rows for "
                        f"{len(current_batch_smiles)} molecules"
                    )

                # The .npy is what marks a batch as done, so write the metadata
                # first and the matrix last, via write-then-rename: a run killed
                # mid-write cannot leave a truncated batch that looks complete.
                batch_info.to_csv(info_file, index=False)
                partial = batch_file + ".part"
                with open(partial, "wb") as f:
                    np.save(f, X_batch)
                os.replace(partial, batch_file)

                del df_batch, batch_info, numeric_df, X_batch
                return
            except Exception as e:
                if attempt < 2:
                    wait = 5 * (attempt + 1)
                    logger.warning(
                        f"Batch {batch_idx} failed ({e}); retrying in {wait}s "
                        f"(attempt {attempt + 1}/3)."
                    )
                    time.sleep(wait)
                    continue
                # Fatal for the same reason as in the CheMeleon featurizer:
                # a skipped batch silently drops molecules from the reference set.
                logger.error(f"Error at batch {batch_idx}: {e}")
                raise RuntimeError(
                    f"CLAMP featurization failed at batch {batch_idx} "
                    f"(molecules {i:,}–{i + len(current_batch_smiles):,}) "
                    f"after 3 attempts: {e}\n"
                    "Successfully computed batches are cached on disk, so "
                        "re-running resumes from this point."
                    ) from e

    def fit(self, smiles_list, reuse=True):
        """
        Compute CLAMP embeddings for the reference set.

        Parameters
        ----------
        smiles_list : list of str
            The validated reference SMILES.
        reuse : bool, default=True
            If True, reuse an ``X.npy`` and ``valid_metadata.csv`` computed from the
            same library with the same model, and resume from matching per-batch
            files after an interrupted run. Outputs from a different library are
            discarded. If False, everything is recomputed.

        Returns
        -------
        CLAMPFeaturizer
            The fitted featurizer (self).
        """
        desc_path = os.path.join(self.dir_path, self.featurizer_name)
        temp_dir = os.path.join(desc_path, "tmp_batches")
        x_path = os.path.join(desc_path, "X.npy")
        csv_out_path = os.path.join(desc_path, "valid_metadata.csv")
        key = self._cache_key(smiles_list)
        self._from_cache = False

        reason = mismatch_reason(desc_path, key) if reuse else "--no-cache was given"
        if reason is None and os.path.exists(x_path) and os.path.exists(csv_out_path):
            logger.info(f"Reusing cached CLAMP embeddings at {x_path}")
            self.X = np.load(x_path)
            self._from_cache = True
            return self
        stale = [x_path, csv_out_path, os.path.join(desc_path, "cache.json")]
        if not reuse:
            stale.append(temp_dir)
        invalidate(stale, "CLAMP embeddings", reason or "incomplete")

        self._compute_fps(smiles_list)

        batch_files = sorted(
            [f for f in os.listdir(temp_dir) if f.startswith('batch_') and f.endswith('.npy')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )

        # Determine total shape without loading data (avoids doubling peak memory)
        full_paths = [os.path.join(temp_dir, f) for f in batch_files]
        shapes = [np.load(p, mmap_mode='r').shape for p in full_paths]
        n_total = sum(s[0] for s in shapes)
        n_feat = shapes[0][1]
        logger.info(f"Merging {len(batch_files)} batch files → {n_total:,} molecules × {n_feat} features")

        # Pre-allocate and fill incrementally — peak memory = final array + one batch
        self.X = np.empty((n_total, n_feat), dtype=np.float32)
        all_metadata = []
        row = 0

        for b_file, shape, full_path in zip(batch_files, shapes, full_paths):
            batch = np.load(full_path)
            n = shape[0]
            self.X[row: row + n] = batch
            row += n
            del batch
            gc.collect()

            info_file = b_file.replace("batch_", "info_").replace(".npy", ".csv")
            all_metadata.append(pd.read_csv(os.path.join(temp_dir, info_file)))

        self.metadata = pd.concat(all_metadata, ignore_index=True)
        del all_metadata
        gc.collect()

        if n_total != len(smiles_list):
            raise RuntimeError(
                f"CLAMP produced {n_total:,} rows for {len(smiles_list):,} input "
                f"molecules. The reference matrices must stay row-aligned; delete "
                f"{temp_dir} and re-run."
            )

        # Persist merged outputs
        np.save(x_path, self.X)
        self.metadata.to_csv(csv_out_path, index=False)
        write_key(desc_path, key)
        logger.success(f"Done. Retained {len(self.metadata):,} of {len(smiles_list):,} molecules.")

        # Clean up temp batch files now that X.npy and metadata are on disk
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary batch files at {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temp directory: {e}")

        return self

    def save(self):
        """
        Save the featurizer metadata.

        ``X.npy``, ``valid_metadata.csv`` and the cache key are written by
        :meth:`fit` as soon as the merge completes. Deliberately does **not** wipe
        the directory first: an `rmtree` here once destroyed `valid_metadata.csv` —
        which is what the UMAP surrogate reads to align the ECFP matrix with the UMAP
        coordinates — and would now also destroy the cache key.
        """
        desc_path = os.path.join(self.dir_path, self.featurizer_name)
        os.makedirs(desc_path, exist_ok=True)

        metadata = {
            "featurizer": self.featurizer_name,
            "model_id": self._model_id,
            "dir_path": self.dir_path
        }

        with open(os.path.join(desc_path, "featurizer.json"), "w") as f:
            json.dump(metadata, f)

        logger.success("CLAMP featurizer saved successfully.")

    @classmethod
    def load(cls, dir_path: str):
        """Load a previously saved CLAMPFeaturizer."""
        desc_path = os.path.join(dir_path, "CLAMP")
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
                close_model(self._model_instance, self._model_id)
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
