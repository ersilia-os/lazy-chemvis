import os

import pandas as pd

from ..helpers.logger import get_logger
from .base import XGBSurrogate

logger = get_logger(__name__)


class UMAPSurrogate(XGBSurrogate):
    """
    Shortcut surrogate for UMAP: maps ECFP bits directly to CLAMP-UMAP coordinates,
    bypassing the need for the CLAMP model during inference.

    See :class:`~lazychemvis.surrogates.base.XGBSurrogate` for the training procedure.
    """

    surrogate_name = "umap_surrogate"
    projection_name = "umap"
    label = "UMAP"

    def _load_targets(self):
        """Return the UMAP reference coordinates and their axis scaler."""
        # Imported here so that importing this module does not pull in umap/numba.
        from ..projectors.umap_projector import UMAPProjector

        projector = UMAPProjector.load(dir_path=self.dir_path)
        return projector.X, projector.scaler

    def _align_features(self, X):
        """
        Select the ECFP rows matching the UMAP coordinates, via CLAMP's metadata.

        ``CLAMP/valid_metadata.csv`` records which reference molecule each CLAMP
        embedding row came from; the UMAP coordinates follow that same row order.
        """
        metadata_path = os.path.join(self.dir_path, "CLAMP", "valid_metadata.csv")
        if not os.path.exists(metadata_path):
            logger.warning("Metadata file not found. Using raw feature matrix as-is.")
            return X

        logger.info("Metadata found — aligning ECFP with UMAP coordinates...")
        valid_indices = pd.read_csv(metadata_path)["original_index"].values
        X = X[valid_indices]  # fancy-index → new copy
        logger.success(f"Alignment complete: {X.shape[0]:,} samples.")
        return X
