from .base import XGBSurrogate


class TSNESurrogate(XGBSurrogate):
    """
    Shortcut surrogate for t-SNE: maps ECFP bits directly to CheMeleon-t-SNE
    coordinates, bypassing the need for the CheMeleon model during inference.

    See :class:`~lazychemvis.surrogates.base.XGBSurrogate` for the training procedure.
    """

    surrogate_name = "tsne_surrogate"
    projection_name = "tsne"
    label = "t-SNE"

    def _load_targets(self):
        """Return the t-SNE reference coordinates and their axis scaler."""
        # Imported here so that importing this module does not pull in openTSNE.
        from ..projectors.tsne_projector import TSNEProjector

        projector = TSNEProjector.load(dir_path=self.dir_path)
        return projector.X, projector.scaler
