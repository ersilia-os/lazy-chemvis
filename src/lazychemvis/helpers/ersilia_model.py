"""
Serving of Ersilia Model Hub models.

Ersilia's own handling of a model that is not present on the machine is an
interactive ``[Y/n]`` prompt that defaults to *no* when stdin cannot be read.
Featurization runs inside a Rich spinner, where that prompt is neither readable
nor answerable, so the pipeline used to die with "Model is not fetched" instead
of downloading the model. Fetching explicitly up front keeps the whole run
non-interactive.
"""

from ersilia import ModelBase
from ersilia.api import Model

from .logger import get_logger, console

logger = get_logger(__name__)


def is_available_locally(model_id: str) -> bool:
    """
    Check whether an Ersilia model is already present on this machine.

    Parameters
    ----------
    model_id : str
        Ersilia model identifier, e.g. ``eos9o72``.

    Returns
    -------
    bool
        True if the model has been fetched (from any source), False otherwise.
    """
    return ModelBase(model_id_or_slug=model_id).is_available_locally()


def serve_model(model_id: str) -> Model:
    """
    Return a served Ersilia model, fetching it first if it is not available.

    Parameters
    ----------
    model_id : str
        Ersilia model identifier, e.g. ``eos9o72``.

    Returns
    -------
    ersilia.api.Model
        A model instance that has already been served and is ready for ``run``.

    Raises
    ------
    RuntimeError
        If the model could not be fetched. The most common cause is Docker not
        running, since models are pulled from DockerHub by default.
    """
    model = Model(model_id=model_id)

    if not is_available_locally(model_id):
        logger.info(f"Model {model_id} is not available locally. Fetching it...")
        console.print(
            f"  Model {model_id} is not available locally. Fetching it, "
            "this may take a few minutes..."
        )
        # The return value is not checked: it is a FetchResult on recent Ersilia
        # versions but None on older ones. Re-reading the local status is the
        # version-independent way of knowing whether the fetch worked.
        model.fetch()

        if not is_available_locally(model_id):
            raise RuntimeError(
                f"Could not fetch Ersilia model {model_id}.\n"
                "Models are pulled from DockerHub by default, so make sure "
                "Docker is installed and running, then re-run. You can also "
                "fetch it manually with: ersilia fetch " + model_id
            )
        logger.info(f"Model {model_id} fetched successfully.")

    logger.info(f"Serving model: {model_id}")
    model.serve()
    return model
