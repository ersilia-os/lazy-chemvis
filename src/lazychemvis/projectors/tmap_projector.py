import os
import subprocess
import numpy as np

from ..helpers.logger import get_logger

logger = get_logger(__name__)

# Layout defaults, duplicated from tmap_base so that this module (which runs in the
# main environment) does not need to import the TMAP-only driver script.
DEFAULT_K = 100
DEFAULT_KC = 1000
LOW_MEMORY_DEFAULT_K = 40
LOW_MEMORY_DEFAULT_KC = 10

_INSTALL_HINT = (
    "Create the TMAP environment with:\n"
    '    conda create -n tmap-env -c tmap -c conda-forge python=3.9 "tmap=1.0.6" numpy -y\n'
    "then pass its full path, e.g. --tmap_env /home/user/anaconda3/envs/tmap-env "
    "(conda env list shows the paths)."
)


def resolve_tmap_python(tmap_env: str) -> str:
    """
    Return the path to the Python interpreter inside a TMAP environment.

    Parameters
    ----------
    tmap_env : str
        Path to the TMAP conda environment directory.

    Returns
    -------
    str
        Absolute path to ``<tmap_env>/bin/python3``.
    """
    return os.path.join(os.path.abspath(os.path.expanduser(tmap_env)), "bin", "python3")


def verify_tmap_env(tmap_env: str, timeout: int = 120) -> str:
    """
    Check that a usable TMAP environment exists before any expensive work starts.

    TMAP runs as a subprocess in a separate conda environment, and the projection
    step comes after featurization. Without an up-front check, a mistyped
    ``--tmap_env`` is only discovered after the reference descriptors have already
    been computed, which can be hours into a large fit.

    Parameters
    ----------
    tmap_env : str
        Path to the TMAP conda environment directory.
    timeout : int, default=120
        Seconds to wait for the ``import tmap`` probe.

    Returns
    -------
    str
        Path to the verified Python interpreter.

    Raises
    ------
    FileNotFoundError
        If the environment directory or its interpreter does not exist.
    RuntimeError
        If the interpreter exists but cannot import ``tmap``.
    """
    env_path = os.path.abspath(os.path.expanduser(tmap_env))

    if not os.path.isdir(env_path):
        raise FileNotFoundError(
            f"TMAP environment not found: {env_path}\n"
            f"--tmap_env must be a path to a conda environment directory, not a "
            f"bare environment name.\n{_INSTALL_HINT}"
        )

    python_exe = resolve_tmap_python(env_path)
    if not os.path.isfile(python_exe):
        raise FileNotFoundError(
            f"No Python interpreter at {python_exe}\n"
            f"{env_path} does not look like a conda environment.\n{_INSTALL_HINT}"
        )

    try:
        result = subprocess.run(
            [python_exe, "-c", "import tmap; print(tmap.__name__)"],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Timed out after {timeout}s probing the TMAP environment at {env_path}."
        ) from e

    if result.returncode != 0:
        raise RuntimeError(
            f"The environment at {env_path} cannot import 'tmap'.\n"
            f"{result.stderr.strip()}\n{_INSTALL_HINT}"
        )

    logger.debug(f"Verified TMAP environment: {python_exe}")
    return python_exe


class TMAPProjector(object):
    """
    Perform TMAP projection on ECFP fingerprint features and scale the output.
    """

    def __init__(self, dir_path: str, k: int = None, kc: int = None, num_threads: int = 4,
                 low_memory: bool = False, n_permutations: int = 128, batch_size: int = 10000):
        """
        Create a TMAPProjector.

        Parameters
        ----------
        dir_path : str
            Directory where the projector will save results
        k : int, optional
            Number of nearest neighbours used to build the k-NN graph. Higher
            values create more edges between distant clusters, producing a more
            connected map. Defaults to 100, or 40 in low_memory mode.
        kc : int, optional
            Node-connectivity factor for the layout. Defaults to 1000, or 10 in
            low_memory mode.
        num_threads : int
            Number of threads (not used in current implementation)
        low_memory : bool, default=False
            If True, use ultra-low memory mode for datasets > 1M molecules
        n_permutations : int, default=128
            Number of LSH permutations (reduced to 64 in low_memory mode)
        batch_size : int, default=10000
            Batch size for processing molecules
        """
        self.projector_name = "tmap"
        self.dir_path = os.path.abspath(dir_path)

        # Ensure the base directory exists
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        self.low_memory = low_memory

        # Resolve the layout parameters against the defaults for the selected mode,
        # so the hard-coded behaviour is preserved unless explicitly overridden.
        if k is None:
            k = LOW_MEMORY_DEFAULT_K if low_memory else DEFAULT_K
        if kc is None:
            kc = LOW_MEMORY_DEFAULT_KC if low_memory else DEFAULT_KC
        self.k = k
        self.kc = kc

        self.num_threads = num_threads
        self.n_permutations = n_permutations
        self.batch_size = batch_size

    def fit(self, tmap_env: str = "tmap-env"):
        """
        Execute the TMAP projection using the optimized memory-efficient version.

        Parameters
        ----------
        tmap_env : str
            Path to the TMAP conda environment
        """
        # 1. Define paths
        input_path = os.path.join(self.dir_path, "ecfp", "X.npy")
        output_dir = os.path.join(self.dir_path, self.projector_name)

        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"ECFP fingerprints not found at {input_path}. "
                f"Run the ECFP featurizer before the TMAP projection."
            )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 2. Locate the companion script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "tmap_base.py")

        # 3. Verify the environment and get its interpreter
        python_exe = verify_tmap_env(tmap_env)

        # 4. Construct the command as a LIST
        cmd = [
            python_exe,
            script_path,
            "--input", input_path,
            "--output_dir", output_dir,
            "--n_permutations", str(self.n_permutations),
            "--batch_size", str(self.batch_size),
            "--k", str(self.k),
            "--kc", str(self.kc),
        ]

        # Add low-memory flag if enabled
        if self.low_memory:
            cmd.append("--low_memory")
            logger.warning(
                "TMAP low-memory mode enabled — fewer permutations and reduced quality settings."
            )

        logger.info(f"Running TMAP: {' '.join(cmd)}")

        # 5. Execute the command
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                logger.debug(result.stdout.strip())
            if result.stderr:
                logger.debug(result.stderr.strip())
            logger.success("TMAP projection complete.")
        except subprocess.CalledProcessError as e:
            logger.error(f"TMAP failed with return code {e.returncode}")
            logger.error(f"STDOUT:\n{e.stdout}")
            logger.error(f"STDERR:\n{e.stderr}")
            raise RuntimeError(
                f"The TMAP subprocess failed with return code {e.returncode}.\n"
                f"STDERR:\n{(e.stderr or '').strip()}"
            ) from e

    @classmethod
    def load(cls, dir_path: str):
        """
        Load the results of a previous projection.
        """
        projector = cls(dir_path=dir_path)
        output_path = os.path.join(dir_path, "tmap", "reduced.npy")
        if os.path.exists(output_path):
            projector.X = np.load(output_path)
        else:
            projector.X = None
        return projector
