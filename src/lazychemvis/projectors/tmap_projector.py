import os
import numpy as np

from ..helpers.logger import get_logger

logger = get_logger(__name__)


class TMAPProjector(object):
    """
    Perform TMAP projection on ECFP fingerprint features and scale the output.
    """

    def __init__(self, dir_path: str, k: int = 30, kc: int = 10, num_threads: int = 4, 
                 low_memory: bool = False, n_permutations: int = 128, batch_size: int = 10000):
        """
        Create a TMAPProjector.
        
        Parameters
        ----------
        dir_path : str
            Directory where the projector will save results
        k : int
            Number of nearest neighbors (not used in current implementation)
        kc : int
            Number of nearest neighbors for layout (not used in current implementation)
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

        self.k = k
        self.kc = kc
        self.num_threads = num_threads
        self.low_memory = low_memory
        self.n_permutations = n_permutations
        self.batch_size = batch_size

    def fit(self, tmap_env: str = "tmap-env"):
        """
        Execute the TMAP projection using the optimized memory-efficient version.
        
        Parameters
        ----------
        tmap_env : str
            Path to the TMAP conda environment (can be relative or absolute)
        """
        # 1. Define paths
        input_path = os.path.join(self.dir_path, "ecfp", "X.npy")
        output_dir = os.path.join(self.dir_path, self.projector_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 2. Locate the companion script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "tmap_base.py")

        # 3. Path to the environment-specific python
        python_exe = f"{tmap_env}/bin/python3"

        # 4. Construct the command as a LIST
        cmd = [
            python_exe,
            script_path,
            "--input", input_path,
            "--output_dir", output_dir,
            "--n_permutations", str(self.n_permutations),
            "--batch_size", str(self.batch_size)
        ]
        
        # Add low-memory flag if enabled
        if self.low_memory:
            cmd.append("--low_memory")
            logger.warning(
                "TMAP low-memory mode enabled — fewer permutations and reduced quality settings."
            )

        logger.info(f"Running TMAP: {' '.join(cmd)}")

        # 5. Execute the command
        import subprocess
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
            raise

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