import os
import numpy as np
#from ersilia.utils.conda import StandaloneConda

class TMAPProjector(object):
    """
    Perform TMAP projection on ECFP fingerprint features and scale the output.
    """

    def __init__(self, dir_path: str, k: int = 30, kc: int = 10, num_threads: int = 4):
        """
        Create a TMAPProjector.
        """
        self.projector_name = "tmap"
        self.dir_path = os.path.abspath(dir_path)
        
        # Ensure the base directory exists
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        self.k = k
        self.kc = kc
        self.num_threads = num_threads

    def fit(self, tmap_env: str = "tmap-env"):
        """
        Execute the TMAP projection using a list of arguments for subprocess.
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
            "--output_dir", output_dir
        ]
        
        print(f"--- Running TMAP Command: {' '.join(cmd)} ---")

        # 5. Execute the command
        import subprocess
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            print("--- TMAP Projection Complete ---")
        except subprocess.CalledProcessError as e:
            print(f"TMAP failed with return code {e.returncode}")
            print(f"Error output: {e.stderr}")

    @classmethod
    def load(cls, dir_path: str):
        """
        Load the results of a previous projection.
        """
        # Logic to return coordinates or state if needed
        output_path = os.path.join(dir_path, "tmap", "reduced.npy")
        if os.path.exists(output_path):
            return np.load(output_path)
        return None