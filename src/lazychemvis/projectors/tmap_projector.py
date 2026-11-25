import os
import re
import subprocess

import numpy as np

from ..helpers.cache import invalidate, mismatch_reason, read_key, write_key
from ..helpers.logger import get_logger

logger = get_logger(__name__)

#: Names the graph construction in the cache key, so a layout produced by a
#: different algorithm (e.g. the pre-MinHash driver) is never reused.
LAYOUT_ALGORITHM = "minhash-lshforest"

# Matches the driver script's "Processed 40000/460000 molecules (8.7%)" progress lines.
_PCT_RE = re.compile(r"\((\d+(?:\.\d+)?)%\)")


def _parse_progress_pct(line: str):
    """Extract a percentage from a TMAP progress line, or None if it carries none."""
    match = _PCT_RE.search(line)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:  # pragma: no cover - defensive
        return None


# Layout defaults, duplicated from tmap_base so that this module (which runs in the
# main environment) does not need to import the TMAP-only driver script.
DEFAULT_K = 100
DEFAULT_KC = 1000
LOW_MEMORY_DEFAULT_K = 40
LOW_MEMORY_DEFAULT_KC = 10
DEFAULT_N_TREES = 8

_INSTALL_HINT = (
    "Create the TMAP environment with:\n"
    '    conda create -n tmap-env -c tmap -c conda-forge python=3.9 "tmap=1.0.6" numpy -y\n'
    "then pass its full path, e.g. --tmap-env /home/user/anaconda3/envs/tmap-env "
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
    ``--tmap-env`` is only discovered after the reference descriptors have already
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
            f"--tmap-env must be a path to a conda environment directory, not a "
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
            capture_output=True,
            text=True,
            timeout=timeout,
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

    def __init__(
        self,
        dir_path: str,
        k: int = None,
        kc: int = None,
        num_threads: int = 4,
        low_memory: bool = False,
        n_permutations: int = 128,
        n_trees: int = DEFAULT_N_TREES,
        batch_size: int = 10000,
    ):
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
            Number of MinHash permutations used to sign each fingerprint
            (capped at 64 in low_memory mode)
        n_trees : int, default=8
            Number of prefix trees in the LSH forest; must divide n_permutations
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
        self.n_trees = n_trees
        self.batch_size = batch_size
        self._from_cache = False

    def _cache_key(self):
        """
        Key identifying the layout this projector would produce, or None.

        The layout is a function of the ECFP matrix and the layout settings only, so
        the key is the ECFP featurizer's own key (library hash, radius, bit length,
        RDKit version) plus those settings. Batch size is left out: it changes how
        the forest is filled, not what it contains. Returns None when the
        fingerprints carry no key, since their origin then cannot be verified.
        """
        ecfp_key = read_key(os.path.join(self.dir_path, "ecfp"))
        if ecfp_key is None:
            return None
        key = dict(ecfp_key)
        key.pop("featurizer", None)
        key.update(
            projector=self.projector_name,
            algorithm=LAYOUT_ALGORITHM,
            k=self.k,
            kc=self.kc,
            n_permutations=min(self.n_permutations, 64)
            if self.low_memory
            else self.n_permutations,
            n_trees=self.n_trees,
            low_memory=self.low_memory,
        )
        return key

    def _reusable(self, output_dir, key):
        """Return None if the layout on disk can be reused, else the reason why not."""
        if key is None:
            return "the ECFP fingerprints have no cache key"
        reason = mismatch_reason(output_dir, key)
        if reason is not None:
            return reason
        reduced = os.path.join(output_dir, "reduced.npy")
        if not (
            os.path.exists(reduced)
            and os.path.exists(os.path.join(output_dir, "edges.npz"))
        ):
            return "the layout files are missing"
        n_rows = np.load(reduced, mmap_mode="r").shape[0]
        if n_rows != key["n_molecules"]:
            return f"reduced.npy has {n_rows:,} rows, expected {key['n_molecules']:,}"
        return None

    def fit(self, tmap_env: str = "tmap-env", reuse: bool = True):
        """
        Execute the TMAP projection using the optimized memory-efficient version.

        Parameters
        ----------
        tmap_env : str
            Path to the TMAP conda environment
        reuse : bool, default=True
            If True and ``tmap/`` already holds a layout computed from the same
            ECFP fingerprints with the same settings, keep it instead of running
            the layout again. On a large library the layout takes hours, so this is
            what makes resuming a fit that failed after the TMAP step cheap.
        """
        # 1. Define paths
        input_path = os.path.join(self.dir_path, "ecfp", "X.npy")
        output_dir = os.path.join(self.dir_path, self.projector_name)
        self._from_cache = False

        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"ECFP fingerprints not found at {input_path}. "
                f"Run the ECFP featurizer before the TMAP projection."
            )

        key = self._cache_key()
        reason = self._reusable(output_dir, key) if reuse else "--no-cache was given"
        if reason is None:
            logger.info(f"Reusing cached TMAP layout at {output_dir}")
            self._from_cache = True
            return
        # Drop the old key before the layout files are overwritten, so a run killed
        # mid-layout cannot leave a key vouching for half-written coordinates.
        invalidate(
            [
                os.path.join(output_dir, name)
                for name in ("cache.json", "cost.json", "reduced.npy", "edges.npz")
            ],
            "TMAP layout",
            reason,
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 2. Locate the companion script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(current_dir, "tmap_base.py")

        # 3. Verify the environment and get its interpreter
        python_exe = verify_tmap_env(tmap_env)

        # 4. Construct the command as a LIST
        # -u: unbuffered output. Writing into a pipe, the child would otherwise
        # block-buffer its prints, and no progress line would reach the log or the
        # progress bar until the layout had finished.
        cmd = [
            python_exe,
            "-u",
            script_path,
            "--input",
            input_path,
            "--output_dir",
            output_dir,
            "--n_permutations",
            str(self.n_permutations),
            "--n_trees",
            str(self.n_trees),
            "--batch_size",
            str(self.batch_size),
            "--k",
            str(self.k),
            "--kc",
            str(self.kc),
        ]

        # Add low-memory flag if enabled
        if self.low_memory:
            cmd.append("--low_memory")
            logger.warning(
                "TMAP low-memory mode enabled — fewer permutations and reduced quality settings."
            )

        logger.info(f"Running TMAP: {' '.join(cmd)}")

        # 5. Execute, streaming the child's output.
        #
        # The driver script prints ~30 progress lines, and on a large library the layout
        # stage runs for a long time. Capturing the output wholesale (capture_output=True)
        # and logging it only at the end means the user watches a frozen screen for the
        # slowest step of the fit, so read it line by line instead: every line goes to the
        # log, and the "[TMAP] Processed n/m" lines are surfaced as live progress.
        tail = self._run_streaming(cmd)

        if not os.path.exists(os.path.join(output_dir, "reduced.npy")):
            raise RuntimeError(
                "The TMAP subprocess reported success but wrote no reduced.npy.\n"
                f"Last output:\n{tail}"
            )
        # Written last: its presence means the layout files are complete.
        if key is not None:
            write_key(output_dir, key)
        logger.success("TMAP projection complete.")

    def _run_streaming(self, cmd, tail_lines: int = 40):
        """
        Run ``cmd``, forwarding each output line to the log and to a progress bar.

        Returns
        -------
        str
            The last ``tail_lines`` lines of output, so a failure message can carry
            the child's diagnostics even though nothing was captured wholesale.

        Raises
        ------
        RuntimeError
            If the subprocess exits non-zero.
        """
        from collections import deque

        from ..helpers.live import LiveProgressBar

        recent = deque(maxlen=tail_lines)
        # Total is unknown until the child reports the dataset size; start at 100 and
        # drive the bar by the percentage the child prints.
        progress = LiveProgressBar("TMAP layout", total=100, show_bar=True)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            with progress.live() as bar:
                for raw_line in proc.stdout:
                    line = raw_line.rstrip()
                    if not line:
                        continue
                    recent.append(line)
                    logger.debug(f"[tmap] {line}")

                    pct = _parse_progress_pct(line)
                    if pct is not None:
                        bar.done = min(100, int(pct))
                    note = (
                        line.replace("[TMAP LOW-MEM]", "").replace("[TMAP]", "").strip()
                    )
                    if note:
                        bar.set_note(note[:70])
                proc.wait()
                # The driver only prints a percentage every ten batches, so a small
                # library finishes without ever reporting one. Fill the bar on a clean
                # exit rather than leaving a persisted line reading "0/100".
                if proc.returncode == 0:
                    bar.done = bar.total
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.poll() is None:  # pragma: no cover - defensive
                proc.kill()
                proc.wait()

        tail = "\n".join(recent)
        if proc.returncode != 0:
            logger.error(f"TMAP failed with return code {proc.returncode}")
            logger.error(f"Output tail:\n{tail}")
            raise RuntimeError(
                f"The TMAP subprocess failed with return code {proc.returncode}.\n"
                f"Output tail:\n{tail}"
            )
        return tail

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
