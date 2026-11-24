"""
Run environment captured in ``run.json``: package versions and peak memory.

Both exist so that a fitted reference space, or a transform run, documents what
produced it and what it cost, without having to be re-timed or re-profiled.
"""

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

# Distributions whose versions change the outputs, or that the pickled artifacts
# depend on to load. Missing ones (e.g. the fit extra on a transform-only install,
# or whichever XGBoost build is not installed) are simply left out.
BASE_DISTRIBUTIONS = [
    "lazychemvis",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "xgboost-cpu",
    "fpsim2",
    "tables",
    "joblib",
]
FIT_DISTRIBUTIONS = ["opentsne", "umap-learn", "optuna", "ersilia"]


def package_versions(fit: bool = False) -> dict:
    """
    Versions of the installed packages that determine a run's results.

    Parameters
    ----------
    fit : bool, default=False
        Also include the fit-only packages.

    Returns
    -------
    dict
        Distribution name → version, plus ``python``, ``platform`` and the
        RDKit version as RDKit itself reports it.
    """
    from rdkit import rdBase

    names = BASE_DISTRIBUTIONS + (FIT_DISTRIBUTIONS if fit else [])
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    versions["rdkit"] = rdBase.rdkitVersion
    versions["python"] = platform.python_version()
    versions["platform"] = platform.platform()
    return versions


def peak_rss_gb() -> dict:
    """
    Peak resident memory so far, in GB, of this process and of its subprocesses.

    The peak never decreases, so a value read after a step is the peak of the run
    up to the end of that step. ``subprocesses`` is the peak of the largest
    finished child, which during a fit is the TMAP layout. Some libraries start
    short helper processes when imported (FPSim2's CPU probe is ~0.1 GB), so a
    small ``subprocesses`` value in a transform run is that, not real work. Memory
    used inside the Docker containers that serve the Ersilia models is not visible
    here.

    Returns
    -------
    dict
        ``{"main": float, "subprocesses": float or None}``. Values are None where
        the platform does not report them.
    """
    try:
        import resource
    except ImportError:  # Windows
        return {"main": _windows_peak_gb(), "subprocesses": None}

    # ru_maxrss is in bytes on macOS and in kibibytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    main = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return {"main": _to_gb(main), "subprocesses": _to_gb(children)}


def _windows_peak_gb():
    """Peak working set of this process on Windows, in GB."""
    try:
        import psutil

        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
    except Exception:
        return None
    return _to_gb(peak) if peak is not None else None


def _to_gb(n_bytes):
    """Bytes → GB, rounded to three decimals."""
    return round(n_bytes / 1e9, 3)
