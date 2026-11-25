"""
Verified reuse of featurizer outputs.

A fit can be re-run into an existing ``--reference`` directory — to resume after a crash, or
after changing a later step — and the featurizers should not recompute matrices
that are already on disk. Reusing a matrix is only safe if it was computed from
the same library with the same settings: a stale ``X.npy`` from a different
library either fails later as a dimension mismatch or, if the sizes happen to
agree, silently maps the wrong molecules onto the reference coordinates.

Every cached output therefore carries a key — a hash of the validated SMILES list
plus the featurizer's parameters — in a ``cache.json`` next to it. The key is
written last, so its presence means the output it describes is complete. An
output without a key (a partial write, or a run from an older version) is never
trusted.
"""

import hashlib
import json
import os
import shutil

from .logger import get_logger

logger = get_logger(__name__)

KEY_FILENAME = "cache.json"


def library_signature(smiles_list):
    """
    Return a stable hash of an ordered SMILES list.

    Parameters
    ----------
    smiles_list : list of str
        The validated reference SMILES, in pipeline order.

    Returns
    -------
    str
        Hex SHA-256 digest. Changes if any SMILES or their order changes.
    """
    digest = hashlib.sha256()
    for smi in smiles_list:
        digest.update(smi.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def cache_key(smiles_list, **params):
    """
    Build the key identifying one featurizer output.

    Parameters
    ----------
    smiles_list : list of str
        The library the output is computed from.
    **params
        Every setting that changes the output (radius, model id, batch size, ...).
        Values must be JSON-serialisable.

    Returns
    -------
    dict
        The key, as stored in ``cache.json``.
    """
    key = {
        "library_sha256": library_signature(smiles_list),
        "n_molecules": len(smiles_list),
    }
    key.update(params)
    # Round-trip through JSON so that tuples and lists compare equal to what is
    # read back from disk.
    return json.loads(json.dumps(key))


def read_key(dir_path):
    """
    Return the key stored in ``dir_path``, or None if there is none.

    Parameters
    ----------
    dir_path : str
        Directory holding the cached output.

    Returns
    -------
    dict or None
        The stored key, or None when it is missing or unreadable.
    """
    path = os.path.join(dir_path, KEY_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read cache key {path}: {e}")
        return None


def write_key(dir_path, key):
    """
    Store ``key`` in ``dir_path``. Call only after the output is fully written.

    Parameters
    ----------
    dir_path : str
        Directory holding the cached output.
    key : dict
        The key built by :func:`cache_key`.
    """
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, KEY_FILENAME), "w") as f:
        json.dump(key, f, indent=2)


def mismatch_reason(dir_path, key):
    """
    Explain why the output in ``dir_path`` cannot be reused, or return None if it can.

    Parameters
    ----------
    dir_path : str
        Directory holding the cached output.
    key : dict
        The key the current run expects.

    Returns
    -------
    str or None
        A short human-readable reason, or None when the stored key matches.
    """
    stored = read_key(dir_path)
    if stored is None:
        return "no cache key (cannot verify which library it came from)"
    if stored.get("library_sha256") != key.get("library_sha256"):
        return "the input library changed"
    changed = sorted(k for k in key if stored.get(k) != key[k])
    if changed:
        return f"settings changed ({', '.join(changed)})"
    return None


def matches(dir_path, key):
    """
    Return True if the output in ``dir_path`` was produced under ``key``.

    Parameters
    ----------
    dir_path : str
        Directory holding the cached output.
    key : dict
        The key the current run expects.

    Returns
    -------
    bool
        False when the key is missing, unreadable or different.
    """
    return mismatch_reason(dir_path, key) is None


def invalidate(paths, label, reason):
    """
    Remove stale cached outputs, logging once why they were discarded.

    Parameters
    ----------
    paths : iterable of str
        Files or directories to remove. Missing paths are ignored.
    label : str
        What is being discarded, for the log message (e.g. ``"CheMeleon batches"``).
    reason : str
        Why, typically from :func:`mismatch_reason`.
    """
    removed = False
    for path in paths:
        if os.path.isdir(path):
            shutil.rmtree(path)
            removed = True
        elif os.path.exists(path):
            os.remove(path)
            removed = True
    if removed:
        logger.warning(f"Discarding cached {label}: {reason}. Recomputing.")
