"""
TMAP must lay out a Jaccard k-NN graph, not a Hamming one.

``tm.LSHForest`` expects MinHash signatures. Fed raw bit vectors instead, its
distance silently becomes the fraction of differing bits, which on sparse ECFP is
close to zero for every pair. These tests run the driver inside the TMAP conda
environment, and are skipped when it is not available: set ``LAZYCHEMVIS_TMAP_ENV``
to its path, or keep it at ``~/anaconda3/envs/tmap-env``.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lazychemvis.featurizers.ecfp import ECFPFeaturizer
from lazychemvis.projectors.tmap_projector import TMAPProjector, resolve_tmap_python

TMAP_BASE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lazychemvis"
    / "projectors"
    / "tmap_base.py"
)


def _find_tmap_env():
    candidates = [
        os.environ.get("LAZYCHEMVIS_TMAP_ENV"),
        os.path.expanduser("~/anaconda3/envs/tmap-env"),
        os.path.expanduser("~/miniconda3/envs/tmap-env"),
    ]
    for env in candidates:
        if env and os.path.isfile(resolve_tmap_python(env)):
            return env
    return None


TMAP_ENV = _find_tmap_env()
pytestmark = pytest.mark.skipif(TMAP_ENV is None, reason="TMAP environment not found")


def _run_in_tmap_env(code, *args):
    """Run ``code`` in the TMAP interpreter with ``tmap_base`` importable."""
    env = dict(os.environ, PYTHONPATH=str(TMAP_BASE.parent))
    result = subprocess.run(
        [resolve_tmap_python(TMAP_ENV), "-c", code, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _fingerprints(tmp_path, smiles):
    featurizer = ECFPFeaturizer(dir_path=str(tmp_path))
    featurizer.fit(smiles, reuse=False)
    featurizer.save()
    return featurizer.X, str(tmp_path / "ecfp" / "X.npy")


def test_forest_distance_estimates_jaccard(tmp_path, library_a):
    X, x_path = _fingerprints(tmp_path, library_a)

    out = _run_in_tmap_env(
        "import sys, json, tmap_base\n"
        "lf, n = tmap_base.build_lsh_forest(sys.argv[1], 512, tag='')\n"
        "print(json.dumps([[lf.get_distance_by_id(i, j) for j in range(n)]"
        " for i in range(n)]))",
        x_path,
    )
    estimated = np.array(json.loads(out.strip().splitlines()[-1]))

    bits = X.astype(bool)
    inter = (bits[:, None, :] & bits[None, :, :]).sum(-1)
    union = (bits[:, None, :] | bits[None, :, :]).sum(-1)
    jaccard_distance = 1 - inter / union
    hamming_distance = (bits[:, None, :] != bits[None, :, :]).mean(-1)

    off_diag = ~np.eye(len(library_a), dtype=bool)
    err = np.abs(estimated - jaccard_distance)[off_diag]
    # 512 permutations: the MinHash estimate has a standard error of ~0.02.
    assert err.mean() < 0.03
    # The Hamming regression would put every pair within a few percent of 0.
    assert np.abs(estimated - hamming_distance)[off_diag].mean() > 0.3


def test_projector_writes_scaled_coordinates(tmp_path, library_a):
    _fingerprints(tmp_path, library_a)

    TMAPProjector(dir_path=str(tmp_path), k=5, kc=10).fit(TMAP_ENV)

    coords = np.load(tmp_path / "tmap" / "reduced.npy")
    assert coords.shape == (len(library_a), 2)
    assert np.isclose(coords.min(axis=0), -1).all()
    assert np.isclose(coords.max(axis=0), 1).all()


def test_layout_is_reproducible(tmp_path, library_a):
    # MinHash is seeded and the layout itself has no random step, so refitting
    # the same fingerprints must give the same map, bit for bit.
    _fingerprints(tmp_path, library_a)
    runs = []
    for _ in range(2):
        TMAPProjector(dir_path=str(tmp_path), k=5, kc=10).fit(TMAP_ENV)
        runs.append(np.load(tmp_path / "tmap" / "reduced.npy"))
    np.testing.assert_array_equal(runs[0], runs[1])


def test_finished_layout_is_reused(tmp_path, library_a):
    _fingerprints(tmp_path, library_a)
    first = TMAPProjector(dir_path=str(tmp_path), k=5, kc=10)
    first.fit(TMAP_ENV)
    assert not first._from_cache
    reduced = tmp_path / "tmap" / "reduced.npy"
    written = reduced.stat().st_mtime_ns

    second = TMAPProjector(dir_path=str(tmp_path), k=5, kc=10)
    second.fit(TMAP_ENV)
    assert second._from_cache
    assert reduced.stat().st_mtime_ns == written


def test_non_binary_input_is_rejected(tmp_path):
    x_path = tmp_path / "X.npy"
    np.save(x_path, np.array([[0, 2, 1, 0]] * 4, dtype=np.int8))

    out = _run_in_tmap_env(
        "import sys, tmap_base\n"
        "try:\n"
        "    tmap_base.build_lsh_forest(sys.argv[1], 8, n_trees=2, tag='')\n"
        "except ValueError as e:\n"
        "    print('REJECTED', e)\n",
        str(x_path),
    )
    assert "REJECTED" in out and "binary" in out
