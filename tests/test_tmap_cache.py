"""
A finished TMAP layout is reused only if it came from the same fingerprints and settings.

The layout takes hours on a large library, so a fit resumed after a later step
failed should not run it again. A cache hit returns before the TMAP environment is
looked up, so these tests pass a path that does not exist: reaching it, and the
FileNotFoundError it raises, is how a test sees that the layout would be recomputed.
"""

import os

import numpy as np
import pytest

from lazychemvis.featurizers.ecfp import ECFPFeaturizer
from lazychemvis.helpers.cache import read_key, write_key
from lazychemvis.projectors.tmap_projector import TMAPProjector

NO_ENV = "/nonexistent/tmap-env"


def _fingerprints(dir_path, smiles):
    featurizer = ECFPFeaturizer(dir_path=dir_path)
    featurizer.fit(smiles, reuse=True)
    featurizer.save()


def _fake_layout(dir_path, n_rows, **projector_args):
    """Write layout files and a key, as a finished projection would."""
    out = os.path.join(dir_path, "tmap")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "reduced.npy"), np.zeros((n_rows, 2), np.float32))
    np.savez_compressed(os.path.join(out, "edges.npz"), s=[0], t=[1])
    write_key(out, TMAPProjector(dir_path, **projector_args)._cache_key())


def test_same_fingerprints_and_settings_are_reused(tmp_path, library_a):
    _fingerprints(str(tmp_path), library_a)
    _fake_layout(str(tmp_path), len(library_a))

    projector = TMAPProjector(str(tmp_path))
    projector.fit(NO_ENV)
    assert projector._from_cache


@pytest.mark.parametrize(
    "fit_args",
    [{"k": 5}, {"kc": 5}, {"n_permutations": 64}, {"low_memory": True}],
)
def test_changed_settings_recompute(tmp_path, library_a, fit_args):
    _fingerprints(str(tmp_path), library_a)
    _fake_layout(str(tmp_path), len(library_a))

    with pytest.raises(FileNotFoundError):
        TMAPProjector(str(tmp_path), **fit_args).fit(NO_ENV)
    # The stale key goes before the layout is rerun, so a crash cannot leave it
    # vouching for coordinates it did not produce.
    assert read_key(str(tmp_path / "tmap")) is None
    assert not os.path.exists(tmp_path / "tmap" / "reduced.npy")


def test_new_library_recomputes(tmp_path, library_a, library_b):
    _fingerprints(str(tmp_path), library_a)
    _fake_layout(str(tmp_path), len(library_a))
    # Same size, different order: only the library hash tells them apart.
    _fingerprints(str(tmp_path), library_b)

    with pytest.raises(FileNotFoundError):
        TMAPProjector(str(tmp_path)).fit(NO_ENV)


def test_no_cache_recomputes(tmp_path, library_a):
    _fingerprints(str(tmp_path), library_a)
    _fake_layout(str(tmp_path), len(library_a))

    with pytest.raises(FileNotFoundError):
        TMAPProjector(str(tmp_path)).fit(NO_ENV, reuse=False)


def test_layout_without_key_is_not_trusted(tmp_path, library_a):
    _fingerprints(str(tmp_path), library_a)
    _fake_layout(str(tmp_path), len(library_a))
    os.remove(tmp_path / "tmap" / "cache.json")

    with pytest.raises(FileNotFoundError):
        TMAPProjector(str(tmp_path)).fit(NO_ENV)


def test_wrong_row_count_is_not_trusted(tmp_path, library_a):
    _fingerprints(str(tmp_path), library_a)
    _fake_layout(str(tmp_path), len(library_a) - 1)

    with pytest.raises(FileNotFoundError):
        TMAPProjector(str(tmp_path)).fit(NO_ENV)
