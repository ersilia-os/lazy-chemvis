import os

import numpy as np
import pytest

from lazychemvis.featurizers.ecfp import ECFPFeaturizer
from lazychemvis.featurizers.rdkit_descriptor import RDKitDescriptor


def _fit_and_save(cls, dir_path, smiles, reuse=True):
    featurizer = cls(dir_path=dir_path)
    featurizer.fit(smiles, reuse=reuse)
    featurizer.save()
    return featurizer


@pytest.mark.parametrize("cls", [RDKitDescriptor, ECFPFeaturizer])
def test_same_library_is_reused(tmp_path, library_a, cls):
    first = _fit_and_save(cls, str(tmp_path), library_a)
    assert not first._from_cache

    # Regression: RDKitDescriptor used to crash here with AttributeError, after
    # having already deleted its saved imputer, filter and scaler.
    second = _fit_and_save(cls, str(tmp_path), library_a)
    assert second._from_cache
    np.testing.assert_array_equal(first.X, second.X)


def test_rdkit_cache_hit_keeps_preprocessing(tmp_path, library_a):
    first = _fit_and_save(RDKitDescriptor, str(tmp_path), library_a)
    expected = first.transform(library_a[:5])

    _fit_and_save(RDKitDescriptor, str(tmp_path), library_a)
    for name in ("imputer.pkl", "feature_filter.pkl", "scaler.pkl", "X.npy"):
        assert os.path.exists(os.path.join(tmp_path, "rdkit_descriptor", name))

    reloaded = RDKitDescriptor.load(str(tmp_path))
    np.testing.assert_allclose(reloaded.transform(library_a[:5]), expected)


@pytest.mark.parametrize("cls", [RDKitDescriptor, ECFPFeaturizer])
def test_different_library_is_recomputed(tmp_path, library_a, library_b, cls):
    _fit_and_save(cls, str(tmp_path), library_a)
    second = _fit_and_save(cls, str(tmp_path), library_b)
    assert not second._from_cache

    third = _fit_and_save(cls, str(tmp_path), library_b)
    assert third._from_cache


@pytest.mark.parametrize("cls", [RDKitDescriptor, ECFPFeaturizer])
def test_no_reuse_recomputes(tmp_path, library_a, cls):
    _fit_and_save(cls, str(tmp_path), library_a)
    second = _fit_and_save(cls, str(tmp_path), library_a, reuse=False)
    assert not second._from_cache


def test_output_without_key_is_not_trusted(tmp_path, library_a):
    _fit_and_save(ECFPFeaturizer, str(tmp_path), library_a)
    os.remove(os.path.join(tmp_path, "ecfp", "cache.json"))

    second = _fit_and_save(ECFPFeaturizer, str(tmp_path), library_a)
    assert not second._from_cache
