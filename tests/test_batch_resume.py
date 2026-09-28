"""
Batch resume for the Ersilia featurizers, with the model replaced by a stub.

No Docker or Ersilia model is needed: the featurizer's lazy ``model`` property is
bypassed by setting ``_model_instance`` directly.
"""

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ersilia")

from lazychemvis.featurizers.chemeleon import CheMeleonFeaturizer  # noqa: E402
from lazychemvis.featurizers.clamp import CLAMPFeaturizer  # noqa: E402


class FakeModel(object):
    """Returns a deterministic 4-dim embedding per SMILES and counts calls."""

    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def run(self, smiles):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise KeyboardInterrupt("simulated crash")
        values = np.array([[len(s), s.count("c"), s.count("O"), s.count("N")] for s in smiles],
                          dtype=float)
        df = pd.DataFrame(values, columns=["f0", "f1", "f2", "f3"])
        df.insert(0, "input", smiles)
        return df

    def close(self):
        pass


def _featurizer(cls, dir_path, model, batch_size=5):
    featurizer = cls(dir_path=dir_path)
    featurizer.batch_size = batch_size
    featurizer._model_instance = model
    return featurizer


@pytest.mark.parametrize("cls", [CheMeleonFeaturizer, CLAMPFeaturizer])
def test_interrupted_run_resumes_from_cached_batches(tmp_path, library_a, cls):
    crashing = FakeModel(fail_on_call=3)
    with pytest.raises(KeyboardInterrupt):
        _featurizer(cls, str(tmp_path), crashing).fit(library_a)

    model = FakeModel()
    featurizer = _featurizer(cls, str(tmp_path), model)
    featurizer.fit(library_a)
    # 20 molecules in batches of 5 = 4 batches; 2 completed before the crash.
    assert model.calls == 2
    assert featurizer.X.shape == (len(library_a), 4)
    np.testing.assert_array_equal(featurizer.X[:, 0], [len(s) for s in library_a])


@pytest.mark.parametrize("cls", [CheMeleonFeaturizer, CLAMPFeaturizer])
def test_batches_from_another_library_are_discarded(tmp_path, library_a, library_b, cls):
    with pytest.raises(KeyboardInterrupt):
        _featurizer(cls, str(tmp_path), FakeModel(fail_on_call=3)).fit(library_a)

    model = FakeModel()
    featurizer = _featurizer(cls, str(tmp_path), model)
    featurizer.fit(library_b)
    assert model.calls == 4
    np.testing.assert_array_equal(featurizer.X[:, 0], [len(s) for s in library_b])


@pytest.mark.parametrize("cls", [CheMeleonFeaturizer, CLAMPFeaturizer])
def test_finished_matrix_is_reused_only_for_same_library(tmp_path, library_a, library_b, cls):
    first = _featurizer(cls, str(tmp_path), FakeModel())
    first.fit(library_a)
    first.save()
    assert os.path.exists(os.path.join(tmp_path, first.featurizer_name, "cache.json"))

    model = FakeModel()
    again = _featurizer(cls, str(tmp_path), model)
    again.fit(library_a)
    assert again._from_cache and model.calls == 0

    model = FakeModel()
    other = _featurizer(cls, str(tmp_path), model)
    other.fit(library_b)
    assert not other._from_cache and model.calls == 4

    model = FakeModel()
    forced = _featurizer(cls, str(tmp_path), model)
    forced.fit(library_b, reuse=False)
    assert not forced._from_cache and model.calls == 4
