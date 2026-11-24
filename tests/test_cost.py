"""
A resumed fit reports what the reused outputs cost when they were produced.

Without this, a fit restarted after its TMAP step would record a TMAP time of a
couple of minutes (plot and FPSim2 database) instead of the hours of the layout.
"""

import time

import pytest

from lazychemvis.fit import Pipeline
from lazychemvis.helpers.cost import (
    combine,
    max_peak,
    merge_measures,
    read_cost,
    write_cost,
)
from lazychemvis.helpers.tracker import PipelineTracker


def _peak(main, sub=None):
    return {"main": main, "subprocesses": sub}


def test_combine_adds_seconds_and_keeps_the_highest_peak():
    total = combine(
        {"seconds": 10.0, "peak_memory_gb": _peak(2.0, 5.0)},
        None,
        {"seconds": 5.0, "peak_memory_gb": _peak(3.0, None)},
    )
    assert total == {"seconds": 15.0, "peak_memory_gb": _peak(3.0, 5.0)}
    assert combine(None, None) is None
    assert max_peak(None, _peak(None, None)) == _peak(None, None)


def test_merge_adds_reused_costs_to_their_step():
    timings = {"pca": 60.0, "tmap": 90.0, "tsne": 600.0}
    after_step = {"pca": _peak(1.0, 0.1), "tmap": _peak(1.5, 0.1), "tsne": _peak(4.0)}
    carried = {
        "tmap": [
            {"output": "ecfp", "seconds": 700.0, "peak_memory_gb": _peak(2.0, 0.1)},
            {"output": "tmap", "seconds": 9600.0, "peak_memory_gb": _peak(2.0, 7.9)},
        ]
    }
    merged, merged_after, run, unknown = merge_measures(
        timings, after_step, _peak(4.0, 0.1), carried
    )
    assert unknown == []

    assert merged == {"pca": 60.0, "tmap": 10390.0, "tsne": 600.0}
    assert merged_after["pca"] == _peak(1.0, 0.1)
    assert merged_after["tmap"] == _peak(2.0, 7.9)
    # Peaks are running maxima, as they would be in one uninterrupted run.
    assert merged_after["tsne"] == _peak(4.0, 7.9)
    assert run == _peak(4.0, 7.9)


def test_cost_without_peak_makes_the_peak_unknown():
    # A cost reconstructed from a log has a time but no memory reading. Reporting
    # the maximum of what was measured would understate the run, so the step and
    # the run peak are reported as unknown instead.
    timings = {"pca": 60.0, "tmap": 90.0, "tsne": 600.0}
    after_step = {"pca": _peak(1.0, 0.1), "tmap": _peak(1.5, 0.2), "tsne": _peak(4.0)}
    carried = {"tmap": [{"output": "tmap", "seconds": 9600.0, "peak_memory_gb": None}]}
    merged, merged_after, run, unknown = merge_measures(
        timings, after_step, _peak(4.0, 0.2), carried
    )
    assert merged["tmap"] == 9690.0
    assert unknown == ["tmap"]
    assert merged_after["tmap"] == _peak(None, None)
    assert merged_after["tsne"] == _peak(4.0, 0.2)
    assert run == _peak(None, None)


class _Unit(object):
    def __init__(self, name, from_cache, prior_cost=None):
        self.featurizer_name = name
        self._from_cache = from_cache
        self.prior_cost = prior_cost


@pytest.fixture
def pipeline(tmp_path):
    p = Pipeline.__new__(Pipeline)
    p.dir_path = str(tmp_path)
    p.tracker = PipelineTracker()
    p.tracker._current = "tmap"
    p.cache_reused, p.carried, p.unmeasured = {}, {}, {}
    return p


def test_computed_output_stores_its_cost(pipeline, tmp_path):
    earlier = {"seconds": 100.0, "peak_memory_gb": _peak(0.5)}
    pipeline._record_reuse(_Unit("ecfp", False, earlier), time.time() - 2)

    stored = read_cost(str(tmp_path / "ecfp"))
    assert 102 <= stored["seconds"] < 110
    # Batches from an interrupted run are part of the step's cost.
    assert pipeline.carried == {"tmap": [dict(earlier, output="ecfp")]}


def test_reused_output_carries_its_stored_cost(pipeline, tmp_path):
    cost = {"seconds": 9600.0, "peak_memory_gb": _peak(2.0, 7.9)}
    write_cost(str(tmp_path / "ecfp"), cost)

    pipeline._record_reuse(_Unit("ecfp", True), time.time())

    assert pipeline.cache_reused == {"ecfp": True}
    assert pipeline.carried == {"tmap": [dict(cost, output="ecfp")]}
    assert read_cost(str(tmp_path / "ecfp")) == cost


def test_reused_output_without_cost_is_flagged(pipeline):
    pipeline._record_reuse(_Unit("ecfp", True), time.time())
    assert pipeline.unmeasured == {"tmap": ["ecfp"]}
    assert pipeline.carried == {}
