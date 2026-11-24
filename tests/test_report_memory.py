from lazychemvis.report.html import _performance_section

TIMINGS = {"pca": 12.0, "tmap": 30.0}


class StubFetcher(object):
    """Stands in for ResultsFetcher, serving only what the section reads."""

    def __init__(self, peak_memory):
        self._peak_memory = peak_memory

    def timings(self):
        """Per-step wall-clock times."""
        return dict(TIMINGS)

    def peak_memory(self):
        """The ``peak_memory_gb`` block of a manifest."""
        return self._peak_memory


def test_peak_memory_column_when_recorded(tmp_path):
    memory = {
        "run": {"main": 3.21, "subprocesses": 7.89},
        "after_step": {
            "pca": {"main": 1.04, "subprocesses": 0.1},
            "tmap": {"main": 3.21, "subprocesses": 7.89},
        },
    }
    html = _performance_section(str(tmp_path), {}, StubFetcher(memory))
    assert "Peak RAM (GB)" in html
    assert ">1.0<" in html and ">3.2<" in html
    assert "TMAP layout runs as a separate process and peaked at 7.9 GB" in html


def test_timings_only_for_runs_without_memory(tmp_path):
    html = _performance_section(str(tmp_path), {}, StubFetcher({}))
    assert "Seconds" in html
    assert "Peak RAM" not in html


def test_unknown_peak_is_not_reported(tmp_path):
    # A resumed run whose TMAP layout cost was reconstructed from a log.
    unknown = {"main": None, "subprocesses": None}
    memory = {
        "run": unknown,
        "after_step": {"pca": {"main": 1.04, "subprocesses": 0.1}, "tmap": unknown},
    }
    html = _performance_section(str(tmp_path), {}, StubFetcher(memory))
    assert "Peak RAM (GB)" in html and "—" in html
    assert "peaked at" not in html
