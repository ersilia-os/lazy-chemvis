from lazychemvis.helpers.environment import package_versions, peak_rss_gb
from lazychemvis.helpers.tracker import PipelineTracker


def test_package_versions_cover_results_relevant_packages():
    versions = package_versions(fit=True)
    for key in ("lazychemvis", "numpy", "scikit-learn", "fpsim2", "rdkit", "python"):
        assert versions.get(key), key
    # Exactly one XGBoost build is installed; whichever it is must be recorded.
    assert "xgboost" in versions or "xgboost-cpu" in versions


def test_peak_memory_is_reported_in_gb():
    peak = peak_rss_gb()
    # A Python process with numpy and RDKit loaded: tens of MB, far below 1 TB.
    assert 0.01 < peak["main"] < 1000


def test_tracker_records_peak_memory_per_step():
    tracker = PipelineTracker()
    tracker.start("pca")
    tracker.complete("pca")
    assert tracker.peak_memory["pca"]["main"] > 0
