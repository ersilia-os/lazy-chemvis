"""
Report orchestration.

Each reporter renders a declared list of figures, skipping any whose inputs are
missing, and then assembles the HTML. Figures are rendered serially: matplotlib is
not thread-safe and the landscape cache makes the repeated work cheap anyway.

Passing ``plot_name`` re-renders a single figure, which is the fast path when
iterating on a figure's styling without redoing a whole report.
"""

import os

from ..helpers.console import echo
from ..helpers.logger import get_logger
from . import REPORT_SUBFOLDER
from . import plots as _plots
from .fetcher import PROJECTIONS, ResultsFetcher
from .html import write_html_report

logger = get_logger(__name__)


class _BaseReporter(object):
    """Shared rendering loop."""

    mode = "fit"

    def __init__(self, path, make_plots=True, plot_name=None):
        """
        Parameters
        ----------
        path : str
            Run directory. The report is written to ``<path>/report``.
        make_plots : bool, default=True
            If False, only the HTML is assembled from figures already on disk.
        plot_name : str, optional
            Render only the figure with this stem.
        """
        self.path = os.path.abspath(path)
        self.make_plots = make_plots
        self.plot_name = plot_name
        self.fetcher = self._build_fetcher()

    def _build_fetcher(self):
        return ResultsFetcher(self.path)

    @property
    def report_dir(self):
        return os.path.join(self.path, REPORT_SUBFOLDER)

    def _plot_jobs(self):
        """Return a list of zero-argument callables, each producing one figure."""
        raise NotImplementedError

    def run(self):
        """Render the figures and write the HTML. Returns the report path."""
        os.makedirs(self.report_dir, exist_ok=True)

        if self.make_plots:
            jobs = self._plot_jobs()
            if self.plot_name:
                jobs = [(name, fn) for name, fn in jobs if name == self.plot_name]
                if not jobs:
                    logger.warning(
                        f"No figure named '{self.plot_name}' in this report."
                    )
            self._render(jobs)

        path = write_html_report(self.path, mode=self.mode, fetcher=self.fetcher)
        echo(f"Report: {path}", kind="success")
        return path

    def _render(self, jobs):
        """Render each job, logging and skipping any that fail."""
        rendered, skipped, failed = 0, 0, 0
        for name, fn in jobs:
            try:
                plot = fn()
                if plot is None:
                    skipped += 1
                    continue
                if plot.save() is None:
                    skipped += 1
                else:
                    rendered += 1
            except Exception as e:
                # One bad figure must not cost the whole report.
                failed += 1
                logger.exception(f"Figure '{name}' failed: {e}")
        logger.info(
            f"Report figures — {rendered} rendered, {skipped} skipped "
            f"(inputs missing), {failed} failed."
        )
        if failed:
            echo(
                f"{failed} report figure(s) failed — see the log for details.",
                kind="warning",
            )


class FitReporter(_BaseReporter):
    """
    Report for a fitted reference space.

    Covers the run configuration, the four reference landscapes, cross-validated
    surrogate quality, and step timings.
    """

    mode = "fit"

    def _plot_jobs(self):
        f = self.fetcher
        jobs = []

        for projection in PROJECTIONS:
            if not f.has_projection(projection):
                continue
            jobs.append(
                (
                    f"{projection}_reference_space",
                    lambda p=projection: _plots.ReferenceLandscapePlot(
                        projection_name=p, path=self.path, fetcher=f
                    ),
                )
            )

        jobs.append(
            (
                "cv_metric_bars",
                lambda: _plots.CvMetricBarsPlot(path=self.path, fetcher=f),
            )
        )
        jobs.append(
            (
                "euclidean_error_histogram",
                lambda: _plots.EuclideanErrorHistogramPlot(path=self.path, fetcher=f),
            )
        )
        jobs.append(
            ("step_timing", lambda: _plots.StepTimingPlot(path=self.path, fetcher=f))
        )
        return jobs


class TransformReporter(_BaseReporter):
    """
    Report for a projection run.

    Covers the input summary (including molecules TMAP could not place), the
    overlay figures, coordinate coverage against the reference, and a preview of
    ``coordinates.csv``.

    Parameters
    ----------
    path : str
        The transform output directory (``--output``).
    reference_path : str
        The fitted reference space (``--reference``), which holds the reference
        coordinates the overlays are drawn on.
    """

    mode = "transform"

    def __init__(self, path, reference_path, make_plots=True, plot_name=None):
        self.reference_path = os.path.abspath(reference_path)
        super().__init__(path, make_plots=make_plots, plot_name=plot_name)

    def _build_fetcher(self):
        return ResultsFetcher(self.path, reference_path=self.reference_path)

    def _plot_jobs(self):
        f = self.fetcher
        jobs = []
        for projection in PROJECTIONS:
            if not f.has_projection(projection):
                continue
            jobs.append(
                (
                    f"{projection}_overlay",
                    lambda p=projection: _plots.OverlayPlot(
                        projection_name=p,
                        path=self.path,
                        reference_path=self.reference_path,
                        fetcher=f,
                    ),
                )
            )
            jobs.append(
                (
                    f"{projection}_coordinate_density",
                    lambda p=projection: _plots.CoordinateDensityPlot(
                        projection_name=p,
                        path=self.path,
                        reference_path=self.reference_path,
                        fetcher=f,
                    ),
                )
            )
        return jobs
