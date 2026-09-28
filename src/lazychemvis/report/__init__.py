"""
Report generation.

Figures are :class:`BasePlot` subclasses drawn with stylia and saved as both PNG
(for the HTML report) and PDF (directly usable as publication figures), into
``<path>/report/png`` and ``<path>/report/pdf``. :mod:`lazychemvis.report.html`
assembles them into a single ``report.html``.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — prevents Tk thread conflicts
import matplotlib.pyplot as plt
import stylia

from ..helpers.logger import get_logger

logger = get_logger(__name__)

#: Report subdirectory, relative to a run's output directory.
REPORT_SUBFOLDER = "report"

#: A figure's `cells` grid is expressed in sixths of the base figure width, so a
#: `cells=(3, 6)` plot spans the full width and `cells=(3, 3)` spans half of it.
#: stylia's base size is 7.09 in (print format).
CELLS_PER_WIDTH = 6

#: Raster resolution for the PNGs embedded in the HTML report.
REPORT_DPI = 200

#: Written alongside the figures; records each figure's cell footprint so the HTML
#: can size its cards to match the aspect ratio the figure was drawn at.
FIGURE_CELLS_FILENAME = "figure_cells.json"


class BasePlot(object):
    """
    Base class for every report figure.

    Subclasses set :attr:`name` (the output file stem) and pass ``cells=(rows, cols)``,
    then draw onto ``self.ax``. A subclass that cannot draw — because the artifact it
    needs is missing from the run — sets ``self.is_available = False`` and the
    orchestrator skips it rather than failing the report.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Existing axes to draw on. When None (the normal case) a new stylia figure is
        created and ``self.ax`` is bound to it.
    path : str
        A run's output directory; the report is written to ``<path>/report``.
    cells : tuple of (int, int), optional
        Figure footprint in sixths of the base width.
    panels : tuple of (int, int), optional
        Panel grid within the figure. Defaults to a single panel; when more than one
        panel is requested, :attr:`axes` holds stylia's axis manager and
        :meth:`next_ax` steps through the panels.
    figsize : tuple, optional
        Explicit size in inches, bypassing the cells arithmetic.
    """

    #: Output file stem; subclasses must set this.
    name = "plot"

    def __init__(self, ax=None, path=None, cells=None, panels=(1, 1), figsize=None):
        self.path = os.path.abspath(path) if path else None
        self.cells = cells or (3, 3)
        self.is_available = True
        self.fig = None
        self.axes = None

        if ax is not None:
            self.ax = ax
            return

        rows, cols = self.cells
        if figsize is not None:
            self.fig, self.ax = plt.subplots(*panels, figsize=figsize)
            return

        self.fig, axs = stylia.create_figure(
            panels[0],
            panels[1],
            width=cols / CELLS_PER_WIDTH,
            height=rows / CELLS_PER_WIDTH,
        )
        self.axes = axs
        self.ax = self.next_ax()

    def next_ax(self):
        """Return the next panel's axes (stylia's axis manager steps through them)."""
        if self.axes is None:
            return self.ax
        return self.axes.next() if hasattr(self.axes, "next") else self.axes

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def report_dir(self):
        """Directory the figure is written into: ``<path>/report``."""
        return os.path.join(self.path, REPORT_SUBFOLDER)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """
        Write the figure as PNG and PDF, and record its cell footprint.

        Closes the figure afterwards, so a report with dozens of figures does not
        accumulate open matplotlib figures.

        Returns
        -------
        str or None
            The PNG path, or None if the figure was unavailable.
        """
        if not self.is_available:
            logger.debug(f"Skipping unavailable figure: {self.name}")
            self.close()
            return None

        png_dir = os.path.join(self.report_dir, "png")
        pdf_dir = os.path.join(self.report_dir, "pdf")
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        png_path = os.path.join(png_dir, self.name + ".png")
        plt.savefig(png_path, dpi=REPORT_DPI, transparent=False, bbox_inches="tight")
        plt.savefig(
            os.path.join(pdf_dir, self.name + ".pdf"),
            transparent=False,
            bbox_inches="tight",
        )
        self._record_cells()
        logger.debug(f"Saved figure: {png_path}")
        self.close()
        return png_path

    def close(self):
        """Close the underlying matplotlib figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        else:
            plt.close()

    def _record_cells(self):
        """Merge this figure's footprint into report/figure_cells.json."""
        path = os.path.join(self.report_dir, FIGURE_CELLS_FILENAME)
        data = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
            except Exception:  # pragma: no cover - a corrupt file is not fatal
                data = {}
        data[self.name] = list(self.cells)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)


def load_figure_cells(report_dir):
    """Return the recorded ``{figure stem: [rows, cols]}`` mapping, or an empty dict."""
    path = os.path.join(report_dir, FIGURE_CELLS_FILENAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # pragma: no cover
        return {}
