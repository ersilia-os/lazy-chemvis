"""
Report figures.

Every figure is a :class:`~lazychemvis.report.BasePlot` subclass drawn with stylia
and saved as PNG + PDF into the run's ``report/`` directory. Figures fall into three
groups:

* **landscapes** — the reference chemical space, and new molecules overlaid on it.
  Rendered with datashader, because a million-point matplotlib scatter is neither
  fast nor legible.
* **validation** — per-fold surrogate diagnostics, constructed by the surrogates
  during cross-validation (they need the fold split, which only exists then).
* **summary** — cross-validation metrics, error distributions and step timings,
  built from persisted artifacts after the fact.
"""

import numpy as np
import pandas as pd

import datashader as ds
import datashader.transfer_functions as tf
import stylia
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

from . import BasePlot
from .colors import grey_ramp, projection_rgb, rgb
from .fetcher import PROJECTION_LABELS, ResultsFetcher
from ..helpers.logger import get_logger

logger = get_logger(__name__)

# All projections are scaled to [-1, 1]; a hair of padding keeps points off the edge.
EXTENT = [-1.05, 1.05, -1.05, 1.05]

_CANVAS_PX = 1000

# Up to this many reference molecules the landscape is drawn as a scatter; above
# it, datashader. Datashader shades by density, which is what a large library
# needs, but on a small one almost every pixel holds a single molecule, drawn at
# its faintest shade, and the landscape disappears. A scatter keeps every molecule
# visible and stops saturating into a solid blob only above a few hundred thousand.
SCATTER_MAX_POINTS = 200_000

# The scatter is drawn onto an off-screen canvas of this physical size, so marker
# sizes (in points) are calibrated against it rather than against the report
# figure the image is later placed in.
_SCATTER_CANVAS_IN = 5.0

# stylia sets patch.linewidth = 0, which makes a histtype="step" outline invisible —
# and, when a dashed linestyle is requested, scales the dash pattern to [0, 0], which
# raises "At least one value in the dash list must be positive" at draw time. Every
# step histogram therefore has to pass an explicit linewidth.
STEP_LINEWIDTH = 1.3

# Shaded reference landscapes, keyed by (path, projection). Rendering the datashader
# aggregation is the expensive part and several figures reuse the same landscape —
# previously every call re-rendered it, so a transform re-shaded all four twice.
_LANDSCAPE_CACHE = {}


def _scatter_landscape(coords):
    """
    Draw an (n, 2) coordinate array as a scatter and return it as a PIL image.

    Rendered with matplotlib's object API onto its own canvas rather than through
    pyplot: the report saves figures with ``plt.savefig``, which acts on pyplot's
    current figure, and a pyplot figure opened here would become that figure.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    fig = Figure(figsize=(_SCATTER_CANVAS_IN, _SCATTER_CANVAS_IN),
                 dpi=_CANVAS_PX / _SCATTER_CANVAS_IN)
    # stylia enables tight layout globally; this canvas is a single full-bleed axes.
    fig.set_layout_engine("none")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    # Marker area shrinks with the number of molecules, so a 1k library shows
    # distinct dots and a 200k one a fine texture instead of a solid blob. The floor
    # keeps each dot at least about a pixel across: smaller ones are antialiased
    # into a faint wash and the landscape fades out again.
    size = float(np.clip(12_000 / max(len(coords), 1), 0.4, 15))
    # Lightened so the landscape recedes behind whatever is drawn on top of it.
    ax.scatter(coords[:, 0], coords[:, 1], s=size,
               color=rgb("reference", lighten=0.4), linewidths=0)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def _shade_landscape(coords, cmap=None):
    """Aggregate an (n, 2) coordinate array into a shaded PIL image."""
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    canvas = ds.Canvas(
        plot_width=_CANVAS_PX, plot_height=_CANVAS_PX,
        x_range=(EXTENT[0], EXTENT[1]), y_range=(EXTENT[2], EXTENT[3]),
    )
    agg = canvas.points(df, "x", "y")
    img = tf.shade(agg, cmap=cmap or grey_ramp(), how="eq_hist")
    img = tf.spread(img, px=1, shape="circle")
    img = tf.set_background(img, "white")
    return img.to_pil()


def landscape_image(path, projection, fetcher=None):
    """
    Return the shaded reference landscape for a projection, or None.

    Cached across figures within a process.
    """
    key = (path, projection)
    if key in _LANDSCAPE_CACHE:
        return _LANDSCAPE_CACHE[key]
    fetcher = fetcher or ResultsFetcher(path)
    coords = fetcher.reference_coords(projection)
    if coords is None:
        img = None
    elif len(coords) <= SCATTER_MAX_POINTS:
        img = _scatter_landscape(coords)
    else:
        img = _shade_landscape(coords)
    _LANDSCAPE_CACHE[key] = img
    return img


def _style_axes(ax, title=None):
    """Strip the axes down to a framed panel — these are maps, not plots."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.spines["bottom"].set_color("#dddddd")
    ax.set_xticks([])
    ax.set_yticks([])
    # stylia labels axes "X-axis / Units" by default, which is meaningless for a
    # projection: the coordinates are arbitrary embedding units with no scale.
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title:
        ax.set_title(title, loc="left")


# ----------------------------------------------------------------------
# Landscapes
# ----------------------------------------------------------------------

class ReferenceLandscapePlot(BasePlot):
    """The reference chemical space for one projection."""

    def __init__(self, projection_name, path, ax=None, fetcher=None):
        self.projection_name = projection_name
        self.name = f"{projection_name}_reference_space"
        super().__init__(ax=ax, path=path, cells=(3, 3))

        img = landscape_image(self.path, projection_name, fetcher)
        if img is None:
            self.is_available = False
            return

        self.ax.imshow(img, extent=EXTENT, interpolation="lanczos")
        label = PROJECTION_LABELS.get(projection_name, projection_name.upper())
        _style_axes(self.ax, f"Chemical space — {label}")


class OverlayPlot(BasePlot):
    """New molecules overlaid on the reference landscape."""

    def __init__(self, projection_name, path, reference_path=None, new_coords=None,
                 label="Input molecules", ax=None, fetcher=None):
        self.projection_name = projection_name
        self.name = f"{projection_name}_overlay"
        super().__init__(ax=ax, path=path, cells=(3, 3))

        fetcher = fetcher or ResultsFetcher(path, reference_path)
        if new_coords is None:
            new_coords = fetcher.projected_coords(projection_name)
        img = landscape_image(fetcher.reference_path, projection_name, fetcher)

        if img is None or new_coords is None:
            self.is_available = False
            return

        self.ax.imshow(img, extent=EXTENT, interpolation="lanczos", alpha=1.0)

        # NaN rows are molecules that could not be placed; matplotlib skips them, but
        # drop them explicitly so the legend count is honest.
        finite = new_coords[~np.isnan(new_coords).any(axis=1)]
        # Shrink markers as the input grows, so a few hundred molecules stay
        # prominent and ten thousand still leave the landscape visible.
        size = float(np.clip(2_000 / max(len(finite), 1), 1, 10))
        self.ax.scatter(
            finite[:, 0], finite[:, 1],
            color=rgb("overlay"), s=size, alpha=1.0,
            edgecolors="white", linewidths=0.3,
            label=f"{label} (n={len(finite):,})", zorder=10,
        )
        proj_label = PROJECTION_LABELS.get(projection_name, projection_name.upper())
        _style_axes(self.ax, f"Overlay — {proj_label}")
        self.ax.legend(loc="upper right", frameon=True, edgecolor="#dddddd")


class CoordinateDensityPlot(BasePlot):
    """
    Where the projected molecules landed, against the reference density.

    One panel per axis, each comparing the input's coordinate distribution (outline)
    with the reference's (filled): a flat overlay tells you *where* points are, this
    tells you whether the input covers the reference space or piles into a corner.
    """

    def __init__(self, projection_name, path, reference_path=None, ax=None, fetcher=None):
        self.projection_name = projection_name
        self.name = f"{projection_name}_coordinate_density"
        super().__init__(ax=ax, path=path, cells=(2, 6), panels=(1, 2))

        fetcher = fetcher or ResultsFetcher(path, reference_path)
        ref = fetcher.reference_coords(projection_name)
        new = fetcher.projected_coords(projection_name)
        if ref is None or new is None:
            self.is_available = False
            return
        new = new[~np.isnan(new).any(axis=1)]
        if len(new) == 0:
            self.is_available = False
            return

        color = projection_rgb(projection_name)
        # Surrogates can place a molecule slightly outside [-1, 1]; widen the bins
        # rather than silently dropping those molecules from the histogram.
        lo = min(EXTENT[0], float(new.min()))
        hi = max(EXTENT[1], float(new.max()))
        bins = np.linspace(lo, hi, 60)

        first = self.ax
        for axis in (0, 1):
            ax = first if axis == 0 else self.next_ax()
            ax.hist(ref[:, axis], bins=bins, density=True, histtype="stepfilled",
                    color=rgb("reference", lighten=0.6), label="reference")
            ax.hist(new[:, axis], bins=bins, density=True, histtype="step",
                    color=color, linewidth=STEP_LINEWIDTH, label="input")
            ax.set_xlim(lo, hi)
            stylia.label(ax, xlabel=f"{'xy'[axis]} coordinate",
                         ylabel="density" if axis == 0 else "")
        first.legend()

        label = PROJECTION_LABELS.get(projection_name, projection_name.upper())
        self.fig.suptitle(f"Coordinate coverage — {label}",
                          x=0.02, y=1.0, ha="left", va="bottom")


# ----------------------------------------------------------------------
# Validation figures (built during surrogate cross-validation)
# ----------------------------------------------------------------------

class FoldComparisonPlot(BasePlot):
    """Ground truth vs surrogate prediction for one held-out fold, side by side."""

    def __init__(self, projection_name, fold_num, y_test_true, y_test_pred,
                 metrics, path, background_image=None):
        self.name = f"{projection_name}_fold_{fold_num}_comparison"
        super().__init__(path=path, cells=(3, 6), panels=(1, 2))

        ax1 = self.ax
        ax2 = self.next_ax()

        r2_val, euc_val = metrics
        for ax, coords, color, name in (
            (ax1, y_test_true, rgb("truth"), "Ground truth"),
            (ax2, y_test_pred, rgb("prediction"), "Surrogate prediction"),
        ):
            # The full-dataset background is rendered once by the caller with
            # datashader and reused for every fold; scattering it per fold was the
            # slowest part of reporting on a large library.
            if background_image is not None:
                ax.imshow(background_image, extent=EXTENT, interpolation="lanczos")
            ax.scatter(coords[:, 0], coords[:, 1], color=color, s=2, alpha=0.6,
                       zorder=10, label=name)
            _style_axes(ax, name)
            ax.set_xlim(EXTENT[0], EXTENT[1])
            ax.set_ylim(EXTENT[2], EXTENT[3])

        label = PROJECTION_LABELS.get(projection_name, projection_name.upper())
        self.fig.suptitle(
            f"{label} surrogate — fold {fold_num}   "
            f"R² = {r2_val:.4f}   mean Euclidean error = {euc_val:.4f}",
            x=0.02, y=1.0, ha="left", va="bottom",
        )


class FoldDistributionsPlot(BasePlot):
    """Per-axis parity hexbins and value distributions for one held-out fold."""

    def __init__(self, projection_name, fold_num, y_test_true, y_test_pred, path):
        self.name = f"{projection_name}_fold_{fold_num}_distributions"
        super().__init__(path=path, cells=(4, 6), panels=(2, 2))

        first = self.ax
        for axis, cmap in ((0, "Blues"), (1, "Greens")):
            ax = first if axis == 0 else self.next_ax()
            true_v, pred_v = y_test_true[:, axis], y_test_pred[:, axis]
            ax.hexbin(true_v, pred_v, gridsize=40, cmap=cmap, mincnt=1)
            lims = [min(true_v.min(), pred_v.min()), max(true_v.max(), pred_v.max())]
            ax.plot(lims, lims, "--", color=rgb("prediction"), linewidth=1)
            try:
                r = pearsonr(true_v, pred_v)[0]
                ax.set_title(f"{'xy'[axis]} parity — Pearson R = {r:.4f}", loc="left")
            except Exception:  # pragma: no cover - degenerate fold
                ax.set_title(f"{'xy'[axis]} parity", loc="left")
            ax.set_xlabel(f"true {'xy'[axis]}")
            ax.set_ylabel(f"predicted {'xy'[axis]}")

        for axis in (0, 1):
            ax = self.next_ax()
            bins = np.linspace(-1.05, 1.05, 50)
            ax.hist(y_test_true[:, axis], bins=bins, density=True, histtype="step",
                    color=rgb("truth"), linewidth=STEP_LINEWIDTH, label="true")
            ax.hist(y_test_pred[:, axis], bins=bins, density=True, histtype="step",
                    color=rgb("prediction"), linewidth=STEP_LINEWIDTH,
                    label="predicted")
            ax.set_title(f"{'xy'[axis]} distribution", loc="left")
            ax.set_xlabel(f"{'xy'[axis]} coordinate")
            ax.set_ylabel("density")
            ax.legend(frameon=False, fontsize="small")


class FoldZonesPlot(BasePlot):
    """
    Whether spatial zones stay coherent under prediction.

    KMeans zones are fitted on the ground truth and the same labels are applied to
    the predictions, so a zone that scatters is a zone the surrogate distorts.
    """

    def __init__(self, projection_name, fold_num, y_test_true, y_test_pred, path,
                 n_zones=10, random_state=42):
        self.name = f"{projection_name}_fold_{fold_num}_zones"
        super().__init__(path=path, cells=(3, 6), panels=(1, 2))

        ax1 = self.ax
        ax2 = self.next_ax()

        n_zones = int(min(n_zones, max(2, len(y_test_true) // 2)))
        km = KMeans(n_clusters=n_zones, random_state=random_state, n_init="auto")
        labels = km.fit_predict(y_test_true)
        cmap = "tab10" if n_zones <= 10 else "tab20"

        for ax, coords, name in (
            (ax1, y_test_true, "Ground truth zones"),
            (ax2, y_test_pred, "Predicted, coloured by true zone"),
        ):
            ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, s=4, alpha=0.7)
            ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                       marker="x", c="black", s=40, zorder=10)
            _style_axes(ax, name)
            ax.set_xlim(EXTENT[0], EXTENT[1])
            ax.set_ylim(EXTENT[2], EXTENT[3])

        label = PROJECTION_LABELS.get(projection_name, projection_name.upper())
        self.fig.suptitle(f"{label} surrogate — fold {fold_num}: zone coherence",
                          x=0.02, y=1.0, ha="left", va="bottom")


# ----------------------------------------------------------------------
# Summary figures
# ----------------------------------------------------------------------

class CvMetricBarsPlot(BasePlot):
    """Cross-validation metrics for every surrogate, mean ± std, side by side."""

    name = "cv_metric_bars"

    def __init__(self, path, ax=None, fetcher=None):
        super().__init__(ax=ax, path=path, cells=(3, 4))
        fetcher = fetcher or ResultsFetcher(path)

        series = []
        for projection in ("tsne", "umap"):
            m = fetcher.metrics(projection)
            if m:
                series.append((projection, m))
        if not series:
            self.is_available = False
            return

        keys = [("r2", "R²"), ("rmse", "RMSE"), ("mae", "MAE"),
                ("euclidean", "Euclidean")]
        width = 0.8 / len(series)
        positions = np.arange(len(keys))

        for i, (projection, m) in enumerate(series):
            means = [m[f"{k}_mean"] for k, _ in keys]
            stds = [m[f"{k}_std"] for k, _ in keys]
            self.ax.bar(
                positions + i * width, means, width, yerr=stds, capsize=3,
                color=projection_rgb(projection),
                label=PROJECTION_LABELS.get(projection, projection),
            )

        self.ax.set_xticks(positions + width * (len(series) - 1) / 2)
        self.ax.set_xticklabels([label for _, label in keys])
        self.ax.set_xlabel("")
        self.ax.set_ylabel("value")
        self.ax.set_title("Surrogate cross-validation metrics (mean ± std)", loc="left")
        self.ax.legend(frameon=False, fontsize="small")


class EuclideanErrorHistogramPlot(BasePlot):
    """
    Distribution of per-molecule coordinate error, pooled across folds.

    The mean alone hides the tail: this shows how many molecules are badly placed,
    which is what determines whether the map can be trusted locally.
    """

    name = "euclidean_error_histogram"

    def __init__(self, path, ax=None, fetcher=None):
        super().__init__(ax=ax, path=path, cells=(3, 4))
        fetcher = fetcher or ResultsFetcher(path)

        drawn = False
        for projection in ("tsne", "umap"):
            df = fetcher.fold_predictions(projection)
            if df is None or "euclidean_error" not in df.columns:
                continue
            errors = df["euclidean_error"].to_numpy()
            self.ax.hist(
                errors, bins=60, density=True, histtype="step",
                linewidth=STEP_LINEWIDTH,
                color=projection_rgb(projection),
                label=f"{PROJECTION_LABELS.get(projection, projection)} "
                      f"(median {np.median(errors):.3f})",
            )
            drawn = True

        if not drawn:
            self.is_available = False
            return

        self.ax.set_xlabel("Euclidean coordinate error")
        self.ax.set_ylabel("density")
        self.ax.set_title("Per-molecule surrogate error, pooled across folds", loc="left")
        self.ax.legend(frameon=False, fontsize="small")


class StepTimingPlot(BasePlot):
    """Wall-clock time per pipeline step, coloured by projection."""

    name = "step_timing"

    def __init__(self, path, ax=None, fetcher=None):
        super().__init__(ax=ax, path=path, cells=(2, 3))
        fetcher = fetcher or ResultsFetcher(path)
        timings = fetcher.timings()
        if not timings:
            self.is_available = False
            return

        steps = list(timings.keys())
        values = [timings[s] for s in steps]
        colors = [projection_rgb(s) for s in steps]

        self.ax.barh(range(len(steps)), values, color=colors)
        self.ax.set_yticks(range(len(steps)))
        self.ax.set_yticklabels([PROJECTION_LABELS.get(s, s) for s in steps])
        self.ax.invert_yaxis()
        self.ax.set_xlabel("seconds")
        self.ax.set_ylabel("")
        self.ax.set_title("Wall-clock time per step", loc="left")
        for i, v in enumerate(values):
            self.ax.text(v, i, f" {v:.1f}s", va="center", fontsize="small")
