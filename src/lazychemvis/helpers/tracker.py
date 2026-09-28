"""
Pipeline step tracking for the console.

A clean top-to-bottom stream rather than a persistent animated panel: each step
prints a themed banner when it opens and a completion line with its wall-clock and
host resource usage when it closes. Every step owns an accent colour, which the
console helpers pick up automatically for the duration of that step.

This deliberately holds **no** long-lived ``Live`` region. The previous
``spinner()`` wrapper did, which silently suppressed every ``rich.Progress`` bar
created inside a step — only one Rich ``Live`` may be active at a time, and a
nested one renders nothing.

Timings recorded here are persisted by the caller and rendered in the HTML report.
"""

import time

from .console import (
    detail,
    echo,
    resource_caption,
    rule,
    set_active_color,
    summary_panel,
)

# Accent colour per pipeline step, from stylia's named palette.
STEP_COLORS = {
    "pca": "cyan",
    "tmap": "magenta",
    "tsne": "green",
    "umap": "yellow",
    "report": "blue",
}

DEFAULT_COLOR = "cyan"

# Banner heading per step. The four projections are pipelines; the report is not.
STEP_TITLES = {
    "pca": "PCA pipeline",
    "tmap": "TMAP pipeline",
    "tsne": "t-SNE pipeline",
    "umap": "UMAP pipeline",
    "report": "Report",
}


class PipelineTracker(object):
    """
    Track and render the progress of a multi-step pipeline.

    Examples
    --------
    >>> tracker = PipelineTracker()
    >>> tracker.begin("Fitting reference space", "10,000 molecules")
    >>> tracker.start("pca", "RDKit descriptors -> PCA")
    >>> tracker.substep("RDKit featurization")
    >>> tracker.complete("pca")
    >>> tracker.finish()
    """

    def __init__(self):
        self._t0 = None
        self._step_t0 = None
        self._current = None
        self.timings = {}

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def begin(self, title, subtitle=None):
        """Print the run header and start the overall clock."""
        self._t0 = time.time()
        rows = []
        if subtitle:
            rows.append(("input", subtitle))
        caption = resource_caption()
        if caption:
            rows.append(("host", caption))
        set_active_color(DEFAULT_COLOR)
        summary_panel(title, rows, border_style=DEFAULT_COLOR)

    def finish(self, extra_rows=None):
        """Print the closing summary with total elapsed time and per-step breakdown."""
        set_active_color(DEFAULT_COLOR)
        total = time.time() - self._t0 if self._t0 else None
        rows = []
        for key, elapsed in self.timings.items():
            rows.append((key, f"{elapsed:.1f}s"))
        if total is not None:
            rows.append(("total", f"[bold]{total:.1f}s[/bold]"))
        for row in extra_rows or []:
            rows.append(row)
        summary_panel("Complete", rows, border_style="green", icon="✓")

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def start(self, key, description=None):
        """Open a step: set its accent colour and print a themed banner."""
        self._current = key
        self._step_t0 = time.time()
        color = STEP_COLORS.get(key, DEFAULT_COLOR)
        set_active_color(color)
        title = STEP_TITLES.get(key, f"{key.upper()} pipeline")
        if description:
            title = f"{title} — {description}"
        rule(title, right=resource_caption())

    def substep(self, text):
        """Announce a sub-step within the current step."""
        echo(text, kind="run")

    def complete(self, key=None, rows=None, ok=True):
        """
        Close a step, recording and printing its wall-clock time.

        Parameters
        ----------
        key : str, optional
            Step key; defaults to the currently open step.
        rows : iterable of (str, str), optional
            Extra label/value detail rows to print beneath the completion line.
        ok : bool, default=True
            False renders the step as failed.
        """
        key = key or self._current
        elapsed = time.time() - self._step_t0 if self._step_t0 else None
        if key and elapsed is not None:
            self.timings[key] = elapsed
        stamp = f"{elapsed:.1f}s" if elapsed is not None else "—"
        caption = resource_caption()
        tail = f"  [dim]·  {caption}[/dim]" if caption else ""
        if ok:
            echo(f"{key} complete  [dim]{stamp}[/dim]{tail}", kind="success")
        else:
            echo(f"{key} failed  [dim]{stamp}[/dim]", kind="error")
        if rows:
            detail(rows)
        self._current = None
        self._step_t0 = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def step(self, key, description=None):
        """
        Context manager wrapping :meth:`start` / :meth:`complete`.

        Marks the step failed and re-raises if the body raises.
        """
        return _StepContext(self, key, description)


class _StepContext(object):
    def __init__(self, tracker, key, description):
        self.tracker = tracker
        self.key = key
        self.description = description

    def __enter__(self):
        self.tracker.start(self.key, self.description)
        return self.tracker

    def __exit__(self, exc_type, exc, tb):
        self.tracker.complete(self.key, ok=exc_type is None)
        return False
