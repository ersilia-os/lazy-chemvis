"""
Live, in-place progress widgets.

A single-line determinate progress bar for the long steps of a fit — the batched
Ersilia featurizers and the TMAP subprocess. Renders to the shared console, and
degrades to one terse line when stdout is not a terminal (piped output, CI).

Deliberately *not* built on ``rich.Progress`` nested inside another live region:
only one Rich ``Live`` may be active at a time, and a nested one silently renders
nothing. Everything here assumes it owns the live region for its duration.
"""

import contextlib
import time

from rich.text import Text

from .console import active_color, console, echo

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _spinner_frame():
    return _SPINNER_FRAMES[int(time.time() * 10) % len(_SPINNER_FRAMES)]


def bar(frac, width=12):
    """Render a mini progress bar (filled █ against dim ─) for a fraction in [0, 1]."""
    try:
        frac = max(0.0, min(1.0, float(frac)))
    except (TypeError, ValueError):
        frac = 0.0
    filled = round(frac * width)
    return f"{'█' * filled}[dim]{'─' * (width - filled)}[/dim]"


class LiveProgressBar(object):
    """
    A one-line, in-place progress bar for a flat sequence of N items.

    Transient while running; prints one final summary line on exit as the step's
    record. Off-TTY it emits a single completion line and no animation.

    Examples
    --------
    >>> with LiveProgressBar("ECFP", total=50).live() as p:
    ...     for chunk in chunks:
    ...         p.set_note("batch 3")
    ...         p.advance()
    """

    def __init__(self, title, total, color=None, width=24, persist=True, show_bar=True):
        self.title = title
        self.total = max(0, int(total))
        self.color = color or active_color()
        self.width = width
        self.persist = persist
        self.show_bar = show_bar
        self.done = 0
        self.note = ""
        self._live = None
        self._plain = False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, frac, spin=True):
        head = "" if (not spin or self.done >= self.total) else f"{_spinner_frame()} "
        note = f"  [dim]{self.note}[/dim]" if self.note else ""
        glyph = f"{bar(frac, self.width)} " if self.show_bar else ""
        return Text.from_markup(
            f"  [bold {self.color}]{head}{self.title}[/bold {self.color}]  "
            f"{glyph}[dim]{self.done}/{self.total}[/dim]{note}"
        )

    def __rich__(self):
        return self._render((self.done / self.total) if self.total else 1.0)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def advance(self, n=1):
        """Mark ``n`` items complete and redraw."""
        self.done = min(self.total, self.done + n)
        self._refresh()

    def set_note(self, note):
        """Set the dim trailing note (e.g. the item being worked on) and redraw."""
        self.note = note or ""
        self._refresh()

    def _refresh(self):
        # No refresh=True: these bars can advance in a tight loop, so swap the renderable
        # and let Live's auto-refresh thread redraw at its capped rate. The spinner still
        # animates on each tick.
        live = self._live
        if live is not None:
            live.update(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def live(self):
        """Animate in place on a TTY, or degrade to a single terse line off-TTY."""
        if not console.is_terminal:
            self._plain, self._live = True, None
            try:
                yield self
            finally:
                if self.persist:
                    echo(f"{self.title}: {self.done}/{self.total}", kind="info")
            return

        from rich.live import Live

        try:
            with Live(
                self, console=console, transient=True, refresh_per_second=8, auto_refresh=True
            ) as live:
                self._live = live
                yield self
        finally:
            self._live = None
            if self.persist:
                console.print(self._render(1.0, spin=False))
