"""
Shared Rich console and themed output helpers.

There is exactly one ``Console`` in the process, exported here. Log records are
rendered through this same console (see :mod:`lazychemvis.helpers.logger`), so a
warning emitted while a live region is active is drawn cleanly above the live
display instead of corrupting it — a second Console would bypass Rich's live
bookkeeping and leave stale lines behind.

Helpers pick up the current pipeline step's accent colour automatically, so call
sites do not have to thread a colour through.
"""

import contextlib
import io
import sys

from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table

# highlight=False: Rich's automatic highlighting turns numbers and paths in our
# own messages into unintended colours.
#
# file=sys.stdout is pinned deliberately. A Console built without a file resolves
# ``sys.stdout`` at every write, so any ``redirect_stdout`` — which is how
# quiet() silences third-party output — would swallow our own live display too.
# Binding the real stream once keeps our output immune to those redirects.
console = Console(highlight=False, file=sys.stdout)

_DEFAULT_COLOR = "cyan"
_active_color = _DEFAULT_COLOR

# Icon and style per echo() kind.
_KINDS = {
    "success": ("✓", "bold green"),
    "warning": ("⚠", "bold yellow"),
    "error": ("✖", "bold red"),
    "run": ("▪", None),  # None → the active step colour
    "info": ("·", "dim"),
}


@contextlib.contextmanager
def quiet(logger=None, label: str = None):
    """
    Swallow anything a third-party library writes to stdout/stderr.

    Ersilia's ``Model.run`` builds its own Rich ``Console`` and ``Progress`` on
    every call and prints unconditionally. Two Rich live regions cannot share a
    terminal: its bar has no idea ours exists, so each batch printed a fresh
    "Running model …" line and a second bar right over our in-place one, turning
    678 in-place updates into 1,356 scrolling lines. Redirecting the interpreter
    streams keeps that output off the terminal while our pinned ``console``
    (see above) keeps drawing.

    The captured text is not thrown away: it goes to the given logger at debug
    level, so ``--verbose`` runs and the log file still have it.

    Parameters
    ----------
    logger : logging.Logger, optional
        Where to send the captured text, at debug level. Dropped if omitted.
    label : str, optional
        Prefix for the debug record, e.g. the batch being run.
    """
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        if logger is not None:
            # One record per call: off-terminal, a Rich progress bar writes several
            # lines of box-drawing characters, and 678 batches of that would drown a
            # --verbose run just as badly as the original leak did.
            captured = " ".join(sink.getvalue().split())
            if captured:
                logger.debug(f"{label + ': ' if label else ''}{captured}")


def set_active_color(color: str) -> None:
    """Set the accent colour used by the themed helpers (called when a step starts)."""
    global _active_color
    _active_color = color or _DEFAULT_COLOR


def active_color() -> str:
    """Return the current step's accent colour."""
    return _active_color


def echo(text: str, kind: str = "info") -> None:
    """
    Print a one-line status message with an icon.

    Parameters
    ----------
    text : str
        Message to display. Rich markup is honoured.
    kind : {'info', 'success', 'warning', 'error', 'run'}
        Selects the icon and colour. Unknown kinds fall back to 'info'.
    """
    icon, style = _KINDS.get(kind, _KINDS["info"])
    if style is None:
        style = active_color()
    console.print(f"  [{style}]{icon}[/{style}]  {text}")


def rule(title: str, *, style: str = None, right: str = None) -> None:
    """
    Print a left-aligned section divider, optionally with dim right-aligned text.

    Parameters
    ----------
    title : str
        Section heading.
    style : str, optional
        Colour override; defaults to the active step colour.
    right : str, optional
        Dim text aligned to the right edge (e.g. a resource caption).
    """
    color = style or active_color()
    left = f"[bold {color}]{title}[/bold {color}]"
    if right:
        pad = max(1, console.width - len(title) - len(right) - 6)
        console.print(f"\n{left}{' ' * pad}[dim]{right}[/dim]")
    else:
        console.print(f"\n{left}")
    console.print(f"[{color}]{'─' * min(console.width - 2, 78)}[/{color}]")


def detail(rows, *, color: str = None, indent: int = 3) -> None:
    """
    Print a borderless block of right-aligned labels against left-aligned values.

    Parameters
    ----------
    rows : iterable of (str, str)
        Label/value pairs. Values may contain Rich markup.
    color : str, optional
        Colour for the labels; defaults to the active step colour.
    indent : int, default=3
        Leading spaces.
    """
    rows = [r for r in rows if r is not None]
    if not rows:
        return
    color = color or active_color()
    table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    table.add_column(justify="right", style=f"dim {color}", no_wrap=True)
    table.add_column(justify="left")
    for label, value in rows:
        table.add_row(str(label), str(value))
    console.print(Padding(table, (0, 0, 0, indent)))


def themed_table(title: str, *, color: str = None, caption: str = None) -> Table:
    """
    Return an empty borderless table (header underline only), themed to the step.

    The caller adds columns and rows, then prints it on ``console``.
    """
    color = color or active_color()
    return Table(
        title=f"[bold {color}]{title}[/bold {color}]" if title else None,
        title_justify="left",
        box=box.SIMPLE_HEAVY,
        pad_edge=False,
        caption=f"[dim]{caption}[/dim]" if caption else None,
        caption_justify="left",
        style=color,
    )


def summary_panel(
    title: str, rows, *, border_style: str = None, icon: str = None
) -> None:
    """
    Print a bordered rounded panel wrapping a two-column label/value table.

    Parameters
    ----------
    title : str
        Panel title.
    rows : iterable of (str, str)
        Label/value pairs.
    border_style : str, optional
        Border colour; defaults to the active step colour.
    icon : str, optional
        Glyph prefixed to the title.
    """
    color = border_style or active_color()
    table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    table.add_column(justify="right", style="dim", no_wrap=True)
    table.add_column(justify="left")
    for label, value in rows:
        table.add_row(str(label), str(value))
    heading = f"{icon} {title}" if icon else title
    console.print(
        Panel(
            table,
            title=f"[bold {color}]{heading}[/bold {color}]",
            title_align="left",
            border_style=color,
            box=box.ROUNDED,
        )
    )


def heat_hex(t: float) -> str:
    """
    Map ``t`` in [0, 1] onto a green → amber → red ramp (0 = calm, 1 = attention).

    Used for resource gauges in the console and the HTML report.
    """
    try:
        t = max(0.0, min(1.0, float(t)))
    except (TypeError, ValueError):
        t = 0.0
    # stylia's lime → amber → crimson, interpolated in RGB.
    stops = [
        (0.0, (0x6B, 0xBF, 0x59)),
        (0.5, (0xFC, 0xBF, 0x49)),
        (1.0, (0xE6, 0x39, 0x46)),
    ]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            r, g, b = (round(a + (b_ - a) * f) for a, b_ in zip(c0, c1))
            return f"#{r:02X}{g:02X}{b:02X}"
    return "#E63946"


def resource_caption():
    """
    Return host CPU and RAM usage as a compact one-liner, or None if unavailable.

    Host-level rather than per-container on purpose: the Ersilia model servers run
    inside Docker, whose load shows up as host CPU/RAM, which is the "is my machine
    saturated" signal that matters. Cheap and non-blocking.
    """
    try:
        import psutil
    except Exception:
        return None
    try:
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        return f"CPU {cpu:.0f}%  ·  RAM {vm.used / 1e9:.1f}/{vm.total / 1e9:.1f} GB ({vm.percent:.0f}%)"
    except Exception:
        return None


def Padding_(renderable, left: int):
    """Indent a renderable by ``left`` spaces (thin wrapper over rich.padding)."""
    from rich.padding import Padding

    return Padding(renderable, (0, 0, 0, left))
