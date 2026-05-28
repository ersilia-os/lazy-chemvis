"""
helpers/logger.py
-----------------
Centralised logging setup for the package.

Usage (in any module):
    from ..helpers.logger import get_logger, console, spinner, echo

    logger = get_logger(__name__)
    spinner("Fitting model", my_func, arg1, arg2)
    echo("Done")
"""

from loguru import logger as _loguru_logger
from rich.console import Console
from rich.text import Text

# ---------------------------------------------------------------------------
# Shared Rich console — import this wherever Rich widgets are needed
# ---------------------------------------------------------------------------
console = Console()

# ---------------------------------------------------------------------------
# Loguru configuration
# ---------------------------------------------------------------------------
# Remove the default sink and disable all loguru output.
# disable("") blocks messages at dispatch time — works even when lazy-imported
# dependencies (e.g. Ersilia) add their own loguru sinks after this point.
# All user-visible output goes through Rich (spinner, echo, console.print).
_loguru_logger.remove()
_loguru_logger.disable("")

# Optional file sink: plain text, full timestamp, auto-rotation.
# Uncomment and adjust the path to enable persistent log files.
# _loguru_logger.add(
#     "logs/package.log",
#     format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
#     level="DEBUG",
#     rotation="10 MB",
#     retention="14 days",
#     compression="zip",
# )


def get_logger(name: str):
    """
    Return a loguru logger bound to the given module name.

    Parameters
    ----------
    name : str
        Typically passed as ``__name__`` from the calling module.
        Appears in the ``{name}`` field of every log record so you
        can tell at a glance which module emitted the message.

    Returns
    -------
    loguru.Logger
        A context-bound logger instance.

    Examples
    --------
    >>> from ..helpers.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Featurizer loaded.")
    """
    return _loguru_logger.bind(name=name)


def spinner(text: str, func, *args, **kwargs):
    """
    Run *func* under a Rich spinner, then print ✓ or ✖ when it finishes.

    Parameters
    ----------
    text : str
        Label shown next to the spinner and in the result line.
    func : callable
        Function to call.
    *args, **kwargs
        Forwarded to *func*.

    Returns
    -------
    The return value of *func*.
    """
    with console.status(Text(f"  {text}...", style="cyan")):
        try:
            result = func(*args, **kwargs)
            console.print(Text(f"  ✓  {text}", style="bold green"))
            return result
        except Exception:
            console.print(Text(f"  ✖  {text}", style="bold red"))
            raise


def echo(text: str, error: bool = False):
    """
    Print a ✓ (or ✖) line without a spinner.

    Parameters
    ----------
    text : str
        Message to display.
    error : bool
        If True, display in red with ✖ instead of green ✓.
    """
    if error:
        console.print(Text(f"  ✖  {text}", style="bold red"))
    else:
        console.print(Text(f"  ✓  {text}", style="bold green"))