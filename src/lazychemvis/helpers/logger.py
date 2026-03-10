"""
helpers/logger.py
-----------------
Centralised logging setup for the package.

Usage (in any module):
    from ..helpers.logger import get_logger, console

    logger = get_logger(__name__)
    logger.info("Something happened")
    console.print(Panel("Hello"))
"""

import sys
from loguru import logger as _loguru_logger
from rich.console import Console

# ---------------------------------------------------------------------------
# Shared Rich console — import this wherever Rich widgets are needed
# ---------------------------------------------------------------------------
console = Console()

# ---------------------------------------------------------------------------
# Loguru configuration
# ---------------------------------------------------------------------------
# Remove the default stderr sink so we have full control.
_loguru_logger.remove()

# Primary sink: routes through Rich's Console for unified, coloured output.
_loguru_logger.add(
    lambda msg: console.print(msg, end=""),
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
    level="DEBUG",
)

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