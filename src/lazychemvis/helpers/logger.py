"""
helpers/logger.py
-----------------
Centralised logging setup for the package.

Two sinks:

* **file** — always at DEBUG, so a completed run carries a full record next to the
  artifacts it produced. Attached at runtime via :func:`attach_file_sink`, because the
  destination depends on ``--reference`` / ``--output``.
* **console** — WARNING by default so a normal run stays readable, DEBUG when
  ``--verbose`` is passed. Rendered through :data:`lazychemvis.helpers.console.console`,
  the same console the progress widgets use, so a log line emitted during a live region
  is drawn above it instead of corrupting it.

Usage (in any module):
    from ..helpers.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Featurizer loaded.")
"""

import os

from loguru import logger as _loguru_logger
from rich.logging import RichHandler

from .console import console

ROTATION = "10 MB"
RETENTION = 5

# Full module:function:line context makes file logs traceable. backtrace=True records
# tracebacks for logged exceptions; diagnose=False avoids writing variable values
# (notably SMILES) into the log file.
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
)

LOG_FILENAME = "lazychemvis.log"

_loguru_logger.remove()

_loguru_logger.level("DEBUG", color="<cyan><bold>")
_loguru_logger.level("INFO", color="<blue><bold>")
_loguru_logger.level("WARNING", color="<white><bold><bg yellow>")
_loguru_logger.level("ERROR", color="<white><bold><bg red>")
_loguru_logger.level("CRITICAL", color="<white><bold><bg red>")
_loguru_logger.level("SUCCESS", color="<black><bold><bg green>")


class _LogManager(object):
    """
    Owns the package's loguru sinks and can re-assert them on demand.
    """

    def __init__(self):
        self.logger = _loguru_logger
        self._console_id = None
        self._file_id = None
        self._file_path = None
        # Quiet by default: the console shows WARNING and above. set_verbosity(True),
        # wired to --verbose, flips it to the full DEBUG stream. The file sink always
        # keeps everything regardless.
        self._verbose = False
        self.configure()

    # ------------------------------------------------------------------
    # Sink management
    # ------------------------------------------------------------------

    def configure(self):
        """
        Re-assert our sinks on the shared, process-global loguru logger.

        This is not merely defensive. ``ersilia`` — lazy-imported partway through a fit,
        when the CheMeleon and CLAMP featurizers first run — calls ``logger.remove()`` at
        import time, which wipes *all* handlers including ours and would leave the rest of
        the run silent. Call this immediately after any such import. The console level
        follows ``self._verbose``, so repeated re-assertions preserve the chosen verbosity.
        """
        self.logger.remove()
        self._console_id = None
        self._file_id = None
        if self._file_path:
            self._add_file_sink(self._file_path)
        self._add_console_sink("DEBUG" if self._verbose else "WARNING")

    def attach_file_sink(self, dir_path: str, filename: str = LOG_FILENAME):
        """
        Start writing a DEBUG log file inside a run's output directory.

        Parameters
        ----------
        dir_path : str
            Directory to write into; created if missing.
        filename : str, default='lazychemvis.log'
            Log file name.

        Returns
        -------
        str or None
            The log file path, or None if it could not be created (logging to a file is
            never allowed to abort a run).
        """
        try:
            os.makedirs(dir_path, exist_ok=True)
            path = os.path.join(os.path.abspath(dir_path), filename)
            if self._file_id is not None:
                self.logger.remove(self._file_id)
                self._file_id = None
            self._add_file_sink(path)
            self._file_path = path
            return path
        except Exception as e:  # pragma: no cover - defensive
            self.logger.warning(f"Could not open log file in {dir_path}: {e}")
            return None

    def _add_file_sink(self, path: str):
        self._file_id = self.logger.add(
            path,
            level="DEBUG",
            rotation=ROTATION,
            retention=RETENTION,
            format=_FILE_FORMAT,
            backtrace=True,
            diagnose=False,
            enqueue=False,
        )

    def _add_console_sink(self, level: str = "WARNING"):
        if self._console_id is not None:
            return
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            log_time_format="%H:%M:%S",
            show_path=False,
            show_level=True,
        )
        self._console_id = self.logger.add(
            handler, level=level, format="{message}", colorize=True
        )

    # ------------------------------------------------------------------
    # Verbosity
    # ------------------------------------------------------------------

    def set_verbosity(self, verbose: bool):
        """
        Set console verbosity and re-assert the sinks so it takes effect.

        Quiet (default) shows WARNING and above; verbose shows the full DEBUG stream.
        The file sink is unaffected — it always records DEBUG.
        """
        self._verbose = bool(verbose)
        self.configure()

    @property
    def verbose(self):
        return self._verbose

    @property
    def file_path(self):
        """Path of the active log file, or None if no file sink is attached."""
        return self._file_path


manager = _LogManager()


def get_logger(name: str):
    """
    Return a loguru logger bound to the given module name.

    Parameters
    ----------
    name : str
        Typically passed as ``__name__`` from the calling module. Appears in the
        ``{name}`` field of every file record so you can tell which module emitted it.

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


def configure():
    """Re-assert the package's log sinks (see :meth:`_LogManager.configure`)."""
    manager.configure()


def attach_file_sink(dir_path: str, filename: str = LOG_FILENAME):
    """Begin writing a DEBUG log file into ``dir_path``. Returns the path or None."""
    return manager.attach_file_sink(dir_path, filename)


def set_verbosity(verbose: bool):
    """Show the full DEBUG stream on the console when ``verbose`` is True."""
    manager.set_verbosity(verbose)


def log_file_path():
    """Path of the active log file, or None."""
    return manager.file_path
