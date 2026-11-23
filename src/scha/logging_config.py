"""Logging configuration shared by scha command-line drivers (mirrors dms_datastore.logging_config)."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LoggingConfig:
    """Immutable settings for :func:`configure_logging_config`.

    Parameters
    ----------
    package_name : str
        Root of the logger tree to configure.
    level : int, optional
        Logging level for the package logger and its handlers (default ``logging.INFO``).
    console : bool, optional
        Attach a stderr handler (default True).
    logdir : Path, optional
        Directory for a timestamped log file; no file is written when None (default).
    logfile_prefix : str, optional
        Prefix of the log file name (default ``"run"``).
    include_pid_in_filename : bool, optional
        Append the process id to the log file name (default True).
    clear_handlers : bool, optional
        Remove existing handlers of the package logger first, so repeated calls and forked
        children do not duplicate output (default True).
    propagate : bool, optional
        Propagate records to the root logger (default False, which avoids double logging).
    include_third_party : bool, optional
        If False (default), the ``numba``, ``matplotlib`` and ``PIL`` loggers are set to
        WARNING.
    """

    package_name: str
    level: int = logging.INFO

    # Handlers
    console: bool = True
    logdir: Optional[Path] = None
    logfile_prefix: str = "run"

    # Multiprocessing / uniqueness
    include_pid_in_filename: bool = True

    # Hygiene / UX
    clear_handlers: bool = True          # important for re-entry and forked children
    propagate: bool = False              # avoid double logging via root
    include_third_party: bool = False    # optionally quiet noisy libs below


def configure_logging(
    *,
    package_name: str = "scha",
    level: int = logging.INFO,
    console: bool = True,
    logdir: Optional[Path] = None,
    logfile_prefix: str = "run",
    include_pid_in_filename: bool = True,
    clear_handlers: bool = True,
    propagate: bool = False,
    include_third_party: bool = False,
) -> Path | None:
    """Configure logging from keyword arguments (wrapper around :func:`configure_logging_config`).

    Builds a :class:`scha.logging_config.LoggingConfig` from the arguments.

    Parameters
    ----------
    package_name : str, optional
        Root of the logger tree to configure (default ``"scha"``). Drivers log under ``scha.*``
        so one call covers both the library and the driver.
    level : int, optional
        Logging level for the package logger and its handlers (default ``logging.INFO``).
    console : bool, optional
        Attach a stderr handler (default True).
    logdir : Path, optional
        Directory for a timestamped log file. No file is written when None (default).
    logfile_prefix : str, optional
        Prefix of the log file name (default ``"run"``).
    include_pid_in_filename : bool, optional
        Append the process id to the log file name (default True).
    clear_handlers : bool, optional
        Remove existing handlers first (safe re-entry; default True).
    propagate : bool, optional
        Propagate records to the root logger (default False).
    include_third_party : bool, optional
        If False (default), noisy third-party loggers (``numba``, ``matplotlib``, ``PIL``) are
        set to WARNING; if True they are left at their own levels.

    Returns
    -------
    Path or None
        Log file path if a file handler was created.
    """
    return configure_logging_config(
        LoggingConfig(
            package_name=package_name,
            level=level,
            console=console,
            logdir=logdir,
            logfile_prefix=logfile_prefix,
            include_pid_in_filename=include_pid_in_filename,
            clear_handlers=clear_handlers,
            propagate=propagate,
            include_third_party=include_third_party,
        )
    )


def configure_logging_config(cfg: LoggingConfig) -> Path | None:
    """Configure logging for a package and, optionally, a log file.

    Contract:

    - Only configures the *package* logger tree rooted at cfg.package_name.
    - Safe to call multiple times (when cfg.clear_handlers=True).
    - Returns logfile path if a file handler is created; otherwise returns None.

    Parameters
    ----------
    cfg : LoggingConfig
        Logging settings. When ``cfg.logdir`` is set the directory is created and the log file is
        named ``<logfile_prefix>_<YYYYmmdd_HHMMSS>[_pid<pid>].log``.

    Returns
    -------
    Path or None
        Log file path if a file handler was created.
    """
    pkg_logger = logging.getLogger(cfg.package_name)
    pkg_logger.setLevel(cfg.level)
    pkg_logger.propagate = cfg.propagate

    if cfg.clear_handlers:
        # repeated calls during development/tests and spawned/forked children
        pkg_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if cfg.console:
        sh = logging.StreamHandler(stream=sys.stderr)
        sh.setLevel(cfg.level)
        sh.setFormatter(formatter)
        pkg_logger.addHandler(sh)

    logfile_path: Path | None = None
    if cfg.logdir is not None:
        logdir = Path(cfg.logdir)
        logdir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid() if cfg.include_pid_in_filename else None

        name_bits = [cfg.logfile_prefix, ts]
        if pid is not None:
            name_bits.append(f"pid{pid}")
        filename = "_".join(name_bits) + ".log"

        logfile_path = logdir / filename

        fh = logging.FileHandler(logfile_path, encoding="utf-8")
        fh.setLevel(cfg.level)
        fh.setFormatter(formatter)
        pkg_logger.addHandler(fh)

    if not cfg.include_third_party:
        for noisy in ("numba", "matplotlib", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    if logfile_path is not None:
        pkg_logger.debug("Logging to file: %s", logfile_path)

    return logfile_path


def resolve_loglevel(
    *,
    debug: bool = False,
    quiet: bool = False,
    verbose: int = 0,
    loglevel: Optional[str] = None,
) -> tuple[int, bool]:
    """Convert common CLI flags into a logging level and console enable flag.

    - quiet controls console only (file logging remains available via logdir)
    - explicit loglevel overrides debug/verbose

    Parameters
    ----------
    debug : bool, optional
        Request DEBUG level (default False).
    quiet : bool, optional
        Disable console output (default False).
    verbose : int, optional
        Verbosity count; 2 or more gives DEBUG level (default 0).
    loglevel : str, optional
        Explicit level name such as ``"warning"`` (case-insensitive); overrides ``debug`` and
        ``verbose``.

    Returns
    -------
    level : int
        Logging level: the named level if ``loglevel`` is given, ``logging.DEBUG`` if ``debug``
        or ``verbose >= 2``, otherwise ``logging.INFO``.
    console : bool
        True unless ``quiet``.

    Raises
    ------
    ValueError
        If ``loglevel`` is not an attribute name of the :mod:`logging` module.
    """
    console = not quiet

    if loglevel is not None:
        name = str(loglevel).upper()
        if not hasattr(logging, name):
            raise ValueError(f"Invalid loglevel={loglevel!r}")
        return getattr(logging, name), console

    if debug or verbose >= 2:
        return logging.DEBUG, console

    return logging.INFO, console
