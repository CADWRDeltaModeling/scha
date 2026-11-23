"""Command-line interface: ``scha fit | decompose | phasors | ampphase | gains | crossval``.

Each subcommand calls a work function with explicit arguments; the same
operations are available in Python (:func:`scha.fit.fit`, :meth:`scha.fit.SCHAFit.decompose`,
...). For ``scha fit``, keys of a ``--config`` YAML are the keyword names of
:func:`scha.cli.fit_file`; command-line flags override the config.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd

from .fit import SCHAFit, fit, load_fit
from .io import read_series
from .logging_config import configure_logging, resolve_loglevel
from .phase import DEFAULT_EPOCH, amplitude_phase

logger = logging.getLogger("scha.cli")

_FIT_KEYS = ("column", "bands", "order", "method", "requirements", "fixed", "start",
             "maxiter", "workers", "decomposition")


def fit_file(input, output, *, column=None, bands="D1-D6", order=(2, 3),
             method="constrained", requirements=None, fixed=None, start=None, maxiter=80, workers=None,
             decomposition=None, verbose=False) -> SCHAFit:
    """Fit a CSV series and write the fit YAML (and optionally the decomposition).

    Parameters
    ----------
    input, output : path-like
        Input CSV (see :func:`scha.io.read_series`) and output fit YAML.
    column : str, optional
        Value column (see :func:`scha.io.read_series`).
    bands : str or sequence of str, optional
        Tidal bands, default ``"D1-D6"``. Passed to :func:`scha.fit.fit`.
    order : (int, int), optional
        Trend order m and cycle order n, default (2, 3). Passed to :func:`scha.fit.fit` as a tuple.
    method : {"constrained", "ml", "em"}, optional
        Estimation method, default ``"constrained"``. Passed to :func:`scha.fit.fit`.
    requirements : scha.requirements.Requirements or dict, optional
        Gain requirements. Passed to :func:`scha.fit.fit`.
    fixed : dict, optional
        Parameters held fixed. Passed to :func:`scha.fit.fit`.
    start : None, "auto", scha.fit.SCHAFit, path or scha.params.SCHAParams, optional
        Starting values. Passed to :func:`scha.fit.fit`.
    maxiter : int, optional
        Optimiser iteration limit, default 80. Passed to :func:`scha.fit.fit`.
    workers : int, optional
        Processes for gradient evaluation. Passed to :func:`scha.fit.fit`.
    decomposition : path-like, optional
        Also write the decomposition CSV here (index label ``datetime``).
    verbose : bool, optional
        Print per-iteration optimiser output (default False). Passed to :func:`scha.fit.fit`.

    Returns
    -------
    scha.fit.SCHAFit
        The fit that was written. Its ``diagnostics["source"]`` is set to ``input``.
    """
    y = read_series(input, column=column)
    f = fit(y, bands=bands, order=tuple(order), method=method, requirements=requirements, fixed=fixed,
            start=start, maxiter=maxiter, workers=workers, verbose=verbose)
    f.diagnostics["source"] = str(input)
    f.to_yaml(output)
    logger.info("Wrote %s", output)
    if decomposition is not None:
        f.decompose().to_csv(decomposition, index_label="datetime")
        logger.info("Wrote %s", decomposition)
    logger.info("\n%s", f.summary())
    return f


def decompose_file(input, fit_path, output, *, column=None) -> pd.DataFrame:
    """Decompose a CSV series with a stored fit and write the components CSV.

    Parameters
    ----------
    input : path-like
        Input CSV (see :func:`scha.io.read_series`).
    fit_path : path-like
        Fit YAML (see :func:`scha.fit.load_fit`).
    output : path-like
        Components CSV to write (index label ``datetime``).
    column : str, optional
        Value column (see :func:`scha.io.read_series`).

    Returns
    -------
    pd.DataFrame
        The decomposition, see :meth:`scha.fit.SCHAFit.decompose`.
    """
    f = load_fit(fit_path)
    y = read_series(input, column=column)
    dec = f.decompose(y)
    dec.to_csv(output, index_label="datetime")
    logger.info("Wrote %s", output)
    return dec


def phasors_file(input, fit_path, output, *, column=None) -> pd.DataFrame:
    """Write the smoothed top-pair phasors of each band.

    Parameters
    ----------
    input : path-like
        Input CSV (see :func:`scha.io.read_series`).
    fit_path : path-like
        Fit YAML (see :func:`scha.fit.load_fit`).
    output : path-like
        Phasor CSV to write (index label ``datetime``).
    column : str, optional
        Value column (see :func:`scha.io.read_series`).

    Returns
    -------
    pd.DataFrame
        Columns ``<band>_c`` and ``<band>_s``, see :meth:`scha.fit.SCHAFit.phasors`.
    """
    f = load_fit(fit_path)
    y = read_series(input, column=column)
    ph = f.phasors(y)
    ph.to_csv(output, index_label="datetime")
    logger.info("Wrote %s", output)
    return ph


def ampphase_file(input, fit_path, output, *, column=None, epoch=DEFAULT_EPOCH,
                  ref: Optional[dict] = None) -> pd.DataFrame:
    """Write local cos/sin coefficients, amplitude and phase of each band.

    Parameters
    ----------
    input : path-like
        Input CSV (see :func:`scha.io.read_series`).
    fit_path : path-like
        Fit YAML (see :func:`scha.fit.load_fit`).
    output : path-like
        Amplitude/phase CSV to write (index label ``datetime``).
    column : str, optional
        Value column (see :func:`scha.io.read_series`).
    epoch : str or Timestamp, optional
        Time origin of the phase, default ``scha.phase.DEFAULT_EPOCH``.
    ref : dict, optional
        Reference frequency per band: band -> tide name or frequency in cpd. Default None uses
        each band's model line (see :func:`scha.phase.amplitude_phase`).

    Returns
    -------
    pd.DataFrame
        Columns ``<band>_a``, ``<band>_b``, ``<band>_amp`` and ``<band>_phase`` (degrees, lag
        convention).
    """
    f = load_fit(fit_path)
    y = read_series(input, column=column)
    ap = amplitude_phase(f.phasors(y), ref=ref, epoch=epoch)
    ap.to_csv(output, index_label="datetime")
    logger.info("Wrote %s", output)
    return ap


def gains_file(fit_path, output, *, cpd_max: Optional[float] = None, npoints: int = 2000) -> pd.DataFrame:
    """Write the steady-state gain table of a stored fit.

    Parameters
    ----------
    fit_path : path-like
        Fit YAML (see :func:`scha.fit.load_fit`).
    output : path-like
        Gain table CSV to write.
    cpd_max : float, optional
        Highest frequency in cycles per day; default 0.01 cpd below Nyquist of the fit step.
    npoints : int, optional
        Number of equally spaced frequencies from 0.01 cpd to ``cpd_max`` (default 2000).

    Returns
    -------
    pd.DataFrame
        Gains indexed by cpd, see :meth:`scha.fit.SCHAFit.gains`.
    """
    f = load_fit(fit_path)
    top = 12.0 / f.step_h - 0.01 if cpd_max is None else cpd_max
    g = f.gains(np.linspace(0.01, top, npoints))
    g.to_csv(output)
    logger.info("Wrote %s", output)
    return g


_CV_KEYS = ("column", "rho", "sigma2_zeta", "bands", "order", "gap", "spacing", "first",
            "refit_folds", "requirements", "maxiter", "workers")


def crossval_file(input, output, *, rho, sigma2_zeta, column=None, bands="D1-D6",
                  order=(2, 3), gap="25h", spacing="250h", first="100h", refit_folds=True, requirements=None,
                  maxiter=80, workers=None) -> pd.DataFrame:
    """Run :func:`scha.crossval.grid_cv` on a CSV series and write the score table.

    Parameters
    ----------
    input : path-like
        Input CSV (see :func:`scha.io.read_series`).
    output : path-like
        Score table CSV to write (without the index).
    rho, sigma2_zeta : sequence of float
        Grid values, native to the step of the series (required keywords).
    column : str, optional
        Value column (see :func:`scha.io.read_series`).
    bands : str or sequence of str, optional
        Tidal bands, default ``"D1-D6"``.
    order : (int, int), optional
        Trend order m and cycle order n, default (2, 3).
    gap, spacing, first : int or str, optional
        Fold design as durations (``"25h"``, ``"250h"``, ``"100h"`` by default) or sample counts,
        see :func:`scha.crossval.blocks`.
    refit_folds : bool, optional
        Re-estimate sigma2_eps and sigma2_kappa for each fold (default True).
    requirements : scha.requirements.Requirements or dict, optional
        Used only to report gain margins of each cell.
    maxiter : int, optional
        Optimiser iteration limit of each inner fit (default 80).
    workers : int, optional
        Processes over folds; default min(cpu count, number of folds).

    Returns
    -------
    pd.DataFrame
        One row per grid cell, see :func:`scha.crossval.grid_cv`.
    """
    from .crossval import grid_cv
    y = read_series(input, column=column)
    df = grid_cv(y, rho, sigma2_zeta, bands=bands, order=tuple(order), gap=gap, spacing=spacing, first=first,
                 refit_folds=refit_folds, requirements=requirements, maxiter=maxiter, workers=workers)
    df.to_csv(output, index=False)
    logger.info("Wrote %s", output)
    return df


# --- click plumbing ---------------------------------------------------------------------------------

def _logging_options(fn):
    """Decorate a command function with ``--logdir``, ``--debug`` and ``--quiet`` options.

    The wrapper resolves the log level with :func:`scha.logging_config.resolve_loglevel`,
    configures the ``scha`` logger with :func:`scha.logging_config.configure_logging` (log file
    prefix ``scha_<function name>``) and then calls ``fn`` without those three arguments.
    """
    @click.option("--logdir", type=click.Path(path_type=Path), default=None, help="Directory for a log file.")
    @click.option("--debug", is_flag=True, help="Debug-level logging.")
    @click.option("--quiet", "-q", is_flag=True, help="No progress messages on the console.")
    @functools.wraps(fn)
    def wrapper(*args, logdir, debug, quiet, **kwargs):
        level, console = resolve_loglevel(debug=debug, quiet=quiet)
        configure_logging(package_name="scha", level=level, console=console, logdir=logdir,
                          logfile_prefix=f"scha_{fn.__name__}")
        return fn(*args, **kwargs)
    return wrapper


def _series_options(fn):
    """Decorate a command function with the ``--column`` option."""
    fn = click.option("--column", default=None, help="Value column (required if the file has several).")(fn)
    return fn


def _none_or(value: Optional[str]):
    """Return None for None or the string ``"none"`` (any case, surrounding space ignored), else ``value``."""
    if value is None:
        return None
    return None if value.strip().lower() == "none" else value


def _parse_fix(items) -> dict:
    """Parse repeated ``--fix`` items into a dict of fixed parameters.

    Parameters
    ----------
    items : iterable of str
        ``NAME=VALUE`` strings; ``sigma2_kappa:BAND=VALUE`` fixes one band's variance.

    Returns
    -------
    dict
        Name -> float, with ``sigma2_kappa`` mapping band label -> float.

    Raises
    ------
    click.BadParameter
        If an item has no ``=``.
    """
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise click.BadParameter(f"{item!r}: expected NAME=VALUE.", param_hint="--fix")
        name, val = item.split("=", 1)
        if name.startswith("sigma2_kappa:"):
            out.setdefault("sigma2_kappa", {})[name.split(":", 1)[1]] = float(val)
        else:
            out[name] = float(val)
    return out


@click.group()
@click.help_option("-h", "--help")
def main():
    """State-space cyclical harmonic analysis (SCHA) of tidal time series."""


@main.command("fit")
@click.help_option("-h", "--help")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Fit YAML to write.")
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="YAML of fit_file keyword arguments; flags override it.")
@_series_options
@click.option("--bands", default=None, help="Bands, e.g. D1-D6 (default) or D1,D2,D4.")
@click.option("--order", default=None, help="Trend and cycle order m,n (default 2,3).")
@click.option("--method", type=click.Choice(["constrained", "ml", "em"]), default=None,
              help="Estimation method (default constrained).")
@click.option("--delta", type=float, default=None, help="Passband tolerance (default 0.01).")
@click.option("--eps", type=float, default=None, help="Subtide gap tolerance (default 0.01).")
@click.option("--sacrificial", default=None, help="Bands without passband requirement: count, spec or none "
                                                  "(default 2 = highest two).")
@click.option("--sub-cross", type=float, default=None, help="Require subtide gain >= D1 gain at this cpd.")
@click.option("--sub-pass", type=float, default=None, help="Require subtide passband up to this cpd.")
@click.option("--post-filter", default=None, help="Subtide post-filter cutoff, e.g. 18h, or none (default 18h).")
@click.option("--fix", multiple=True, help="Hold a parameter fixed: rho=0.985, sigma2_zeta=1e-7, "
                                           "sigma2_kappa:D1=1e-10. Repeatable.")
@click.option("--start", "start_fit", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Start from this fit YAML (default automatic).")
@click.option("--maxiter", type=int, default=None, help="Optimiser iteration limit (default 80).")
@click.option("--workers", type=int, default=None, help="Processes for gradients (default automatic).")
@click.option("--decomposition", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Also write the decomposition CSV.")
@click.option("--verbose", "-v", is_flag=True, help="Per-iteration optimiser output.")
@_logging_options
def fit_cmd(input, output, config, column, bands, order, method, delta, eps, sacrificial, sub_cross,
            sub_pass, post_filter, fix, start_fit, maxiter, workers, decomposition, verbose):
    """Fit a SCHA model to the series in INPUT and write the fit YAML.

    \f

    Parameters
    ----------
    input : path
        Input CSV (argument).
    output : path
        Fit YAML to write (``--output``).
    config : path or None
        YAML of :func:`scha.cli.fit_file` keyword arguments; unknown keys raise
        ``click.BadParameter``. Flags override it (``--config``).
    column : str or None
        Value column (``--column``).
    bands : str or None
        Bands, e.g. ``D1-D6`` or ``D1,D2,D4`` (``--bands``).
    order : str or None
        Trend and cycle order as ``m,n`` (``--order``).
    method : {"constrained", "ml", "em"} or None
        Estimation method (``--method``).
    delta, eps : float or None
        Passband and subtide gap tolerances (``--delta``, ``--eps``).
    sacrificial : str or None
        Count, band spec or ``none`` (``--sacrificial``).
    sub_cross, sub_pass : float or None
        Subtide crossover and passband requirements in cpd (``--sub-cross``, ``--sub-pass``).
    post_filter : str or None
        Subtide post-filter cutoff such as ``18h``, or ``none`` (``--post-filter``).
    fix : tuple of str
        Repeatable ``NAME=VALUE`` items holding a parameter fixed (``--fix``); use
        ``sigma2_kappa:BAND=VALUE`` for one band's variance.
    start_fit : path or None
        Fit YAML to start from (``--start``).
    maxiter : int or None
        Optimiser iteration limit (``--maxiter``).
    workers : int or None
        Processes for gradients (``--workers``).
    decomposition : path or None
        Also write the decomposition CSV here (``--decomposition``).
    verbose : bool
        Per-iteration optimiser output (``--verbose``).
    """
    kw: dict = {}
    if config is not None:
        import yaml
        kw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        unknown = set(kw) - set(_FIT_KEYS)
        if unknown:
            raise click.BadParameter(f"Unknown config keys {sorted(unknown)}; allowed {list(_FIT_KEYS)}.",
                                     param_hint="--config")
    req = dict(kw.get("requirements") or {})
    flags = dict(delta=delta, eps=eps, sub_cross=sub_cross, sub_pass=sub_pass)
    req.update({k: v for k, v in flags.items() if v is not None})
    if sacrificial is not None:
        s = _none_or(sacrificial)
        req["sacrificial"] = int(s) if s is not None and s.isdigit() else s
    if post_filter is not None:
        req["post_filter"] = _none_or(post_filter)
    kw["requirements"] = req
    if fix:
        kw["fixed"] = {**(kw.get("fixed") or {}), **_parse_fix(fix)}
    over = dict(column=column, bands=bands, method=method, maxiter=maxiter,
                workers=workers, decomposition=decomposition,
                start=None if start_fit is None else str(start_fit))
    kw.update({k: v for k, v in over.items() if v is not None})
    if order is not None:
        kw["order"] = tuple(int(v) for v in order.split(","))
    fit_file(input, output, verbose=verbose, **kw)


@main.command("decompose")
@click.help_option("-h", "--help")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--fit", "fit_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Fit YAML.")
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Components CSV to write.")
@_series_options
@_logging_options
def decompose_cmd(input, fit_path, output, column):
    """Write the smoothed decomposition of INPUT under a stored fit.

    \f

    Parameters
    ----------
    input : path
        Input CSV (argument).
    fit_path : path
        Fit YAML (``--fit``).
    output : path
        Components CSV to write (``--output``).
    column : str or None
        Value column (``--column``).
    """
    decompose_file(input, fit_path, output, column=column)


@main.command("phasors")
@click.help_option("-h", "--help")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--fit", "fit_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Fit YAML.")
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Phasor CSV to write.")
@_series_options
@_logging_options
def phasors_cmd(input, fit_path, output, column):
    """Write smoothed top-pair phasors (<band>_c, <band>_s) of INPUT.

    \f

    Parameters
    ----------
    input : path
        Input CSV (argument).
    fit_path : path
        Fit YAML (``--fit``).
    output : path
        Phasor CSV to write (``--output``).
    column : str or None
        Value column (``--column``).
    """
    phasors_file(input, fit_path, output, column=column)


@main.command("ampphase")
@click.help_option("-h", "--help")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--fit", "fit_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Fit YAML.")
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Amplitude/phase CSV to write.")
@click.option("--epoch", default=DEFAULT_EPOCH, show_default=True, help="Time origin of the phase.")
@click.option("--ref", multiple=True, help="Reference line per band, e.g. D1=K1 (default: band model line). "
                                           "Repeatable.")
@_series_options
@_logging_options
def ampphase_cmd(input, fit_path, output, epoch, ref, column):
    """Write local cos/sin coefficients, amplitude and phase (deg, lag) of each band.

    \f

    Parameters
    ----------
    input : path
        Input CSV (argument).
    fit_path : path
        Fit YAML (``--fit``).
    output : path
        Amplitude/phase CSV to write (``--output``).
    epoch : str
        Time origin of the phase (``--epoch``).
    ref : tuple of str
        Repeatable ``BAND=REF`` items; ``REF`` is a number (cpd) or a tide name such as ``K1``
        (``--ref``). Empty uses each band's model line.
    column : str or None
        Value column (``--column``).
    """
    refs = {}
    for item in ref:
        band, val = item.split("=", 1)
        try:
            refs[band] = float(val)
        except ValueError:
            refs[band] = val
    ampphase_file(input, fit_path, output, column=column, epoch=epoch, ref=refs or None)


@main.command("gains")
@click.help_option("-h", "--help")
@click.argument("fit_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Gain table CSV to write.")
@click.option("--cpd-max", type=float, default=None, help="Highest frequency (default: Nyquist).")
@_logging_options
def gains_cmd(fit_path, output, cpd_max):
    """Write the steady-state gain table of a stored fit.

    \f

    Parameters
    ----------
    fit_path : path
        Fit YAML (argument).
    output : path
        Gain table CSV to write (``--output``).
    cpd_max : float or None
        Highest frequency in cpd; default just below Nyquist (``--cpd-max``).
    """
    gains_file(fit_path, output, cpd_max=cpd_max)


@main.command("crossval")
@click.help_option("-h", "--help")
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="YAML of crossval_file keyword arguments (rho and sigma2_zeta lists required).")
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Score table CSV to write.")
@click.option("--workers", type=int, default=None, help="Processes over folds (overrides config).")
@_logging_options
def crossval_cmd(input, config, output, workers):
    """Gridded (rho, sigma2_zeta) cross-validation of INPUT by withheld gaps.

    \f

    Parameters
    ----------
    input : path
        Input CSV (argument).
    config : path
        YAML of :func:`scha.cli.crossval_file` keyword arguments; ``rho`` and ``sigma2_zeta``
        lists are required and unknown keys raise ``click.BadParameter`` (``--config``).
    output : path
        Score table CSV to write (``--output``).
    workers : int or None
        Processes over folds; overrides the config (``--workers``).
    """
    import yaml
    kw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    unknown = set(kw) - set(_CV_KEYS)
    if unknown:
        raise click.BadParameter(f"Unknown config keys {sorted(unknown)}; allowed {list(_CV_KEYS)}.",
                                 param_hint="--config")
    if workers is not None:
        kw["workers"] = workers
    crossval_file(input, output, **kw)


if __name__ == "__main__":
    main()
