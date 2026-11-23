"""Fit SCHA to the Freeport 2014 discharge example with the library defaults.

Usage::

    python fit_freeport.py [--output-dir out] [--no-post-filter]

``freeport_2014_select.csv`` is a dms-format file (``datetime,value`` with a
``#`` YAML header): 18001 samples at 15 min (about six months) of USGS 11447650
discharge in m3/s, the record used in .github/docs/scha_design.md. Its start
time is nominal (see the header's ``time_note``), so absolute phases from this
example are nominal too.

Fits do not depend on the data units. The design notes worked in units of
100 m3/s; their variances are this example's divided by 1e4. Equivalent command
line::

    scha fit freeport_2014_select.csv -o out/fit.yaml --decomposition out/components.csv

The research scripts behind the design notes are in the paper repository
(scha-paper, legacy/freeport).
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

import scha
from scha.io import read_series
from scha.logging_config import configure_logging, resolve_loglevel

HERE = Path(__file__).resolve().parent
DATA = HERE / "freeport_2014_select.csv"
logger = logging.getLogger("scha.examples.fit_freeport")


def fit_freeport(output_dir: Path, post_filter: bool = True) -> scha.SCHAFit:
    """Fit the default SCHA(2,3) D1-D6 model and write the fit and its decomposition.

    Parameters
    ----------
    output_dir : Path
        Directory for ``fit.yaml`` and ``components.csv`` (m3/s).
    post_filter : bool
        Use the default 18 h subtide post-filter (False: none).

    Returns
    -------
    scha.SCHAFit
    """
    y = read_series(DATA)
    req = None if post_filter else {"post_filter": None}
    f = scha.fit(y, requirements=req)
    output_dir.mkdir(parents=True, exist_ok=True)
    f.to_yaml(output_dir / "fit.yaml")
    f.decompose().to_csv(output_dir / "components.csv", index_label="datetime")
    logger.info("\n%s", f.summary())
    return f


@click.command()
@click.help_option("-h", "--help")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path), default=HERE / "out",
              show_default=True, help="Directory for fit.yaml and components.csv.")
@click.option("--no-post-filter", is_flag=True, help="Fit without the 18 h subtide post-filter.")
@click.option("--logdir", type=click.Path(path_type=Path), default=None, help="Directory for a log file.")
@click.option("--debug", is_flag=True, help="Debug-level logging.")
@click.option("--quiet", "-q", is_flag=True, help="No progress messages on the console.")
def main(output_dir, no_post_filter, logdir, debug, quiet):
    """Fit SCHA to the Freeport 2014 discharge example."""
    level, console = resolve_loglevel(debug=debug, quiet=quiet)
    configure_logging(package_name="scha", level=level, console=console, logdir=logdir,
                      logfile_prefix="fit_freeport")
    fit_freeport(output_dir=output_dir, post_filter=not no_post_filter)


if __name__ == "__main__":
    main()
