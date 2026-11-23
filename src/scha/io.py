"""I/O helpers for SCHA parameter YAML and time series.

Provides strict parsing/validation for frequency specification and conversion to
radians-per-sample required by :class:`scha.params.SCHAParams` and
:func:`scha.state_space.build_scha_state_space`.

Behavior choices:

- ``freqs`` in YAML MUST be a mapping label -> (alias or numeric radians/hour).
- ``sigma2_kappa`` may be a scalar or list. If scalar or single-item list -> replicate to
  match number of freqs. If provided as a mapping, all keys must be present for the declared
  freqs (no silent fills; errors raised).
- The time step comes from the data index, never from the YAML.
  Contract: a naive ``pd.DatetimeIndex`` with ``freq`` set (see :mod:`scha.timestep`).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd

from .params import SCHAParams
from .tide_registry import rph
from .timestep import lambdas_from_rph, step_hours

try:
    import yaml
except Exception:  # pragma: no cover - YAML tests run only in CI with PyYAML
    yaml = None


def validate_series(ts: pd.Series) -> pd.Series:
    """Check that a series meets the scha input contract and set its ``freq``.

    The contract: a naive, increasing, duplicate-free DatetimeIndex on a regular
    grid, with gaps as NaN rows (not missing rows), and float values.

    Parameters
    ----------
    ts : pd.Series
        Candidate series.

    Returns
    -------
    pd.Series
        The same values as float, with ``index.freq`` set.

    Raises
    ------
    ValueError
        Naming the first violated condition.
    """
    idx = ts.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(f"Index must be datetimes; got {type(idx).__name__}.")
    if idx.tz is not None:
        raise ValueError("Index must be time-zone naive.")
    if idx.size < 3:
        raise ValueError("Need at least 3 samples.")
    if idx.has_duplicates:
        raise ValueError(f"Duplicate timestamps, first at {idx[idx.duplicated()][0]}.")
    if not idx.is_monotonic_increasing:
        raise ValueError("Timestamps are not in increasing order.")
    freq = idx.freq or pd.infer_freq(idx)
    if freq is None:
        d = pd.Series(idx).diff().dropna()
        step = d.mode().iloc[0]
        bad = idx[1:][(d != step).to_numpy()]
        raise ValueError(f"Index is not regular: most common step {step}, first other step ending at {bad[0]}. "
                         "Gaps must be NaN rows on the regular grid.")
    out = ts.astype(float)
    out.index = pd.DatetimeIndex(idx, freq=freq)
    step_hours(out.index)
    return out


def read_series(path, column: Optional[str] = None) -> pd.Series:
    """Read a single time series from a CSV with a datetime first column.

    Reads ``pd.read_csv(path, index_col=0, parse_dates=True, comment="#")`` (so
    dms_datastore files with their ``#`` YAML header work) and checks the result
    with :func:`validate_series`.

    Parameters
    ----------
    path : str or path-like
        CSV file.
    column : str, optional
        Value column; required when the file has more than one.

    Returns
    -------
    pd.Series
        Float series with ``index.freq`` set and NaN gaps.

    Raises
    ------
    ValueError
        If ``column`` is None and the file has more than one value column, if ``column`` is not in
        the file, or if the series fails :func:`validate_series`.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True, comment="#")
    if column is None:
        if df.shape[1] != 1:
            raise ValueError(f"{path} has columns {list(df.columns)}; choose one with column=.")
        column = df.columns[0]
    elif column not in df.columns:
        raise ValueError(f"Column {column!r} not in {list(df.columns)}.")
    ts = pd.to_numeric(df[column], errors="coerce")
    return validate_series(ts)

def _resolve_freq_value(entry: Union[str, float, int]) -> float:
    """Resolve a YAML frequency value to radians per hour.

    Parameters
    ----------
    entry : str, float or int
        A number (radians per hour), a tide name or alias known to
        :func:`scha.tide_registry.rph`, or a string holding a number in radians per hour.

    Returns
    -------
    float
        Frequency in radians per hour.

    Raises
    ------
    ValueError
        If a string is neither a known tide name nor numeric.
    """
    if isinstance(entry, (float, int)):
        return float(entry)
    s = str(entry)
    try:
        return float(rph(s))
    except KeyError:
        try:
            return float(s)
        except Exception:
            raise ValueError(
                f"Frequency value '{entry}' is not a known alias and not numeric radians/hour."
            )

def freqs_from_config(freqs_cfg: Dict) -> Tuple[List[float], List[str]]:
    """Return (radians_per_hour, labels) in declared order.

    Parameters
    ----------
    freqs_cfg : dict
        Mapping label -> tide name or alias, or numeric frequency in radians per hour.

    Returns
    -------
    omega_rph : list of float
        Frequencies in radians per hour.
    labels : list of str
        Labels, in the same order as ``omega_rph``.

    Raises
    ------
    ValueError
        If ``freqs_cfg`` is not a dict, or a value cannot be resolved to a frequency.
    """
    if not isinstance(freqs_cfg, dict):
        raise ValueError("'freqs' must be a mapping (dict) so output labels are explicit.")
    omega_rph = []
    labels = []
    for label, val in freqs_cfg.items():
        omega_rph.append(_resolve_freq_value(val))
        labels.append(label)
    return omega_rph, labels


def normalize_sigma2_kappa(s2k_cfg: Union[float, List[float], Dict[str, float]], n_freq: int, labels: List[str]) -> List[float]:
    """Normalize sigma2_kappa to a list of length n_freq.

    Rules (strict):

    - If s2k_cfg is scalar or single-element list, replicate to n_freq (emit a warning).
    - If s2k_cfg is list of length==n_freq, accept.
    - If s2k_cfg is list of other length -> raise ValueError.
    - If s2k_cfg is dict: keys must exactly match declared labels; map values to labels order;
      otherwise raise ValueError.

    Parameters
    ----------
    s2k_cfg : float, list of float or dict
        The ``sigma2_kappa`` entry of the configuration.
    n_freq : int
        Number of declared frequencies.
    labels : list of str
        Frequency labels in declared order.

    Returns
    -------
    list of float
        Innovation variance per cycle, in the order of ``labels``.

    Raises
    ------
    ValueError
        If a mapping lacks keys for ``labels`` or has unknown keys, a list has a length other
        than 1 or ``n_freq``, or ``s2k_cfg`` is of another type.

    Warns
    -----
    UserWarning
        When a scalar or single-element list is replicated to ``n_freq`` entries.
    """
    if isinstance(s2k_cfg, dict):
        # require exact keys match labels
        missing = [lab for lab in labels if lab not in s2k_cfg]
        extra = [k for k in s2k_cfg.keys() if k not in labels]
        if missing:
            raise ValueError(f"sigma2_kappa mapping missing keys for freqs: {missing}")
        if extra:
            raise ValueError(f"sigma2_kappa mapping contains unknown keys: {extra}")
        return [float(s2k_cfg[lab]) for lab in labels]

    if isinstance(s2k_cfg, (int, float)):
        warnings.warn("sigma2_kappa provided as scalar; replicating to match number of frequencies", UserWarning)
        return [float(s2k_cfg)] * n_freq

    if isinstance(s2k_cfg, list):
        if len(s2k_cfg) == 1:
            warnings.warn("sigma2_kappa provided as single-element list; replicating to match number of frequencies", UserWarning)
            return [float(s2k_cfg[0])] * n_freq
        if len(s2k_cfg) == n_freq:
            return [float(x) for x in s2k_cfg]
        raise ValueError(f"sigma2_kappa list length {len(s2k_cfg)} does not match number of frequencies {n_freq}")

    raise ValueError("Unrecognized sigma2_kappa configuration")


def load_scha_params_from_yaml(yaml_path: str, *, index) -> Tuple[SCHAParams, Dict]:
    """Load params from YAML and return (SCHAParams, metadata).

    Parameters
    ----------
    yaml_path : str or path-like
        Parameter YAML. ``freqs`` maps labels to tide aliases or radians per hour. Optional
        keys and defaults: ``m`` (2), ``n`` (4), ``rho`` (0.9935), ``sigma2_eps`` (1e-3),
        ``sigma2_zeta`` (1e-10). ``sigma2_kappa`` is required whenever ``freqs`` is non-empty
        (see :func:`normalize_sigma2_kappa`). A ``dt`` key is rejected.
    index : pd.DatetimeIndex
        Index of the data the parameters will be used with (keyword-only). Its ``freq`` sets
        the time step used to convert ``freqs`` to radians per sample.

    Returns
    -------
    params : scha.params.SCHAParams
        Parameters with ``lambdas`` in radians per sample.
    metadata : dict
        ``labels`` (cycle labels in declared order) and ``dt_hours`` (time step).

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    ValueError
        If the YAML contains ``dt``, ``freqs`` is invalid, ``sigma2_kappa`` is missing or
        invalid, a frequency is outside (0, pi) radians per sample for the step, or ``index``
        is not a regular naive DatetimeIndex (see :func:`scha.timestep.step_hours`).
    """
    if yaml is None:
        raise ImportError("PyYAML required to read YAML; install with 'pip install pyyaml'")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}

    # Keep the YAML surface focused: time step comes from the data index.
    if "dt" in cfg:
        raise ValueError("Field 'dt' is not part of the SCHA YAML interface; dt is inferred from the data index.")

    dt_hours = step_hours(index)
    freqs_cfg = cfg.get('freqs', None)
    if freqs_cfg is None:
        lambdas, labels = [], []
    else:
        omega_rph_list, labels = freqs_from_config(freqs_cfg)
        lambdas = [float(v) for v in lambdas_from_rph(omega_rph_list, dt_hours)]


    m = int(cfg.get('m', 2))
    n = int(cfg.get('n', 4))
    rho = float(cfg.get('rho', 0.9935))
    sigma2_eps = float(cfg.get('sigma2_eps', 1e-3))
    sigma2_zeta = float(cfg.get('sigma2_zeta', 1e-10))

    s2k_cfg = cfg.get('sigma2_kappa', None)
    n_freq = len(lambdas)
    if s2k_cfg is None and n_freq > 0:
        raise ValueError("sigma2_kappa must be provided in YAML and match frequencies (no defaults)")
    sigma2_kappa = normalize_sigma2_kappa(s2k_cfg, n_freq, labels) if n_freq > 0 else []

    params = SCHAParams(m=m, n=n, rho=rho, lambdas=lambdas, sigma2_eps=sigma2_eps, sigma2_zeta=sigma2_zeta, sigma2_kappa=sigma2_kappa)
    metadata = {'labels': labels, 'dt_hours': dt_hours} 
    return params, metadata
