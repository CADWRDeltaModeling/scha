"""I/O helpers for SCHA parameter YAML and time series.

Provides strict parsing/validation for frequency specification, dt parsing, and
conversion to radians-per-sample required by `SCHAParams` and `build_scha_state_space`.

Behavior choices:
- `freqs` in YAML may be a list (aliases or numeric cpd) or a mapping label -> (alias or numeric cpd).
- `sigma2_kappa` may be a scalar or list. If scalar or single-item list -> replicate to match number of freqs.
  If provided as a mapping, all keys must be present for the declared freqs (no silent fills; errors raised).
- `dt` may be a string like `1h`, `15min`, or a numeric days-per-sample value. If omitted, callers should pass
  a datetime index to `load_scha_params_from_yaml` for inference.
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Union, Optional
import warnings

import numpy as np

from .params import SCHAParams
from .tide_registry import rph, select, all_names

try:
    import yaml
except Exception:  # pragma: no cover - YAML tests run only in CI with PyYAML
    yaml = None


def parse_dt(dt_spec: Union[str, float, None]) -> float:
    """Parse dt spec and return days-per-sample (float).

    Accepts:
    - None: returns None (caller may infer dt from data)
    - numeric: interpreted as days-per-sample
    - strings: '1h', '15min', '30s', '1d' etc.

    Raises ValueError for unrecognized formats.
    """
    if dt_spec is None:
        return None
    if isinstance(dt_spec, (int, float)):
        return float(dt_spec)
    s = str(dt_spec).strip().lower()
    # simple suffix parsing
    if s.endswith("h"):
        val = float(s[:-1])
        return val / 24.0
    if s.endswith("min"):
        val = float(s[:-3])
        return val / (24.0 * 60.0)
    if s.endswith("s"):
        val = float(s[:-1])
        return val / (24.0 * 3600.0)
    if s.endswith("d"):
        val = float(s[:-1])
        return val
    raise ValueError(f"Unrecognized dt specification: {dt_spec}")


def _resolve_freq_entry(entry: Union[str, float, int]) -> Tuple[float, str]:
    """Resolve a single YAML freq entry to (cycles_per_day, label).

    - If entry is a string and matches a registered alias in tide_registry, return its
      radians-per-hour converted to cycles-per-day and use the alias as label.
    - If entry is a string that looks like a number, parse as numeric cycles/day.
    - If entry is numeric, treat as cycles/day.
    """
    if isinstance(entry, (float, int)):
        return float(entry), str(entry)
    s = str(entry)
    # first try as a known name
    try:
        # rph returns radians per hour; convert to cycles per day: (radians/hour)*(24)/(2*pi)
        r = rph(s)
        cpd = float(r) * 24.0 / (2.0 * math.pi)
        return cpd, s
    except KeyError:
        # try numeric
        try:
            val = float(s)
            return val, s
        except Exception:
            raise ValueError(f"Frequency '{entry}' is not a known alias and not numeric cycles-per-day.")


def freqs_from_config(freqs_cfg: Union[List, Dict]) -> Tuple[List[float], List[str]]:
    """Return (list_of_cpd, list_of_labels) in declared order.

    Accepts either:
    - list: entries are alias names or numeric cpd
    - dict: keys are labels, values are alias names or numeric cpd

    In the dict case the insertion order of keys is preserved (Py3.7+).
    """
    if isinstance(freqs_cfg, dict):
        cpd_list = []
        labels = []
        for label, val in freqs_cfg.items():
            cpd, _ = _resolve_freq_entry(val)
            cpd_list.append(cpd)
            labels.append(label)
        return cpd_list, labels
    elif isinstance(freqs_cfg, list):
        cpd_list = []
        labels = []
        for entry in freqs_cfg:
            cpd, lab = _resolve_freq_entry(entry)
            cpd_list.append(cpd)
            labels.append(lab)
        return cpd_list, labels
    else:
        raise ValueError("'freqs' must be a list or a mapping in the YAML file")


def normalize_sigma2_kappa(s2k_cfg: Union[float, List[float], Dict[str, float]], n_freq: int, labels: List[str]) -> List[float]:
    """Normalize sigma2_kappa to a list of length n_freq.

    Rules (strict):
    - If s2k_cfg is scalar or single-element list, replicate to n_freq (emit a warning).
    - If s2k_cfg is list of length==n_freq, accept.
    - If s2k_cfg is list of other length -> raise ValueError.
    - If s2k_cfg is dict: keys must exactly match declared labels; map values to labels order; otherwise raise ValueError.
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


def load_scha_params_from_yaml(yaml_path: str, *, dt_spec: Optional[Union[str, float]] = None, infer_dt_from_index: bool = False, index=None) -> Tuple[SCHAParams, Dict]:
    """Load params from YAML and return (SCHAParams, metadata).

    - `dt_spec`: optional explicit dt string or numeric days-per-sample.
    - If `infer_dt_from_index` is True, `index` must be provided (DatetimeIndex) to infer dt and overrides dt_spec.

    Returns a tuple containing the constructed `SCHAParams` and a metadata dict with `labels` and `dt_days`.

    Raises ValueError on inconsistent specifications (no silent defaults).
    """
    if yaml is None:
        raise ImportError("PyYAML required to read YAML; install with 'pip install pyyaml'")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # freqs
    freqs_cfg = cfg.get('freqs', None)
    if freqs_cfg is None:
        # no cycles
        cpd_list = []
        labels = []
    else:
        cpd_list, labels = freqs_from_config(freqs_cfg)

    # dt handling
    dt_days = None
    if infer_dt_from_index:
        if index is None:
            raise ValueError("infer_dt_from_index=True requires a datetime index to be provided")
        # infer median diff in days (robust conversion to int64 nanoseconds)
        if hasattr(index, 'asi8'):
            arr_ns = index.asi8
        else:
            # try to coerce to datetime first (allows string-like datetime indices)
            import pandas as pd
            idx_dt = pd.to_datetime(index, errors='coerce')
            if not idx_dt.isna().all():
                arr_ns = idx_dt.astype('int64')
            else:
                raise ValueError("infer_dt_from_index=True requires a datetime-like index; could not parse provided index")
        diffs_ns = np.diff(arr_ns)
        if diffs_ns.size == 0:
            raise ValueError("Index too short to infer dt")
        median_ns = int(np.median(diffs_ns))
        dt_days = median_ns / 1e9 / 86400.0
    else:
        dt_days = parse_dt(cfg.get('dt', dt_spec))

    # convert cpd -> radians per sample (expected by SCHAParams lambdas)
    if dt_days is None and cpd_list:
        raise ValueError("dt not provided and cannot be inferred; required when freqs are specified")
    lambdas = [ (2.0 * math.pi * cpd * dt_days) for cpd in cpd_list ]

    m = int(cfg.get('m', 2))
    n = int(cfg.get('n', 4))
    rho = float(cfg.get('rho', 0.9935))
    sigma2_eps = float(cfg.get('sigma2_eps', 1e-3))
    sigma2_zeta = float(cfg.get('sigma2_zeta', 1e-10))

    s2k_cfg = cfg.get('sigma2_kappa', None)
    n_freq = len(lambdas)
    if s2k_cfg is None:
        raise ValueError("sigma2_kappa must be provided in YAML and match frequencies (no defaults)")
    sigma2_kappa = normalize_sigma2_kappa(s2k_cfg, n_freq, labels)

    params = SCHAParams(m=m, n=n, rho=rho, lambdas=lambdas, sigma2_eps=sigma2_eps, sigma2_zeta=sigma2_zeta, sigma2_kappa=sigma2_kappa)

    metadata = {'labels': labels, 'dt_days': dt_days}
    return params, metadata
