"""Turnkey SCHA fitting and the fit result object.

Typical use::

    import scha
    fit = scha.fit(y)                 # y: pd.Series, naive DatetimeIndex with freq, NaN gaps
    print(fit.summary())
    dec = fit.decompose()             # observed, fitted, subtide, D1.., residual
    fit.to_yaml("fit.yaml")
    fit2 = scha.load_fit("fit.yaml")

Parameters are native to the sampling step they were fitted at (``fit_freq``).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .bands import BandSpec, band_cpd, parse_bands
from .fastll import (diffuse_variance, scha_innovations, scha_loglik_diffuse, scha_signal_moments,
                     scha_smoothed_states)
from .gains import cosine_lanczos_response, crossover, gains
from .mlfit import fit_ml
from .params import SCHAParams
from .requirements import CompiledRequirements, Requirements
from .start import auto_start
from .timestep import freq_hours, lambdas_from_cpd, step_hours

METHODS = ("constrained", "ml", "em")
_SCHEMA = 1
logger = logging.getLogger("scha.fit")


def _as_series(y) -> pd.Series:
    """Return ``y`` as a validated float Series (see :func:`scha.io.validate_series`).

    A single-column DataFrame is reduced to its column.

    Raises
    ------
    ValueError
        If ``y`` is a DataFrame with other than one column, is not a Series or DataFrame, or
        fails validation.
    """
    from .io import validate_series
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Pass a Series or a single-column DataFrame.")
        y = y.iloc[:, 0]
    if not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series with a regular DatetimeIndex.")
    return validate_series(y)


def _freqstr(index: pd.DatetimeIndex) -> str:
    """Return the pandas frequency string of ``index.freq`` (for example ``"15min"``)."""
    return pd.tseries.frequencies.to_offset(index.freq).freqstr


def _resolve_fixed(fixed: Optional[dict], bands: Sequence[str]) -> dict:
    """Map band labels in a fixed sigma2_kappa to cycle indices.

    Parameters
    ----------
    fixed : dict or None
        Fixed parameters as passed to :func:`scha.fit.fit`. A ``sigma2_kappa`` mapping may be keyed
        by band label or cycle index; a list must have one value per band.
    bands : sequence of str
        Band labels, one per cycle.

    Returns
    -------
    dict
        Copy of ``fixed`` whose ``sigma2_kappa`` entry, if present, is a dict cycle index ->
        float. Empty if ``fixed`` is empty or None.

    Raises
    ------
    ValueError
        If a fixed ``sigma2_kappa`` list does not have one value per band.
    """
    if not fixed:
        return {}
    out = dict(fixed)
    fk = out.get("sigma2_kappa")
    if isinstance(fk, dict):
        out["sigma2_kappa"] = {bands.index(k) if isinstance(k, str) else int(k): float(v) for k, v in fk.items()}
    elif fk is not None:
        fk = list(fk)
        if len(fk) != len(bands):
            raise ValueError("A fixed sigma2_kappa list must have one value per band.")
        out["sigma2_kappa"] = {c: float(v) for c, v in enumerate(fk)}
    return out


@dataclass
class SCHAFit:
    """Result of :func:`scha.fit.fit` (or :func:`scha.fit.load_fit`).

    Attributes
    ----------
    params : scha.params.SCHAParams
        Fitted parameters, native to ``fit_freq`` (lambdas in radians per sample).
    bands : list of str
        Band labels, one per cycle.
    fit_freq : str
        Sampling step the parameters belong to, e.g. ``"15min"``.
    method : str
        ``"constrained"``, ``"ml"`` or ``"em"``.
    requirements : scha.requirements.Requirements
        Gain requirements (imposed for ``"constrained"``, reported otherwise).
    fixed : dict
        Parameters held fixed during the fit.
    diagnostics : dict
        Likelihood, convergence, requirement report, crossover, data summary.
    y : pd.Series or None
        Data the model was fitted to (None after :func:`scha.fit.load_fit`).
    """

    params: SCHAParams
    bands: list
    fit_freq: str
    method: str
    requirements: Requirements
    fixed: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    y: Optional[pd.Series] = None

    @property
    def step_h(self) -> float:
        """Sampling step of ``fit_freq`` in hours."""
        return freq_hours(self.fit_freq)

    @property
    def cpd(self) -> list:
        """Model line of each band in cycles per day."""
        return [band_cpd(b) for b in self.bands]

    def compiled_requirements(self) -> CompiledRequirements:
        """Return the requirements resolved for this fit's bands and step.

        Returns
        -------
        scha.requirements.CompiledRequirements
            Result of ``CompiledRequirements.build`` for ``requirements``, ``bands`` and ``step_h``.
        """
        return CompiledRequirements.build(self.requirements, self.bands, self.step_h)

    def _data(self, y) -> pd.Series:
        """Return the data to use: ``self.y`` if ``y`` is None, else the validated ``y``.

        Raises
        ------
        ValueError
            If ``y`` is None and no data is attached, or the step of ``y`` differs from
            ``step_h``.
        """
        if y is None:
            if self.y is None:
                raise ValueError("No data attached to this fit; pass y.")
            return self.y
        y = _as_series(y)
        if not np.isclose(step_hours(y.index), self.step_h):
            raise ValueError(f"Data step {_freqstr(y.index)} differs from the fit step {self.fit_freq}.")
        return y

    # --- products -------------------------------------------------------------------------------

    @staticmethod
    def _centred(y: pd.Series) -> tuple[np.ndarray, float, float]:
        """Return ``(y - mean, mean, trend prior variance)``: the level prior is centred on the data mean.

        Parameters
        ----------
        y : pd.Series
            Observations; NaN marks gaps.

        Returns
        -------
        yc : np.ndarray
            ``y`` minus its NaN-ignoring mean.
        c : float
            The mean of ``y`` ignoring NaN.
        sd : float
            Trend prior variance, :func:`scha.fastll.diffuse_variance` of ``yc``.
        """
        yv = y.to_numpy()
        c = float(np.nanmean(yv))
        yc = yv - c
        return yc, c, diffuse_variance(yc)

    def states(self, y=None) -> np.ndarray:
        """Return the smoothed state means.

        Parameters
        ----------
        y : pd.Series, optional
            Data at the fit's step; default the data the model was fitted to.

        Returns
        -------
        np.ndarray
            Shape (n, dim), unscaled state coordinates; column 0 is the trend level.

        Raises
        ------
        ValueError
            If ``y`` is None and the fit has no attached data, or the step of ``y`` differs from
            the fit step.
        """
        y = self._data(y)
        p = self.params
        yc, c, sd = self._centred(y)
        x = scha_smoothed_states(p, yc, sigma2_diffuse=sd, scale_rho=p.rho)
        x[:, 0] += c  # trend level
        return x

    def signal_moments(self, y=None) -> pd.DataFrame:
        """Return the filter-predicted (one-sided) and smoothed signal mean and standard deviation.

        Parameters
        ----------
        y : pd.Series, optional
            Data at the fit's step; default the data the model was fitted to.

        Returns
        -------
        pd.DataFrame
            ``pred_mean``, ``pred_sd``: given data before t (inside a gap, the
            real-time forecast). ``smooth_mean``, ``smooth_sd``: given all data.
            These are moments of the signal; ``obs_pred_sd`` and ``obs_smooth_sd``
            add sigma2_eps for an observation.

        Raises
        ------
        ValueError
            If ``y`` is None and the fit has no attached data, or the step of ``y`` differs from
            the fit step.
        """
        y = self._data(y)
        p = self.params
        yc, c, sd = self._centred(y)
        mo = scha_signal_moments(p, yc, sigma2_diffuse=sd, scale_rho=p.rho)
        eps = p.sigma2_eps
        return pd.DataFrame({"pred_mean": mo["pred_mean"] + c, "pred_sd": np.sqrt(np.maximum(mo["pred_var"], 0.0)),
                             "smooth_mean": mo["smooth_mean"] + c,
                             "smooth_sd": np.sqrt(np.maximum(mo["smooth_var"], 0.0)),
                             "obs_pred_sd": np.sqrt(np.maximum(mo["pred_var"], 0.0) + eps),
                             "obs_smooth_sd": np.sqrt(np.maximum(mo["smooth_var"], 0.0) + eps)}, index=y.index)

    def decompose(self, y=None) -> pd.DataFrame:
        """Return the smoothed decomposition.

        Parameters
        ----------
        y : pd.Series, optional
            Data at the fit's step; default the data the model was fitted to.

        Returns
        -------
        pd.DataFrame
            ``observed``, ``fitted`` (smoothed signal: raw subtide plus all bands),
            ``subtide``, ``subtide_raw`` (only with a post-filter), one column per
            band, and ``residual`` = observed - subtide - bands. With a post-filter,
            ``subtide`` is the post-filtered subtide (NaN near the ends, as for the
            classical filter) and the remainder of the raw subtide is in the residual.

        Raises
        ------
        ValueError
            If ``y`` is None and the fit has no attached data, or the step of ``y`` differs from
            the fit step.
        """
        y = self._data(y)
        x = self.states(y)
        p = self.params
        sub_raw = x[:, 0]
        cyc = np.column_stack([x[:, p.m + c * 2 * p.n] for c in range(len(p.lambdas))])
        out = pd.DataFrame(index=y.index)
        out["observed"] = y.to_numpy()
        out["fitted"] = sub_raw + cyc.sum(axis=1)
        pf = self.requirements.post_filter_hours()
        if pf is not None:
            from vtools.functions.filter import cosine_lanczos
            out["subtide"] = cosine_lanczos(pd.Series(sub_raw, index=y.index), pd.Timedelta(hours=pf)).to_numpy()
            out["subtide_raw"] = sub_raw
        else:
            out["subtide"] = sub_raw
        for c, b in enumerate(self.bands):
            out[b] = cyc[:, c]
        out["residual"] = out["observed"] - out["subtide"] - cyc.sum(axis=1)
        return out

    def fill(self, y=None) -> pd.Series:
        """Return the observed values with gaps filled by the smoothed signal.

        Parameters
        ----------
        y : pd.Series, optional
            Data at the fit's step; default the data the model was fitted to.

        Returns
        -------
        pd.Series
            ``y`` where observed, the smoothed signal (``decompose()["fitted"]``) at NaN.

        Raises
        ------
        ValueError
            If ``y`` is None and the fit has no attached data, or the step of ``y`` differs from
            the fit step.
        """
        y = self._data(y)
        dec = self.decompose(y)
        return y.where(y.notna(), dec["fitted"]).rename(y.name)

    def phasors(self, y=None) -> pd.DataFrame:
        """Return the smoothed top-pair states of each band.

        Parameters
        ----------
        y : pd.Series, optional
            Data at the fit's step; default the data the model was fitted to.

        Returns
        -------
        pd.DataFrame
            ``<band>_c`` (psi, the band's contribution to y) and ``<band>_s`` (psi*).
            With z = psi - i psi*, z rotates forward at the band frequency and
            z ~ A exp(i(w t - phi)). See :func:`scha.phase.amplitude_phase`.

        Raises
        ------
        ValueError
            If ``y`` is None and the fit has no attached data, or the step of ``y`` differs from
            the fit step.
        """
        y = self._data(y)
        x = self.states(y)
        p = self.params
        cols = {}
        for c, b in enumerate(self.bands):
            j = p.m + c * 2 * p.n
            cols[f"{b}_c"] = x[:, j]
            cols[f"{b}_s"] = x[:, j + 1]
        return pd.DataFrame(cols, index=y.index)

    def innovations(self, y=None) -> pd.Series:
        """Return the standardized one-step prediction errors v_t / sqrt(F_t).

        Parameters
        ----------
        y : pd.Series, optional
            Data at the fit's step; default the data the model was fitted to.

        Returns
        -------
        pd.Series
            Innovations, NaN at gaps; white with unit variance if the model is adequate.

        Raises
        ------
        ValueError
            If ``y`` is None and the fit has no attached data, or the step of ``y`` differs from
            the fit step.
        """
        y = self._data(y)
        p = self.params
        yc, _, sd = self._centred(y)
        e = scha_innovations(p, yc, sigma2_diffuse=sd, scale_rho=p.rho)
        return pd.Series(e, index=y.index, name="innovation")

    def gains(self, cpd=None) -> pd.DataFrame:
        """Return the steady-state smoother gains by frequency.

        Parameters
        ----------
        cpd : array_like, optional
            Frequencies in cycles per day; default 2000 points up to Nyquist.

        Returns
        -------
        pd.DataFrame
            Indexed by cpd: ``subtide``, one column per band, ``noise``; with a
            post-filter also ``subtide_filtered`` (subtide gain times the post-filter
            response).
        """
        if cpd is None:
            cpd = np.linspace(0.01, 12.0 / self.step_h - 0.01, 2000)
        cpd = np.asarray(cpd, dtype=float)
        g = gains(self.params, cpd, self.step_h)
        out = pd.DataFrame(g.T, index=pd.Index(cpd, name="cpd"), columns=["subtide"] + list(self.bands))
        out["noise"] = 1.0 - g.sum(axis=0)
        pf = self.requirements.post_filter_hours()
        if pf is not None:
            out["subtide_filtered"] = out["subtide"] * cosine_lanczos_response(pf, cpd, self.step_h)
        return out

    def crossover(self) -> float:
        """Return the subtide/D1 gain crossover.

        Returns
        -------
        float
            Lowest frequency (cpd) at which the D1 gain reaches the subtide gain
            (NaN if there is none).
        """
        return crossover(self.params, self.step_h)

    # --- reporting and persistence ---------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary: data, likelihood, parameters, crossover, requirements.

        Returns
        -------
        str
        """
        p, d = self.params, self.diagnostics
        lines = [f"SCHA({p.m},{p.n}) fit, method={self.method}, step={self.fit_freq}, bands={','.join(self.bands)}"]
        if "n_obs" in d:
            lines.append(f"  data: {d.get('start')} to {d.get('end')}, n={d['n']}, observed={d['n_obs']} "
                         f"({100 * d['gap_fraction']:.1f}% missing)")
        lines.append(f"  loglik={d.get('loglik', float('nan')):.2f}  success={d.get('success')}  "
                     f"{d.get('message', '')}")
        lines.append(f"  rho={p.rho:.6f}  sigma2_eps={p.sigma2_eps:.4e}  sigma2_zeta={p.sigma2_zeta:.4e}")
        lines.append("  sigma2_kappa: " + "  ".join(f"{b}={k:.3e}" for b, k in zip(self.bands, p.sigma2_kappa)))
        if self.fixed:
            lines.append(f"  fixed: {self.fixed}")
        lines.append(f"  subtide/D1 crossover: {self.crossover():.3f} cpd")
        lines.append("  requirements: " + self.compiled_requirements().report_line(p))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Return the fit as a plain dict (the YAML schema).

        Returns
        -------
        dict
            Keys ``schema``, ``scha_version``, ``method``, ``fit_freq``, ``model``
            (m, n, bands with cpd and lambda), ``params`` (native to ``fit_freq``),
            ``fixed``, ``requirements`` and ``diagnostics``.
        """
        from . import __version__
        p = self.params
        fixed = dict(self.fixed)
        if isinstance(fixed.get("sigma2_kappa"), dict):
            fixed["sigma2_kappa"] = {self.bands[int(c)]: float(v) for c, v in fixed["sigma2_kappa"].items()}
        return {
            "schema": _SCHEMA,
            "scha_version": str(__version__),
            "method": self.method,
            "fit_freq": self.fit_freq,
            "model": {"m": p.m, "n": p.n,
                      "bands": {b: {"cpd": float(c), "lambda": float(l)}
                                for b, c, l in zip(self.bands, self.cpd, p.lambdas)}},
            "params": {"rho": float(p.rho), "sigma2_eps": float(p.sigma2_eps), "sigma2_zeta": float(p.sigma2_zeta),
                       "sigma2_kappa": {b: float(k) for b, k in zip(self.bands, p.sigma2_kappa)}},
            "fixed": {k: (float(v) if not isinstance(v, dict) else v) for k, v in fixed.items()},
            "requirements": self.requirements.to_dict(),
            "diagnostics": self.diagnostics,
        }

    def to_yaml(self, path) -> None:
        """Write the fit to YAML (see :meth:`to_dict`; read back with :func:`load_fit`).

        Parameters
        ----------
        path : str or path-like
            Output file.
        """
        import yaml
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def _from_dict(d: dict) -> SCHAFit:
    """Build a :class:`scha.fit.SCHAFit` without attached data from the dict of :meth:`SCHAFit.to_dict`.

    Raises
    ------
    ValueError
        If the schema version of ``d`` is not supported.
    """
    if d.get("schema") != _SCHEMA:
        raise ValueError(f"Unsupported fit file schema {d.get('schema')!r}.")
    bands = list(d["model"]["bands"])
    step_h = freq_hours(d["fit_freq"])
    lam = [float(v) for v in lambdas_from_cpd([band_cpd(b) for b in bands], step_h)]
    pr = d["params"]
    p = SCHAParams(m=int(d["model"]["m"]), n=int(d["model"]["n"]), rho=float(pr["rho"]), lambdas=lam,
                   sigma2_eps=float(pr["sigma2_eps"]), sigma2_zeta=float(pr["sigma2_zeta"]),
                   sigma2_kappa=[float(pr["sigma2_kappa"][b]) for b in bands])
    return SCHAFit(params=p, bands=bands, fit_freq=d["fit_freq"], method=d["method"],
                   requirements=Requirements.from_dict(d.get("requirements")),
                   fixed=_resolve_fixed(d.get("fixed"), bands), diagnostics=dict(d.get("diagnostics") or {}))


def load_fit(path) -> SCHAFit:
    """Read a fit written by :meth:`scha.fit.SCHAFit.to_yaml`.

    Parameters
    ----------
    path : str or path-like
        Fit YAML.

    Returns
    -------
    scha.fit.SCHAFit
        Fit without attached data (pass ``y`` to its methods).

    Raises
    ------
    ValueError
        If the file's schema version is not supported.
    """
    import yaml
    return _from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _start_params(start, y: pd.Series, bands: list, step_h: float, m: int, n: int, lam: list) -> SCHAParams:
    """Resolve ``start`` (see :func:`scha.fit.fit`) to starting :class:`scha.params.SCHAParams`.

    None or ``"auto"`` calls :func:`scha.start.auto_start`; a path is loaded with
    :func:`scha.fit.load_fit`; a fit contributes its parameters.

    Parameters
    ----------
    start : None, str, path-like, SCHAFit or SCHAParams
        Start specification.
    y : pd.Series
        Data, used for the automatic start.
    bands : list of str
        Band labels of the model being fitted.
    step_h : float
        Sampling step in hours.
    m, n : int
        Trend and cycle orders.
    lam : list of float
        Cycle frequencies in radians per sample.

    Raises
    ------
    ValueError
        If a start fit has different bands or step, ``start`` is of an unsupported type, or the
        start parameters do not match ``m``, ``n`` and ``lam``.
    """
    if start is None or (isinstance(start, str) and start == "auto"):
        return auto_start(y.to_numpy(), step_h, lam, m=m, n=n)
    if isinstance(start, (str, Path)):
        start = load_fit(start)
    if isinstance(start, SCHAFit):
        if start.bands != bands or not np.isclose(start.step_h, step_h):
            raise ValueError("Start fit has different bands or step; use start=None for an automatic start.")
        start = start.params
    if not isinstance(start, SCHAParams):
        raise ValueError("start must be None, 'auto', a fit, a fit YAML path or SCHAParams.")
    if (start.m, start.n) != (m, n) or not np.allclose(start.lambdas, lam):
        raise ValueError("Start parameters do not match the requested order and bands.")
    return start


def fit(y, bands: BandSpec = "D1-D6", order: tuple = (2, 3), method: str = "constrained",
        requirements: Union[Requirements, dict, None] = None, fixed: Optional[dict] = None, start=None,
        maxiter: int = 80, workers: Optional[int] = None, verbose: bool = False) -> SCHAFit:
    """Fit a SCHA model to a regular time series.

    Parameters
    ----------
    y : pd.Series
        Observations with a naive DatetimeIndex whose ``freq`` is set; NaN marks gaps. A
        single-column DataFrame is also accepted.
    bands : str or sequence of str, optional
        Tidal bands, e.g. ``"D1-D6"`` (default) or ``["D1", "D2", "D4"]``.
    order : (int, int), optional
        Trend order m and cycle order n; default (2, 3).
    method : {"constrained", "ml", "em"}, optional
        ``"constrained"``: maximum likelihood under the gain requirements (default).
        ``"ml"``: plain maximum likelihood. ``"em"``: EM (no requirements; cannot
        fix sigma2_kappa; rho is held at its start value, or at the fixed value if given).
    requirements : scha.requirements.Requirements or dict, optional
        Gain requirements; default ``Requirements()`` (delta = eps = 0.01, highest
        two bands sacrificial, 18 h cosine-Lanczos post-filter on the subtide). The
        requirements are checked and reported for every method but only imposed by
        ``"constrained"``.
    fixed : dict, optional
        Parameters held fixed: ``rho``, ``sigma2_eps``, ``sigma2_zeta`` and
        ``sigma2_kappa`` (dict band -> value or full list). Fixing a variance switches
        from the concentrated to the ordinary likelihood.
    start : None, "auto", scha.fit.SCHAFit, path or scha.params.SCHAParams, optional
        Starting values; default automatic (:func:`scha.start.auto_start`). A fit or fit YAML
        path must have the same bands and step; parameters must match the order and bands.
    maxiter : int, optional
        Optimiser iteration limit (default 80); for ``"em"`` the maximum number of EM iterations.
    workers : int, optional
        Processes for gradient evaluation (see :func:`scha.mlfit.fit_ml`).
    verbose : bool, optional
        Print per-iteration progress (default False).

    Returns
    -------
    scha.fit.SCHAFit
        Fit with parameters native to the sampling step of ``y`` (``fit_freq``), diagnostics
        and ``y`` attached.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``"constrained"``, ``"ml"``, ``"em"``; if ``method="em"``
        is combined with a fixed ``sigma2_kappa``; or if ``y``, ``bands``, ``fixed`` or
        ``start`` are invalid (including a band at or above Nyquist for the step of ``y``).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}.")
    y = _as_series(y)
    bands = parse_bands(bands)
    m, n = (int(v) for v in order)
    step_h = step_hours(y.index)
    lam = [float(v) for v in lambdas_from_cpd([band_cpd(b) for b in bands], step_h)]
    req = requirements if isinstance(requirements, Requirements) else Requirements.from_dict(requirements)
    creq = CompiledRequirements.build(req, bands, step_h)
    fixed_idx = _resolve_fixed(fixed, bands)
    p0 = _start_params(start, y, bands, step_h, m, n, lam)
    logger.info("Fitting SCHA(%d,%d) %s to %d samples at %s (%.1f%% missing), method=%s", m, n, ",".join(bands),
                y.size, _freqstr(y.index), 100.0 * y.isna().mean(), method)

    t0 = time.time()
    yv = y.to_numpy()
    diag: dict = {}
    if method == "em":
        from .em_fit import scha_em
        if "sigma2_kappa" in fixed_idx:
            raise ValueError("method='em' cannot hold sigma2_kappa fixed.")
        p_init = SCHAParams(m=p0.m, n=p0.n, rho=p0.rho, lambdas=list(p0.lambdas),
                            sigma2_eps=float(fixed_idx.get("sigma2_eps", p0.sigma2_eps)),
                            sigma2_zeta=float(fixed_idx.get("sigma2_zeta", p0.sigma2_zeta)),
                            sigma2_kappa=list(p0.sigma2_kappa))
        rho_em = float(fixed_idx.get("rho", p0.rho))
        yc, _, sd = SCHAFit._centred(y)
        p = scha_em(yc, p_init, fix_rho=rho_em,
                    fix_sigma2_eps="sigma2_eps" in fixed_idx, fix_sigma2_zeta="sigma2_zeta" in fixed_idx,
                    sigma2_diffuse=sd, scale_rho=rho_em, max_iter=maxiter, accelerate=True,
                    verbose=verbose)
        ll = scha_loglik_diffuse(p, yc, sd, scale_rho=p.rho)
        diag.update(loglik=float(ll), success=True, message="EM finished", nit=None)
        if "rho" not in fixed_idx:
            diag["message"] = "EM finished (rho held at its start value; EM does not update rho)"
    else:
        res = fit_ml(yv, p0, creq if method == "constrained" else None, fixed=fixed_idx, maxiter=maxiter,
                     workers=workers, verbose=verbose)
        p = res.params
        diag.update(loglik=res.loglik, loglik_objective=res.loglik_objective, concentrated=res.concentrated,
                    success=res.success, message=res.message, nit=res.nit)
    margins = creq.margins(p)
    diag.update(runtime_s=round(time.time() - t0, 2), min_margin=float(margins.min()),
                requirements_met=bool(margins.min() > -1e-6), crossover_cpd=crossover(p, step_h),
                requirement_report=creq.report(p),
                start=str(y.index[0]), end=str(y.index[-1]), n=int(y.size), n_obs=int(y.notna().sum()),
                gap_fraction=float(y.isna().mean()), name=None if y.name is None else str(y.name))
    logger.info("Done: loglik=%.2f success=%s requirements_met=%s crossover=%.3f cpd (%.0f s)", diag["loglik"],
                diag["success"], diag["requirements_met"], diag["crossover_cpd"], diag["runtime_s"])
    return SCHAFit(params=p, bands=bands, fit_freq=_freqstr(y.index), method=method, requirements=req,
                   fixed=fixed_idx, diagnostics=diag, y=y)


def decompose(y, fit_or_path) -> pd.DataFrame:
    """Decompose a series with a fit.

    Parameters
    ----------
    y : pd.Series
        Data at the fit's step.
    fit_or_path : scha.fit.SCHAFit, str or path-like
        Fit, or a fit YAML path.

    Returns
    -------
    pd.DataFrame
        See :meth:`scha.fit.SCHAFit.decompose`.

    Raises
    ------
    ValueError
        If ``y`` is invalid or its step differs from the fit step.
    """
    f = fit_or_path if isinstance(fit_or_path, SCHAFit) else load_fit(fit_or_path)
    return f.decompose(y)
