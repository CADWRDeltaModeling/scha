"""Gain requirements imposed as constraints on the SCHA fit.

Requirements are stated on the steady-state smoother gains (:mod:`scha.gains`)
in cycles per day:

- tidal band j (not sacrificial): G_j >= 1 - delta on its named lines
  (:data:`scha.bands.BAND_LINES`);
- subtide: G_sub * H <= eps between bands and above the last band, with H the
  response of an optional cosine-Lanczos post-filter applied to the subtide;
- optional subtide passband: G_sub >= 1 - delta from 0.02 cpd to ``sub_pass``;
- optional crossover: G_sub >= G_D1 at ``sub_cross``.

Sacrificial bands have no passband requirement; they absorb high-frequency
content that would otherwise go to the subtide, the noise or the main bands.
See .github/docs/scha_design.md, sections 1-3.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

from .bands import BAND_LINES, BandSpec, parse_sacrificial
from .gains import cosine_lanczos_response, gains
from .params import SCHAParams
from .timestep import freq_hours

_GRID_STEP = 0.01
_LINEAR_TOP = 8.0


@dataclass
class Requirements:
    """User-facing gain requirements.

    Parameters
    ----------
    delta : float
        Passband tolerance: G_j >= 1 - delta on each required band's lines (default 0.01).
    eps : float
        Subtide tolerance: G_sub * H <= eps on the gap grid (default 0.01).
    sacrificial : int, band spec or None
        Bands without a passband requirement: an int k means the highest k
        fitted bands; a spec such as ``"D5-D6"`` names them (default 2; None means none).
    sub_cross : float, optional
        Require G_sub >= G_D1 at this frequency (cpd). Default None (no requirement).
    sub_pass : float, optional
        Require G_sub >= 1 - delta from 0.02 cpd up to this frequency (cpd). Default None
        (no requirement).
    post_filter : float or str, optional
        Cosine-Lanczos cutoff applied to the subtide (hours, or a duration such
        as ``"18h"``). The subtide product is then the post-filtered subtide. Default
        ``"18h"``; None means no post-filter.
    """

    delta: float = 0.01
    eps: float = 0.01
    sacrificial: Union[int, BandSpec, None] = 2
    sub_cross: Optional[float] = None
    sub_pass: Optional[float] = None
    post_filter: Union[float, str, None] = "18h"

    def post_filter_hours(self) -> Optional[float]:
        """Return the post-filter cutoff in hours.

        Returns
        -------
        float or None
            Cutoff period in hours, or None without a post-filter.

        Raises
        ------
        ValueError
            If ``post_filter`` is a string that is not a fixed positive step (see
            :func:`scha.timestep.freq_hours`).
        """
        pf = self.post_filter
        if pf is None:
            return None
        if isinstance(pf, str):
            return freq_hours(pf)
        return float(pf)

    def to_dict(self) -> dict:
        """Return the settings as a plain dict (YAML-safe).

        Returns
        -------
        dict
            One entry per field of :class:`scha.requirements.Requirements`.
        """
        d = asdict(self)
        if isinstance(d["sacrificial"], tuple):
            d["sacrificial"] = list(d["sacrificial"])
        return d

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Requirements":
        """Return requirements from a dict of settings.

        Parameters
        ----------
        d : dict or None
            Field names and values; None or missing fields take the defaults.

        Returns
        -------
        scha.requirements.Requirements

        Raises
        ------
        ValueError
            For unknown setting names.
        """
        if d is None:
            return cls()
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown requirement settings: {sorted(unknown)}")
        return cls(**d)


def subtide_gap_grid(bands: Sequence[str], step_h: float) -> np.ndarray:
    """Return the frequencies where the subtide gain is bounded by eps.

    The grid covers the gaps between consecutive fitted bands' line sets and
    everything above the last band, up to 0.1 cpd below Nyquist: 0.01 cpd spacing to
    8 cpd, then 40 log-spaced points.

    Parameters
    ----------
    bands : sequence of str
        Fitted band labels, increasing.
    step_h : float
        Sampling step in hours.

    Returns
    -------
    np.ndarray
        Frequencies in cycles per day.
    """
    nyq = 12.0 / step_h
    top = nyq - 0.1
    lines = [BAND_LINES[b] for b in bands]
    parts = [np.arange(lines[j].max() + _GRID_STEP, lines[j + 1].min(), _GRID_STEP)
             for j in range(len(lines) - 1)]
    start = lines[-1].max() + _GRID_STEP
    lin_top = min(_LINEAR_TOP, top)
    parts.append(np.arange(start, lin_top, _GRID_STEP))
    if top > _LINEAR_TOP:
        parts.append(np.geomspace(max(_LINEAR_TOP, start), top, 40))
    return np.concatenate(parts)


@dataclass
class CompiledRequirements:
    """Requirements resolved for a set of bands and a sampling step.

    Build with :meth:`build`. All requirement frequencies are evaluated in one call
    to :func:`scha.gains.gains`; the private slices pick each requirement's rows.

    Attributes
    ----------
    req : scha.requirements.Requirements
        User settings.
    bands : list of str
        Fitted bands.
    step_h : float
        Sampling step in hours.
    required : list of str
        Bands with a passband requirement (fitted minus sacrificial).
    """

    req: Requirements
    bands: list
    step_h: float
    required: list = field(default_factory=list)
    _f: np.ndarray = None
    _pass_slices: list = None
    _sub_slice: slice = None
    _sub_h: np.ndarray = None
    _subpass_slice: Optional[slice] = None
    _cross_index: Optional[int] = None

    @classmethod
    def build(cls, req: Requirements, bands: Sequence[str], step_h: float) -> "CompiledRequirements":
        """Resolve requirements for given bands and step.

        Parameters
        ----------
        req : scha.requirements.Requirements
            User settings.
        bands : sequence of str
            Fitted band labels, increasing.
        step_h : float
            Sampling step in hours.

        Returns
        -------
        scha.requirements.CompiledRequirements

        Raises
        ------
        ValueError
            If ``req.sacrificial`` does not resolve to fitted bands (see
            :func:`scha.bands.parse_sacrificial`), or ``req.post_filter`` is an invalid duration.
        """
        bands = list(bands)
        sacr = set(parse_sacrificial(req.sacrificial, bands))
        required = [b for b in bands if b not in sacr]
        chunks, pass_slices, pos = [], [], 0
        for b in required:
            f = BAND_LINES[b]
            pass_slices.append((bands.index(b), slice(pos, pos + f.size)))
            chunks.append(f)
            pos += f.size
        sub = subtide_gap_grid(bands, step_h)
        sub_slice = slice(pos, pos + sub.size)
        chunks.append(sub)
        pos += sub.size
        pf = req.post_filter_hours()
        sub_h = cosine_lanczos_response(pf, sub, step_h) if pf is not None else np.ones(sub.size)
        subpass_slice = None
        if req.sub_pass is not None:
            sp = np.arange(0.02, req.sub_pass + 1e-9, _GRID_STEP)
            subpass_slice = slice(pos, pos + sp.size)
            chunks.append(sp)
            pos += sp.size
        cross_index = None
        if req.sub_cross is not None:
            cross_index = pos
            chunks.append(np.array([req.sub_cross]))
            pos += 1
        return cls(req=req, bands=bands, step_h=float(step_h), required=required, _f=np.concatenate(chunks),
                   _pass_slices=pass_slices, _sub_slice=sub_slice, _sub_h=sub_h,
                   _subpass_slice=subpass_slice, _cross_index=cross_index)

    def margins(self, p: SCHAParams) -> np.ndarray:
        """Return the log-gain margins of every requirement.

        Parameters
        ----------
        p : scha.params.SCHAParams
            Parameters at this step.

        Returns
        -------
        np.ndarray
            log G - log(1 - delta) on passband lines, log eps - log(G_sub H) on the
            gap grid, and the optional subtide passband and crossover terms; all
            >= 0 when every requirement is met (SLSQP inequality form).
        """
        g = gains(p, self._f, self.step_h)
        d, e = self.req.delta, self.req.eps
        out = [np.log(g[1 + c, s]) - np.log(1.0 - d) for c, s in self._pass_slices]
        out.append(np.log(e) - np.log(g[0, self._sub_slice] * self._sub_h))
        if self._subpass_slice is not None:
            out.append(np.log(g[0, self._subpass_slice]) - np.log(1.0 - d))
        if self._cross_index is not None:
            k = self._cross_index
            out.append(np.array([np.log(g[0, k]) - np.log(g[1, k])]))
        return np.concatenate(out)

    def report(self, p: SCHAParams) -> dict:
        """Return the achieved gains on the requirement grids.

        Parameters
        ----------
        p : scha.params.SCHAParams
            Parameters at this step.

        Returns
        -------
        dict
            ``<band>_min_gain`` per required band, ``subtide_max_gain`` and
            ``subtide_max_gain_cpd`` (including the post-filter), and when set
            ``subtide_pass_min_gain`` and ``sub_cross_gain_ratio``.
        """
        g = gains(p, self._f, self.step_h)
        out = {}
        for c, s in self._pass_slices:
            out[f"{self.bands[c]}_min_gain"] = float(g[1 + c, s].min())
        gs = g[0, self._sub_slice] * self._sub_h
        i = int(np.argmax(gs))
        out["subtide_max_gain"] = float(gs[i])
        out["subtide_max_gain_cpd"] = float(self._f[self._sub_slice][i])
        if self._subpass_slice is not None:
            out["subtide_pass_min_gain"] = float(g[0, self._subpass_slice].min())
        if self._cross_index is not None:
            k = self._cross_index
            out["sub_cross_gain_ratio"] = float(g[0, k] / g[1, k])
        return out

    def report_line(self, p: SCHAParams) -> str:
        """Return a one-line summary of :meth:`report` for logs.

        Parameters
        ----------
        p : scha.params.SCHAParams
            Parameters at this step.

        Returns
        -------
        str
            Minimum gain per required band, maximum subtide gain (times the post-filter
            response if there is one) with its frequency in cpd, and the minimum
            subtide passband gain when ``sub_pass`` is set.
        """
        r = self.report(p)
        parts = [f"{b} min {r[f'{b}_min_gain']:.4f}" for b in self.required]
        tag = "sub*H" if self.req.post_filter is not None else "sub"
        parts.append(f"{tag} max {r['subtide_max_gain']:.2e}@{r['subtide_max_gain_cpd']:.2f}")
        if "subtide_pass_min_gain" in r:
            parts.append(f"sub pass min {r['subtide_pass_min_gain']:.4f}")
        return "  ".join(parts)
