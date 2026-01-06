"""
Stochastic Cycle Harmonic Analysis (SCHA) — Gibbs sampler scaffold

Goal: a near-reference translation of the R function `sc_gibbs()` with the same sampling order
and parameterizations. This file focuses on the Gibbs loop; model-construction utilities are
stubbed as pure functions so we can drop in exact translations of the R helpers later.

Key choices that must match the R reference:
- Inverse-Gamma parameterization is (shape, rate), i.e. if X ~ IG(alpha, beta) then p(x) ∝ beta^alpha / Γ(alpha) * x^{-(alpha+1)} * exp(-beta/x).
  Sampling implemented via: X = 1 / Gamma(shape=alpha, scale=1/beta).
- Step 1 updates sigma2_epsilon using residuals from the *previous* state draw (state_part),
  except for itr==1 which uses an initial fixed value.
- Only the last trend component has innovation variance (others are zero) both in W and in the
  initial state covariance (diffuse prior for trend, harmonic blocks for cycles).
- Kappa (cycle variances) are drawn per-frequency with shape = nobs + order_cycle - 1 and a
  scale term computed by `cal_shape_kappa(...)` that we will port exactly later.
- Rho is updated via a random-walk Metropolis-Hastings step using the same log-likelihood pieces
  (quadratic forms + log-determinants) accumulated across frequencies.

This scaffold is intentionally modular; individual pieces can be unit-tested in isolation and
replaced by exact translations of the R helpers as they are ported.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

Array = np.ndarray

# -----------------------------
# Utilities / helpers (exact-match semantics)
# -----------------------------

def inv_gamma_sample(shape: float, rate: float, rng: np.random.Generator) -> float:
    """Sample from Inverse-Gamma(shape=shape, rate=rate) with R's invgamma::rinvgamma semantics.
    Equivalent to 1 / Gamma(shape, scale=1/rate).
    """
    if shape <= 0 or rate <= 0:
        raise ValueError("shape and rate must be positive for inverse-gamma")
    g = rng.gamma(shape=shape, scale=1.0 / rate)
    return 1.0 / g


def blkdiag(blocks: List[Array]) -> Array:
    """Simple block-diagonal (numpy-only) utility for small matrices.
    Accepts 2D arrays. Returns a 2D array with blocks on the diagonal.
    """
    if not blocks:
        return np.zeros((0, 0))
    rows = sum(b.shape[0] for b in blocks)
    cols = sum(b.shape[1] for b in blocks)
    out = np.zeros((rows, cols), dtype=blocks[0].dtype)
    r, c = 0, 0
    for b in blocks:
        rr, cc = b.shape
        out[r:r+rr, c:c+cc] = b
        r += rr
        c += cc
    return out


# -----------------------------
# Model container (dlm-like)
# -----------------------------
@dataclass
class SCModel:
    FF: Array  # (1 x nstate) observation matrix
    GG: Array  # (nstate x nstate) state evolution matrix
    W: Array   # (nstate x nstate) innovation covariance (will be updated)
    m0: Array  # (nstate,) initial mean
    C0: Array  # (nstate x nstate) initial covariance (will be updated)

    # convenience setters matching the R code intent
    def set_V(self, sigma2_eps: float) -> None:
        self.V = np.array([[sigma2_eps]], dtype=float)

    def set_W(self, Wtrend_vec: Array, Wcycle_vec: Array) -> None:
        self.W = np.diag(np.concatenate([Wtrend_vec, Wcycle_vec]))

    def set_C0(self, C0_trend: Array, C0_cycle_blocks: List[Array], sigma2_kappa_per_freq: Array) -> None:
        # Scale each cycle covariance block by the corresponding sigma2_kappa
        scaled_blocks = [C0_cycle_blocks[j] * float(sigma2_kappa_per_freq[j]) for j in range(len(C0_cycle_blocks))]
        C0_cycle = blkdiag(scaled_blocks) if scaled_blocks else np.zeros((0, 0))
        self.C0 = blkdiag([C0_trend, C0_cycle]) if C0_cycle.size else C0_trend


# -----------------------------
# Placeholders for exact R helpers to be ported verbatim
# -----------------------------

def sc_model_mat(order_trend: int, order_cycle: int, freqs: Array, rho: float) -> Tuple[Array, Array]:
    """Return (FF, GG) exactly matching the R `sc_model_mat()`.
    TODO: port directly from R.
    """
    raise NotImplementedError("sc_model_mat() must be ported from R")


def calc_cycle_covariance_matrix(order_cycle: int, rho: float, freq: float) -> Array:
    """Exact port of R `calc_cycle_covariance_matrix()`.
    Produces the stationary covariance for the cycle block for given (order_cycle, rho, freq).
    """
    raise NotImplementedError("calc_cycle_covariance_matrix() must be ported from R")


def cal_shape_kappa(order_trend: int, order_cycle: int, jj: int, freq: float,
                    rho: float, covar_inv: Array, nobs: int, alpha_draw: Array) -> float:
    """Exact port of R `cal_shape_kappa()`.
    Returns the quadratic form entering the kappa inverse-gamma rate (R uses `rate=scale_kappa/2`).
    """
    raise NotImplementedError("cal_shape_kappa() must be ported from R")


def ck_bsample(res: Array, model: SCModel, rng: np.random.Generator) -> Array:
    """Conditional simulator for the full state path α_{1:T} | y, θ using Carter-Kohn/Durbin-Koopman.
    Must match the distribution used by R's dlm::dlmBSample(). Returns array with shape (T, nstate).
    TODO: connect to an existing CK/DK implementation.
    """
    raise NotImplementedError("ck_bsample() must be provided (CK/DK backward sampler)")


# -----------------------------
# Gibbs sampler (near-reference order)
# -----------------------------

def sc_gibbs(y: Array,
             order_trend: int,
             order_cycle: int,
             freqs: Array,
             mc_iter: int,
             *,
             initial_sigma2_epsilon: float = 1e-4,
             sigma2_zeta_fixed: bool = False,
             initial_sigma2_zeta: float = 1e-6,
             initial_sigma2_kappa: float | Array = 1e-11,
             sigma2_diffuse: float = 1e4,
             rho_fixed: bool = False,
             initial_rho: float = 0.9925,
             rho_high: float = 0.9945,
             rho_low: float = 0.99,
             mh_sd_rho: float = 2.5e-5,
             thin_rate: int = 5,
             log_state_only: bool = False,
             rng: Optional[np.random.Generator] = None) -> Dict[str, object]:
    """Gibbs sampler for SCHA following the exact order and parameterization of the R reference.

    Args mirror the R defaults where possible. `y` may contain NaNs for missing values.
    Returns dict with posterior traces and (optionally) thinned state-derived summaries.
    """
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, dtype=float).reshape(-1)
    nobs = y.shape[0]
    miss = np.isnan(y)
    ygood = y[~miss]
    ngood = ygood.size

    n_freq = len(freqs)
    rho = float(initial_rho)

    # Build model matrices
    FF, GG = sc_model_mat(order_trend, order_cycle, freqs, rho)

    # V will be set below via set_V
    # W initial: trend last element gets initial_sigma2_zeta; cycles last two per frequency
    Wtrend = np.concatenate([np.zeros(order_trend - 1), np.array([1.0])]) * initial_sigma2_zeta
    if np.isscalar(initial_sigma2_kappa):
        initial_sigma2_kappa = np.repeat(float(initial_sigma2_kappa), n_freq)
    Wcycle = np.hstack([
        np.concatenate([np.zeros(2 * (order_cycle - 1)), np.array([1.0, 1.0])]) * initial_sigma2_kappa[j]
        for j in range(n_freq)
    ])

    nstate = order_trend + 2 * order_cycle * n_freq

    # Cache cycle covariance inverses for current rho, and placeholders for proposed rho
    covar_matrix_cycle = []
    covar_matrix_cycle_inv = []
    for j in range(n_freq):
        Cj = calc_cycle_covariance_matrix(order_cycle, rho, float(freqs[j]))
        covar_matrix_cycle.append(Cj)
        covar_matrix_cycle_inv.append(np.linalg.inv(Cj))

    covar_new = [None] * n_freq
    covar_inv_new = [None] * n_freq

    # Initial state mean/covariance
    m0 = np.zeros(nstate)
    C0_trend = np.eye(order_trend) * sigma2_diffuse
    C0_cycle_blocks = [covar_matrix_cycle[j] for j in range(n_freq)]

    model = SCModel(FF=FF, GG=GG, W=np.diag(np.concatenate([Wtrend, Wcycle])), m0=m0, C0=None)
    model.set_V(initial_sigma2_epsilon)
    model.set_C0(C0_trend=C0_trend, C0_cycle_blocks=C0_cycle_blocks, sigma2_kappa_per_freq=np.asarray(initial_sigma2_kappa))

    # Storage
    if log_state_only:
        log_variances = False
    else:
        log_variances = True
        sigma2_epsilon_save = np.empty(mc_iter)
        sigma2_kappa_save = np.empty((mc_iter, n_freq))
        sigma2_zeta_save = np.empty(mc_iter)
        subtide_thinned = []  # alpha[:,0] in R
        amplitudes_thinned: Dict[str, List[Array]] = {}
        phase_thinned: Dict[str, List[Array]] = {}
        freq_thinned: Dict[str, List[Array]] = {}

    state_thinned = []
    rho_save = np.empty((mc_iter, 1))
    accept_reject = np.array([0, 0], dtype=int)  # accepted, rejected

    # For itr==1 we won't have alpha_draw yet; initialize placeholder
    alpha_draw_prev: Optional[Array] = None

    for itr in range(1, mc_iter + 1):
        # --- Step 1: observation equation variance (sigma2_epsilon)
        if itr == 1 or alpha_draw_prev is None:
            y_mod = y - np.nanmean(y)
            sigma2_eps_draw = float(initial_sigma2_epsilon)
        else:
            state_part = alpha_draw_prev @ model.FF.T  # (T x nstate) @ (nstate x 1) -> (T x 1)
            y_mod = y - state_part.reshape(-1)
            tide_res = y_mod[~miss]
            s2 = float(tide_res @ tide_res)
            sigma2_eps_draw = inv_gamma_sample(shape=ngood / 2.0, rate=s2 / 2.0, rng=rng)
        model.set_V(sigma2_eps_draw)
        if log_variances:
            sigma2_epsilon_save[itr - 1] = sigma2_eps_draw

        # Pack residuals with NAs masked (same size as y)
        res = y.copy()
        res[miss] = np.nan  # kept for alignment

        # --- Step 2: sample state path alpha using CK/DK conditional simulator
        alpha_draw = ck_bsample(res, model, rng)  # (T x nstate)
        # R removes the first row from dlmBSample (initial-time dummy); our CK should not include it.
        alpha_draw_prev = alpha_draw

        # Optionally thin reconstructed signal y_hat = FF alpha
        do_thinning = (itr % thin_rate) == 0
        if do_thinning:
            reconstruct = alpha_draw @ model.FF.T  # (T x 1)
            state_thinned.append(reconstruct.reshape(1, -1))

        # --- Step 3.1: sigma2_zeta (trend)
        # Only the last trend component evolves; difference that component over time
        trend_col = order_trend - 1  # 0-based index matching R's [order_trend]
        trend_diff = alpha_draw[1:, trend_col] - alpha_draw[:-1, trend_col]
        scale_zeta = float(trend_diff @ trend_diff)
        shape_zeta = nobs / 2.0 + 0.5
        if sigma2_zeta_fixed:
            sigma2_zeta = float(initial_sigma2_zeta)
        else:
            sigma2_zeta = inv_gamma_sample(shape=shape_zeta, rate=scale_zeta / 2.0, rng=rng)
        if log_variances:
            sigma2_zeta_save[itr - 1] = sigma2_zeta
        Wtrend_vec = np.concatenate([np.zeros(order_trend - 1), np.array([sigma2_zeta])])

        # --- Step 3.2: cycles: sigma2_kappa per frequency and MH step for rho
        rho_old = rho
        rho_new = rho_old + rng.normal(loc=0.0, scale=mh_sd_rho)
        rho_in_bounds = (rho_low <= rho_new <= rho_high)
        if rho_fixed:
            rho_in_bounds = False
            rho_new = rho_old

        pold = 0.0
        pnew = 0.0
        sigma2_kappa_itr = np.zeros(n_freq)
        Wcycle_vec_parts: List[Array] = []

        for jj in range(n_freq):
            freq_j = float(freqs[jj])
            covar_inv_sel = covar_matrix_cycle_inv[jj]
            # scale_kappa is the quadratic form entering IG rate (R uses rate=scale_kappa/2)
            scale_kappa = cal_shape_kappa(order_trend, order_cycle, jj + 1, freq_j,
                                          rho_old, covar_inv_sel, nobs, alpha_draw)
            shape_kappa = nobs + order_cycle - 1
            sigma2_kappa = inv_gamma_sample(shape=shape_kappa, rate=scale_kappa / 2.0, rng=rng)
            sigma2_kappa_itr[jj] = sigma2_kappa
            # Append zeros for non-innovating cycle states, followed by two variances
            Wcycle_vec_parts.append(np.zeros(2 * (order_cycle - 1)))
            Wcycle_vec_parts.append(np.array([sigma2_kappa, sigma2_kappa]))

            if (not rho_fixed) and rho_in_bounds:
                covar_old = covar_matrix_cycle[jj]
                arg_old = scale_kappa
                # pdivold = 0.5 * log det (sigma2_kappa * covar_old)
                sign, logdet_old = np.linalg.slogdet(sigma2_kappa * covar_old)
                pdivold = 0.5 * logdet_old
                pold += -0.5 * arg_old - pdivold

                covar_new[jj] = calc_cycle_covariance_matrix(order_cycle, rho_new, freq_j)
                covar_inv_new[jj] = np.linalg.inv(covar_new[jj])
                arg_new = cal_shape_kappa(order_trend, order_cycle, jj + 1, freq_j,
                                          rho_new, covar_inv_new[jj], nobs, alpha_draw)
                sign, logdet_new = np.linalg.slogdet(sigma2_kappa * covar_new[jj])
                pdivnew = 0.5 * logdet_new
                pnew += -0.5 * arg_new - pdivnew

        Wcycle_vec = np.concatenate(Wcycle_vec_parts) if Wcycle_vec_parts else np.zeros(0)

        # MH accept/reject for rho
        rho = rho_old  # clarity
        if (not rho_fixed) and rho_in_bounds:
            if pnew >= pold:
                rho = rho_new
            else:
                logpdiff = pnew - pold
                accept_rate = np.exp(logpdiff)
                if rng.uniform() < accept_rate:
                    rho = rho_new
            if rho == rho_new:
                accept_reject[0] += 1
                for jj in range(n_freq):
                    covar_matrix_cycle[jj] = covar_new[jj]
                    covar_matrix_cycle_inv[jj] = covar_inv_new[jj]
            else:
                accept_reject[1] += 1
        rho_save[itr - 1, 0] = rho

        # Update model W and C0 with the new draws
        model.set_W(Wtrend_vec=Wtrend_vec, Wcycle_vec=Wcycle_vec)
        model.set_C0(C0_trend=C0_trend, C0_cycle_blocks=covar_matrix_cycle,
                     sigma2_kappa_per_freq=sigma2_kappa_itr)

        # Optional logging / thinning of components, matching R names
        if do_thinning and (not log_state_only):
            # subtide_thinned: trend intercept? In R this is alpha_draw[,1]; keep exact index here
            subtide_thinned.append(alpha_draw[:, 0][None, :])
            # For amplitudes/phase/freq per harmonic, we preserve structure and names to be filled later
            # TODO: port frequency labels (names(sc_freq) in R) and exact indices jndx

            sigma2_kappa_save[itr - 1, :] = sigma2_kappa_itr
            # sigma2_epsilon_save already set above
            # sigma2_zeta_save set above

    out: Dict[str, object] = {
        "rho": rho_save,
        "state_thinned": np.vstack(state_thinned) if state_thinned else np.empty((0, nobs)),
    }
    if not log_state_only:
        out.update({
            "sigma2_epsilon": sigma2_epsilon_save,
            "sigma2_zeta": sigma2_zeta_save,
            "sigma2_kappa": sigma2_kappa_save,
            # placeholders for thinned component logs
            "subtide_thinned": np.vstack(subtide_thinned) if subtide_thinned else np.empty((0, nobs)),
            "amplitude_thinned": amplitudes_thinned,
            "phase_thinned": phase_thinned,
            "freq_thinned": freq_thinned,
            "order_trend": order_trend,
            "order_cycle": order_cycle,
            "freqs": np.asarray(freqs),
        })
    return out
