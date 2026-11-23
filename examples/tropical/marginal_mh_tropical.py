"""Marginal adaptive random-walk Metropolis for SCHA variance ratios (states integrated out).

Samples u = (log q_zeta, log q_kappa_1..Nc), q = sigma2 / sigma2_eps, with sigma2_eps integrated out
analytically. Priors match those implied by stochCycle::sc_gibbs:
p(sigma2_eps) ~ 1/sigma2_eps, p(sigma2_kappa) ~ 1/sigma2_kappa, p(sigma2_zeta) ~ 1/sigma2_zeta^2,
which give  log p(u | y) = -0.5 sum log F_t - (N/2 + 1) log S - u_zeta + const,
with S = sum v_t^2 / F_t from a filter run at sigma2_eps = 1.

Usage: python marginal_mh_tropical.py <label> <sigma2_eps0> <sigma2_zeta0> <kappa0> [<kappa0> ...]
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd

from scha.logging_config import configure_logging, resolve_loglevel
from scha.params import SCHAParams
from scha.state_space import build_scha_state_space, default_prior
from scha.ffbs import kalman_filter

DATA = Path(r"f:\projects\stochCycle\example\tropical\test_tide_noise01.csv")
OUT = Path(__file__).resolve().parent / "mh_out"
LAMBDAS = [0.06323351, 0.12646701, 0.18970052, 0.25293402]
RHO = 0.9935
# Trend prior variance in units of sigma2_eps; equals 1e4 absolute at sigma2_eps ~ 1e-4 (as in sc_gibbs)
TREND_P0_REL = 1e8
N_ITER = 6000
N_ADAPT = 1000

logger = logging.getLogger("scha.examples.marginal_mh_tropical")


def log_target(u: np.ndarray, y: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    """Return (log posterior up to a constant, S)."""
    q = np.exp(u)
    p = SCHAParams(m=2, n=4, rho=RHO, lambdas=LAMBDAS, sigma2_eps=1.0,
                   sigma2_zeta=float(q[0]), sigma2_kappa=[float(x) for x in q[1:]])
    ss = build_scha_state_space(p)
    a0, P0 = default_prior(p, sigma2_diffuse=TREND_P0_REL)
    f = kalman_filter(y, T=ss["T"], R=ss["R"], Q=ss["Q"], Z=ss["Z"], H=ss["H"], a0=a0, P0=P0)
    v = np.asarray(f["v"]).reshape(-1)[obs]
    F = np.asarray(f["F"]).reshape(-1)[obs]
    if not np.all(F > 0.0):
        return -np.inf, np.nan
    S = float(np.sum(v * v / F))
    N = int(obs.sum())
    return -0.5 * float(np.sum(np.log(F))) - (0.5 * N + 1.0) * np.log(S) - u[0], S


def marginal_mh(label: str, sigma2_eps0: float, sigma2_zeta0: float, kappa0: list[float]) -> None:
    """Run the marginal random-walk Metropolis chain; writes ``mh_out/mh_trace_<label>.csv``.

    Parameters
    ----------
    label : str
        Chain label (also seeds the generator).
    sigma2_eps0, sigma2_zeta0 : float
        Starting variances; only the ratio sigma2_zeta0 / sigma2_eps0 enters the chain.
    kappa0 : list of float
        Starting sigma2_kappa, one per cycle or a single value shared by all.
    """
    eps0, zeta0 = sigma2_eps0, sigma2_zeta0
    kappa0 = list(kappa0)
    if len(kappa0) == 1:
        kappa0 = kappa0 * len(LAMBDAS)

    y = pd.read_csv(DATA)["value"].to_numpy()
    obs = ~np.isnan(y)
    N = int(obs.sum())
    rng = np.random.default_rng(12345 + sum(label.encode()))

    u = np.log(np.array([zeta0, *kappa0]) / eps0)
    d = u.size
    lp, S = log_target(u, y, obs)
    cov = np.eye(d) * 0.05**2
    chol = np.linalg.cholesky(cov)
    rows, hist = [], []
    n_acc = 0
    t0 = time.time()
    OUT.mkdir(exist_ok=True)

    for it in range(1, N_ITER + 1):
        u_prop = u + chol @ rng.standard_normal(d)
        lp_prop, S_prop = log_target(u_prop, y, obs)
        if np.log(rng.uniform()) < lp_prop - lp:
            u, lp, S = u_prop, lp_prop, S_prop
            n_acc += 1
        # sigma2_eps | q, y ~ InvGamma(N/2 + 1, S/2)
        s2_eps = (S / 2.0) / rng.gamma(0.5 * N + 1.0)
        hist.append(u.copy())
        rows.append([it, lp, s2_eps, *(np.exp(u) * s2_eps)])

        # Haario-style adaptation during burn-in only
        if 200 <= it <= N_ADAPT and it % 50 == 0:
            emp = np.cov(np.array(hist[it // 2:]).T)
            cov = (2.38**2 / d) * emp + 1e-8 * np.eye(d)
            chol = np.linalg.cholesky(cov)

        if it % 100 == 0:
            cols = ["iter", "log_post", "sigma2_eps", "sigma2_zeta"] + [f"kappa_{k + 1}" for k in range(d - 1)]
            pd.DataFrame(rows, columns=cols).to_csv(OUT / f"mh_trace_{label}.csv", index=False)
        if it % 250 == 0:
            r = rows[-1]
            print(f"[{label}] it={it} acc={n_acc / it:.2f} log_post={lp:.2f} eps={r[2]:.3e} "
                  f"zeta={r[3]:.3e} kappa={np.array2string(np.array(r[4:]), precision=2)} "
                  f"({(time.time() - t0) / it:.2f} s/it)", flush=True)

    logger.info("[%s] done: %d iterations, acceptance %.2f, %.0f s", label, N_ITER, n_acc / N_ITER,
                time.time() - t0)


@click.command()
@click.argument("label")
@click.argument("sigma2_eps0", type=float)
@click.argument("sigma2_zeta0", type=float)
@click.argument("kappa0", type=float, nargs=-1, required=True)
@click.option("--logdir", type=click.Path(path_type=Path), default=None, help="Directory for a log file.")
@click.option("--debug", is_flag=True, help="Debug-level logging.")
@click.option("--quiet", is_flag=True, help="No progress messages on the console.")
@click.help_option("-h", "--help")
def main(label, sigma2_eps0, sigma2_zeta0, kappa0, logdir, debug, quiet):
    """Marginal MH for SCHA variance ratios on the noisy tropical test (KAPPA0: one or one per cycle)."""
    level, console = resolve_loglevel(debug=debug, quiet=quiet)
    configure_logging(package_name="scha", level=level, console=console, logdir=logdir,
                      logfile_prefix=f"marginal_mh_{label}")
    marginal_mh(label=label, sigma2_eps0=sigma2_eps0, sigma2_zeta0=sigma2_zeta0, kappa0=list(kappa0))


if __name__ == "__main__":
    main()
