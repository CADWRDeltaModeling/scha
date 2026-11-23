"""Post-processing of SCHA states: amplitude and phase of each cycle."""
import numpy as np

def extract_cycle_amplitude_phase(state: np.ndarray, dim_tr: int, n: int, Nc: int):
    """
    From state vector(s) x, return amplitude and phase arrays for each cycle using top pair.

    Parameters
    ----------
    state : (d,) or (n_time, d) ndarray
        State vector, or one state vector per time, in the ordering of
        :func:`scha.state_space.build_scha_state_space` (trend states
        first, then the cycles). A 1-D input is treated as a single time.
        The top pair of each cycle has scale 1 in the scaled-state
        convention, so scaled and decoded states give the same result.
    dim_tr : int
        Number of trend states, i.e. the offset of the first cycle (the
        trend order m).
    n : int
        Cycle order; each cycle occupies ``2 * n`` consecutive states.
    Nc : int
        Number of cycles.

    Returns
    -------
    amp : ndarray
        Amplitude ``hypot(psi, psi*)`` of each cycle's top pair, shape
        (n_time, Nc) with length-1 dimensions removed.
    phase : ndarray
        Phase ``arctan2(-psi*, psi)`` in radians, in (-pi, pi], of the top
        pair, the angle of ``psi - i psi*``; same shape as `amp`. It is the
        phase of the state itself and is not demodulated by the cycle
        frequency.

    See Also
    --------
    scha.phase : Phase conventions and demodulation to a fixed epoch.
    """
    x = np.atleast_2d(state)
    Tt, d = x.shape
    amp = np.zeros((Tt, Nc)); phase = np.zeros((Tt, Nc))
    for t in range(Tt):
        for c in range(Nc):
            start = dim_tr + c*(2*n)
            re_ = x[t, start + 0]; im_ = x[t, start + 1]
            amp[t, c] = np.hypot(re_, im_)
            phase[t, c] = np.arctan2(-im_, re_)
    return amp.squeeze(), phase.squeeze()
