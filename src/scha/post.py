import numpy as np

def extract_cycle_amplitude_phase(state: np.ndarray, dim_tr: int, n: int, Nc: int):
    """
    From state vector(s) x, return amplitude and phase arrays for each cycle using top pair.
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
