import numpy as np
import pytest

from scha.crossval import blocks, cv_blocks, grid_cv

from test_fit import _synthetic


def _r_calculate_blocks(ylen, foldsize, blocksize, first):
    """Literal transliteration of stochCycle calculate_blocks (1-based indices)."""
    out, sblock = [], first
    while sblock < first + blocksize:
        starts = np.arange(sblock, ylen + 1, blocksize)
        mask = [np.arange(s, s + foldsize) for s in starts if s + foldsize - 1 <= ylen]
        out.append(np.concatenate(mask))
        sblock += foldsize
    return out


def test_cv_blocks_matches_r_freeport_design():
    r = _r_calculate_blocks(18001, 100, 1001, 400)
    py = cv_blocks(18001, 100, 1001, first=399)
    assert len(py) == len(r) == 11
    for a, b in zip(py, r):
        assert np.array_equal(a, b - 1)
    # the R design overlaps: 11 folds of 100 per 1001-sample period
    allpos = np.concatenate(py)
    assert allpos.size > np.unique(allpos).size
    # spacing a multiple of gap tiles the record exactly
    tiled = np.concatenate(cv_blocks(18001, 100, 1000, first=399))
    assert tiled.size == np.unique(tiled).size


def test_blocks_from_durations():
    y = _synthetic(days=30)
    f = blocks(y.index, gap="25h", spacing="250h", first="100h")
    assert len(f) == 10 and all(np.diff(m).min() >= 1 for m in f)
    assert f[0][0] == 400 and f[0][100] - f[0][0] == 1000
    with pytest.raises(ValueError):
        blocks(y.index, gap="25h7min")


@pytest.mark.parametrize("refit", [False, True])
def test_grid_cv_scores(refit):
    y = _synthetic(freq="h", days=40)
    folds = blocks(y.index, gap="25h", spacing="250h", first="50h")[:3]
    df = grid_cv(y, rho=[0.94], sigma2_zeta=[1e-6, 1e-3], bands="D1-D4", folds=folds, refit_folds=refit,
                 maxiter=30, workers=1)
    assert list(df["sigma2_zeta"]) == [1e-6, 1e-3]
    assert (df["n_withheld"] > 0).all()
    assert np.isfinite(df["cv_mse"]).all()
    # a far too rough trend must predict withheld gaps worse
    assert df["cv_mse"].iloc[1] > df["cv_mse"].iloc[0]
