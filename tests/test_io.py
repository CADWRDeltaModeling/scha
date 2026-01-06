import pytest
import math
from pathlib import Path
from scha.io import parse_dt, freqs_from_config, normalize_sigma2_kappa, load_scha_params_from_yaml
import numpy as np


def test_parse_dt_strings():
    assert abs(parse_dt('1h') - 1.0/24.0) < 1e-12
    assert abs(parse_dt('15min') - 15.0/(24.0*60.0)) < 1e-14
    assert abs(parse_dt('1d') - 1.0) < 1e-12
    assert abs(parse_dt(1.0/24.0) - 1.0/24.0) < 1e-12


def test_freqs_from_list_and_dict():
    # list with alias names
    cpd, labs = freqs_from_config(['D1','D2'])
    assert len(cpd) == 2
    assert labs == ['D1','D2']
    # dict mapping keeps keys order
    cpd2, labs2 = freqs_from_config({'A': 'D1', 'B': 1.0})
    assert labs2 == ['A','B']
    assert cpd2[1] == pytest.approx(1.0)


def test_normalize_sigma2_kappa_list_and_scalar_and_dict():
    labels = ['D1','D2','D3']
    out = normalize_sigma2_kappa(1e-13, 3, labels)
    assert out == [1e-13,1e-13,1e-13]
    out = normalize_sigma2_kappa([1e-12], 3, labels)
    assert out == [1e-12,1e-12,1e-12]
    out = normalize_sigma2_kappa([1e-12,2e-12,3e-12], 3, labels)
    assert out == [1e-12,2e-12,3e-12]
    with pytest.raises(ValueError):
        normalize_sigma2_kappa([1e-12,2e-12], 3, labels)
    with pytest.raises(ValueError):
        normalize_sigma2_kappa({'D1':1,'D3':2}, 3, labels)


def test_load_scha_params_from_yaml_and_missing_sigma(tmp_path):
    here = Path(__file__).resolve().parent / '..' / 'examples' / 'tropical'
    cfg = here / 'tropical_params.yml'
    # load using the example csv index to infer dt
    csv = here / 'tropical_sample.csv'
    import pandas as pd
    df = pd.read_csv(csv, parse_dates=['time']).set_index('time')
    params, meta = load_scha_params_from_yaml(cfg, infer_dt_from_index=True, index=df.index)
    assert params.m == 2
    assert len(params.lambdas) == 4

    # now create a yaml with no sigma2_kappa and expect ValueError
    s = cfg.read_text()
    s2 = "\n".join([line for line in s.splitlines() if not line.strip().startswith('sigma2_kappa')])
    f = tmp_path / 'no_s2k.yml'
    f.write_text(s2)
    with pytest.raises(ValueError):
        load_scha_params_from_yaml(str(f), infer_dt_from_index=True, index=df.index)
