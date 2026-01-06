import numpy as np
from pathlib import Path
import importlib.util
import pandas as pd

# dynamically load the generator module by path so examples/ doesn't need to be a package
_spec_path = Path(__file__).resolve().parents[1] / 'examples' / 'tropical' / 'generate_test_tide.py'
spec = importlib.util.spec_from_file_location('generate_test_tide', str(_spec_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
generate_tide = mod.generate_tide


def test_generated_matches_reference():
    here = Path(__file__).resolve().parent / '..' / 'examples' / 'tropical'
    stoch_ref = Path('f:/projects/stochCycle/example/tropical/test_tide.csv')
    if stoch_ref.exists():
        ref = stoch_ref
    else:
        full_ref = here / 'r_reference' / 'test_tide_full.csv'
        ref = full_ref if full_ref.exists() else (here / 'r_reference' / 'test_tide.csv')
    if not ref.exists():
        # If reference is not present, skip
        import pytest
        pytest.skip('R reference test_tide.csv not present')
    t_ref = pd.read_csv(ref, header=0, index_col=0)
    t_gen, y_gen = generate_tide()
    # Compare shapes
    assert len(y_gen) == len(t_ref)
    # numeric comparison within a small tolerance
    y_ref = t_ref.iloc[:, 0].values.astype(float)
    diff = np.abs(y_ref - y_gen)
    assert np.max(diff) <= 1e-12
