# -*- coding: utf-8
"""
Post-processing utilities for SCHA Gibbs output (alias module)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from scha import SCHAParams, build_scha_state_space
except Exception:
    SCHAParams = None
    build_scha_state_space = None

# (Duplicate utilities; kept as compatibility wrapper)

def _ensure_burn(n: int, nburn: int) -> int:
    if nburn < 0:
        nburn = n // 2
    nburn = max(0, min(nburn, n-1))
    return nburn
