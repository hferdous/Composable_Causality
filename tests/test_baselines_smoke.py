import numpy as np
import pytest

# These wrappers need the optional [discovery] extra.
pytest.importorskip("statsmodels", reason="requires pip install 'dtcd[discovery]'")
pytest.importorskip("lingam", reason="requires pip install 'dtcd[discovery]'")
from dtcd.baselines.grangerf_wrapper import GrangerF
from dtcd.baselines.varlingam_wrapper import VARLiNGAM

import pytest as _pytest
from pathlib import Path as _Path
_DATA_DIR = _Path("data/datasets")
if not _DATA_DIR.exists() or not any(_DATA_DIR.glob("*_X.npy")):
    _pytest.skip(
        "benchmark datasets not generated; run generate_datasets.py first",
        allow_module_level=True,
    )


@pytest.mark.parametrize("method_cls", [GrangerF, VARLiNGAM])
def test_baseline_smoke(method_cls):
    T, d = 200, 5
    max_lag = 1
    X = np.random.randn(T, d)
    
    method = method_cls(max_lag=max_lag, seed=42)
    res = method.fit(X)
    
    assert res.status == "ok"
    assert res.W_inst.shape == (d, d)
    assert res.W_lag.shape == (max_lag, d, d)
    assert isinstance(res.runtime_s, float)
    assert isinstance(res.peak_memory_mb, float)
