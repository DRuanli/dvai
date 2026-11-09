import numpy as np
import pandas as pd
import pytest

from d2vi.data.splits import calibration_split, percentile_threshold


def _docs():
    return pd.DataFrame({"doc_id": [f"d{i:03d}" for i in range(200)],
                         "a": np.r_[np.zeros(120), np.ones(80)],
                         "source": ["human"] * 120 + ["gen1"] * 40 + ["gen2"] * 40})


def test_split_is_stratified_deterministic_and_order_free():
    d = _docs()
    cal = calibration_split(d, ["source"])
    assert cal.groupby(d.source).mean().round(2).tolist() == [0.3, 0.3, 0.3]
    shuffled = d.sample(frac=1, random_state=4)
    cal2 = calibration_split(shuffled, ["source"])
    assert set(d.doc_id[cal]) == set(shuffled.doc_id[cal2])


def test_percentile_threshold_ignores_nan():
    assert percentile_threshold(np.r_[np.arange(100.0), np.nan], 95) == pytest.approx(94.05)
    with pytest.raises(ValueError):
        percentile_threshold(np.array([np.nan]))
