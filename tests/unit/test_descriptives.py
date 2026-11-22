"""Post hoc descriptive summaries (d2vi.analysis.descriptives)."""

import numpy as np
import pandas as pd
import pytest

from d2vi.analysis.descriptives import (
    best_single_estimator,
    excess_summary,
    lemma1_terms,
    rule_summary,
)


def _response_set(seed: int = 0, n: int = 4000):
    rng = np.random.default_rng(seed)
    a = (rng.random(n) < 0.25).astype(float)
    y = rng.normal(0.2 + 0.5 * a, 1.0)
    # the flag depends on the outcome within each class, so M1 fails and C_0 != 0
    d = (rng.random(n) < np.where(a == 1, 0.7, 0.1) + 0.15 * (y > 0.5)).astype(float)
    return y, a, d


def test_lemma1_terms_add_up_to_the_deletion_bias_exactly():
    y, a, d = _response_set()
    t = lemma1_terms(y, a, d)
    assert t["eps"] == pytest.approx(a.mean())
    assert t["deletion_bias"] == pytest.approx(y[d == 0].mean() - y[a == 0].mean(), abs=1e-12)
    assert t["naive_bias"] == pytest.approx(y.mean() - y[a == 0].mean(), abs=1e-12)
    assert abs(t["selection"]) > 0.01 and abs(t["residual"]) > 0.01


def test_selection_term_does_not_depend_on_the_contamination_rate():
    y, a, d = _response_set(1)
    low, high = lemma1_terms(y, a, d, 0.02), lemma1_terms(y, a, d, 0.45)
    assert low["selection"] == high["selection"]
    assert abs(high["residual"]) > abs(low["residual"]) > 0
    assert lemma1_terms(y, a, d, 0.0)["residual"] == 0.0


def test_no_residual_contamination_when_every_machine_record_is_flagged():
    y, a, d = _response_set(2)
    d = np.where(a == 1, 1.0, d)
    t = lemma1_terms(y, a, d)
    assert t["s"] == 1.0 and t["residual"] == 0.0
    assert t["deletion_bias"] == pytest.approx(t["selection"])


def test_lemma1_terms_rejects_graded_origins_and_f_equal_one():
    y, a, d = _response_set(3)
    with pytest.raises(ValueError, match="binary"):
        lemma1_terms(y, np.where(a == 1, 0.5, a), d)
    with pytest.raises(ValueError, match="f = 1"):
        lemma1_terms(y, a, np.ones_like(d))


def _cells() -> pd.DataFrame:
    return pd.DataFrame({
        "pool": ["p", "p", "q", "q"], "budget": [0.05] * 4, "eps": [0.01, 0.2, 0.01, 0.2],
        "outcome": ["y"] * 4,
        "mse_naive": [0.0001, 0.0400, 0.0001, 0.2500],
        "mse_deletion": [0.0004, 0.0009, 0.0004, 0.1600],
        "mse_hajek": [0.0300, 0.0300, 0.0300, 0.0300],
        "mse_flag_ppi": [0.0025, 0.0036, 0.0002, 0.0049],
        "mse_rc_ppi_pp": [0.0025, 0.0016, 0.0002, 0.0025],     # ITT: equal to flag-PPI at 1%
        "mse_policy_debiased": [0.0009, 0.0004, 0.0001, 0.0016],
    })


def test_best_single_estimator_labels_the_itt_tie():
    best = best_single_estimator(_cells())
    assert best.tolist() == ["naive", "deletion", "naive", "rc_ppi_pp"]
    tie = _cells().assign(mse_naive=1.0, mse_deletion=1.0, mse_hajek=1.0)
    assert best_single_estimator(tie).tolist()[::2] == ["flag_ppi=rc_ppi_pp"] * 2


def test_excess_is_measured_against_the_best_single_estimator():
    ex = excess_summary(_cells())
    assert ex["deletion"]["max"] == pytest.approx(0.4 - 0.05)
    assert ex["deletion"]["worst_cell"] == {"pool": "q", "budget": 0.05, "eps": 0.2,
                                            "outcome": "y"}
    assert ex["deletion"]["share_above"]["0.2"] == 0.25
    # a rule that switches can beat every single estimator: negative excess in two cells
    assert ex["policy_debiased"]["share_negative"] == 0.5
    assert min(v["max"] for v in ex.values()) == ex["policy_debiased"]["max"]


def test_rule_summary_gives_both_pooling_conventions():
    s = rule_summary(_cells())["deletion"]
    mse = _cells()["mse_deletion"]
    assert s["pooled"] == pytest.approx(np.sqrt(mse.mean()))
    assert s["mean_cell"] == pytest.approx(np.sqrt(mse).mean())
    assert s["pooled"] > s["mean_cell"]          # the pooled figure is dominated by the bad cell
