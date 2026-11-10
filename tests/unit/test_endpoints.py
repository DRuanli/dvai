"""Endpoint analysis (PE1-PE4) on constructed replication tables (tests only)."""

import numpy as np
import pandas as pd
import pytest

from d2vi.analysis.endpoints import (
    EndpointSpec,
    _pattern_bootstrap_upper,
    bca_interval,
    cell_stats,
    cert_tables,
    evaluate,
)

SPEC = EndpointSpec()


def _rows(corpus, rep, est, point, feasible=True, lo=None, hi=None, **kw):
    base = {"corpus": corpus, "pool": corpus + "-p", "budget": 0.05, "eps": 0.1,
            "outcome": "y", "rep": rep, "estimator": est, "point": point,
            "feasible": feasible, "ci_low": lo, "ci_high": hi, "truth": 0.0, "sd_h": 2.0,
            "deletion_bias": 0.0, "certified": None, "tau": None}
    return {**base, **kw}


def test_cell_stats_scores_rc_intention_to_treat_and_per_protocol():
    raw = pd.DataFrame([
        # rep 0: RC feasible, error 2 (standardised 1); flag error 4 (standardised 2)
        _rows("c", 0, "rc_ppi_pp", 2.0, True, -1, 5), _rows("c", 0, "flag_ppi", 4.0, True, 3, 5),
        # rep 1: RC infeasible (its point must be ignored); flag error -2, interval covers
        _rows("c", 1, "rc_ppi_pp", 99.0, False, 98, 100),
        _rows("c", 1, "flag_ppi", -2.0, True, -6, 2),
        _rows("c", 0, "naive", 1.0), _rows("c", 1, "naive", 1.0),
    ])
    s = cell_stats(raw, SPEC).iloc[0]
    assert s["mse_rc_ppi_pp"] == pytest.approx((1.0 ** 2 + 1.0 ** 2) / 2)   # ITT: 2/2, -2/2
    assert s["mse_flag_ppi"] == pytest.approx((2.0 ** 2 + 1.0 ** 2) / 2)
    assert s["mse_naive"] == pytest.approx(0.25)
    assert s["feasible"] == 0.5 and s["n_feasible"] == 1
    assert s["mse_pp_rc"] == pytest.approx(1.0) and s["mse_pp_fb"] == pytest.approx(4.0)
    assert s["cover_itt"] == 1.0                    # rep 0: RC interval; rep 1: flag interval
    assert s["width_itt"] == pytest.approx(((5 + 1) / 2 + (2 + 6) / 2) / 2)


def _simulate(rc_sd, flag_sd, n_units=6, n_rep=300, seed=0, cert_when_biased=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        c = f"corpus{u}"
        for rep in range(n_rep):
            feas = rng.random() < 0.7
            e_rc, e_fl = rng.normal(0, rc_sd), rng.normal(0, flag_sd)
            rows.append(_rows(c, rep, "rc_ppi_pp", e_rc, feas, e_rc - 1.96 * rc_sd,
                              e_rc + 1.96 * rc_sd))
            rows.append(_rows(c, rep, "flag_ppi", e_fl, True, e_fl - 1.96 * flag_sd,
                              e_fl + 1.96 * flag_sd))
            for name, bias in (("naive", 0.5), ("deletion", 1.0), ("hajek", 0.0)):
                rows.append(_rows(c, rep, name, bias + rng.normal(0, 0.3)))
            pick = e_rc if feas else e_fl
            rows.append(_rows(c, rep, "policy_debiased", pick))
            biased = rep % 2 == 0
            rows.append(_rows(c, rep, "cert_tost0.2", 0.0,
                              deletion_bias=(0.5 if rep % 4 == 0 else 0.42) if biased
                              else 0.01, tau=0.4,
                              certified=bool(rng.random() < (cert_when_biased if biased else 0.9))))
    return pd.DataFrame(rows)


def _evaluate(raw):
    clusters, cc = cert_tables(raw, SPEC)
    return evaluate(cell_stats(raw, SPEC), clusters, cc, SPEC, n_boot=500)


def test_endpoints_pass_when_rc_is_better_and_intervals_are_valid():
    res = _evaluate(_simulate(rc_sd=0.2, flag_sd=0.4))
    assert res["PE1"]["passes"] and res["PE1"]["itt"]["pooled_ratio"] < 0.6
    assert res["PE1"]["per_protocol"]["pooled_ratio"] == pytest.approx(0.25, abs=0.05)
    assert res["PE2"]["passes"] and res["PE2"]["mean_coverage"] == pytest.approx(0.95, abs=0.02)
    assert res["PE3"]["passes"] and res["PE3"]["bad"]["rate"] == 0.0
    assert res["PE3"]["near"]["replications"] == res["PE3"]["bad"]["replications"] // 2
    assert res["PE3"]["power_small_bias"] == pytest.approx(0.9, abs=0.03)
    assert res["PE4"]["pooled"]["rules"]["deletion"]["max_excess_srmse"] > 0.3


def test_pe1_fails_when_rc_is_worse_and_pe3_fails_when_bias_is_certified():
    res = _evaluate(_simulate(rc_sd=0.5, flag_sd=0.4, cert_when_biased=0.2))
    assert not res["PE1"]["passes"]
    assert res["PE1"]["per_protocol"]["pooled_ratio"] > 1.3
    assert not res["PE3"]["passes"] and res["PE3"]["bad"]["rate"] == pytest.approx(0.2, abs=0.03)


def test_pattern_bootstrap_matches_direct_cluster_resampling():
    rng = np.random.default_rng(3)
    n = rng.integers(0, 5, 4000)
    k = rng.binomial(n, 0.03)
    fast = _pattern_bootstrap_upper(n, k, 3000, seed=1)
    nn, kk = n[n > 0], k[n > 0]
    idx = rng.integers(0, len(nn), size=(3000, len(nn)))
    direct = np.quantile(kk[idx].sum(1) / nn[idx].sum(1), 0.95)
    assert fast == pytest.approx(direct, rel=0.03)


def test_bca_interval_covers_a_ratio_and_needs_three_units():
    rng = np.random.default_rng(0)
    a, b = rng.gamma(5, 1, 12), rng.gamma(5, 2, 12)
    lo, hi = bca_interval(lambda i: a[i].sum() / b[i].sum(), 12, n_boot=1000)
    assert lo < a.sum() / b.sum() < hi
    assert np.isnan(bca_interval(lambda i: 1.0, 2)[0])
