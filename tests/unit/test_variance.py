import numpy as np
import pytest

from d2vi.inference.variance import ht_variance, variance_report
from d2vi.sampling.audit import AuditSample


def _audit(m0, m1, n0, n1):
    strata = np.r_[np.zeros(m0, int), np.ones(m1, int)]
    rho = np.r_[np.full(m0, m0 / n0), np.full(m1, m1 / n1)]
    return AuditSample(idx=np.arange(m0 + m1), labels=np.zeros(m0 + m1), rho=rho, strata=strata,
                       stratum_sizes={0: n0, 1: n1}, kind="stratified", design="test")


def test_report_matches_textbook_stratified_formula():
    rng = np.random.default_rng(0)
    audit = _audit(20, 25, 800, 100)
    z = rng.normal(size=45)
    rep = variance_report(z, audit)
    textbook = sum(n ** 2 * (1 - m / n) * z[s].var(ddof=1) / m
                   for s, m, n in ((slice(0, 20), 20, 800), (slice(20, 45), 25, 100)))
    assert rep["var"] == pytest.approx(ht_variance(z, audit))
    assert rep["var"] == pytest.approx(textbook)
    assert rep["df"] == 43


def test_shrinkage_rescues_an_all_zero_stratum_and_satterthwaite_flags_dominance():
    audit = _audit(20, 25, 800, 100)
    z = np.r_[np.zeros(20), np.random.default_rng(1).normal(size=25)]
    rep = variance_report(z, audit)
    assert rep["var_shrink"] > rep["var"]          # the zero stratum now contributes
    assert rep["df_satt"] == pytest.approx(24.0)   # only stratum 1 carries variance


def test_shrinkage_is_neutral_when_strata_share_a_variance():
    audit = _audit(30, 30, 300, 300)
    z = np.r_[np.tile([-1.0, 1.0], 15), np.tile([-1.0, 1.0], 15)]
    rep = variance_report(z, audit)
    assert rep["var_shrink"] == pytest.approx(rep["var"])
