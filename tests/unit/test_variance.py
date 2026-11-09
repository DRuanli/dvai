import numpy as np
import pytest

from d2vi.inference.variance import (
    anticipated_s2,
    ht_variance,
    stratum_rate,
    variance_report,
)
from d2vi.sampling.audit import AuditSample


def _audit(m0, m1, n0, n1, labels=None):
    strata = np.r_[np.zeros(m0, int), np.ones(m1, int)]
    rho = np.r_[np.full(m0, m0 / n0), np.full(m1, m1 / n1)]
    labels = np.zeros(m0 + m1) if labels is None else labels
    return AuditSample(idx=np.arange(m0 + m1), labels=labels, rho=rho, strata=strata,
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


def test_pooled_shrinkage_is_neutral_when_strata_share_a_variance_even_with_unequal_sizes():
    # regression test: the first implementation shrank towards a pooled *sum of squares*,
    # which inflated small strata by about (m_big - 1) / (m_small - 1); equal sizes hid it
    audit = _audit(3, 31, 800, 100)
    z = np.r_[[-1.0, 0.0, 1.0], np.tile([-1.0, 1.0], 15), 0.0]
    assert z[:3].var(ddof=1) == pytest.approx(z[3:].var(ddof=1))
    rep = variance_report(z, audit)
    assert rep["var_pool"] == pytest.approx(rep["var"])


def test_pooled_shrinkage_rescues_an_all_zero_stratum_and_satterthwaite_flags_dominance():
    audit = _audit(20, 25, 800, 100)
    z = np.r_[np.zeros(20), np.random.default_rng(1).normal(size=25)]
    rep = variance_report(z, audit)
    assert rep["var_pool"] > rep["var"]            # the zero stratum now contributes
    assert rep["df_satt"] == pytest.approx(24.0)   # only stratum 1 carries variance


def test_anticipated_variance_equals_bernoulli_formula():
    # one stratum, all audited records human, residual jump dz = y - mu
    dz = np.array([1.0, -1.0, 2.0, -2.0])
    z = 0.1 * dz                                   # small human residuals
    a = np.zeros(4)
    p = 0.5 / 5
    expected = np.mean(dz ** 2) * p * (1 - p) + np.var(z - p * dz, ddof=1)
    sel = np.ones(4, bool)
    assert stratum_rate(a, sel, "jeffreys", 1.0, 0.3) == pytest.approx(p)
    assert anticipated_s2(z, dz, a, sel, p) == pytest.approx(expected)


def test_empirical_bayes_rate_shrinks_towards_the_audit_machine_share():
    a = np.r_[np.zeros(4), np.ones(2)]
    sel = np.r_[np.ones(4, bool), np.zeros(2, bool)]
    assert stratum_rate(a, sel, "eb", 1.0, 0.05) == pytest.approx(0.05 / 5)
    assert stratum_rate(a, ~sel, "eb", 2.0, 0.05) == pytest.approx(2.1 / 4)


def test_model_shrinkage_gives_an_all_human_stratum_its_anticipated_variance():
    rng = np.random.default_rng(2)
    labels = np.r_[np.zeros(4), (rng.random(30) < 0.5).astype(float)]
    audit = _audit(4, 30, 800, 100, labels)
    dz = rng.normal(size=34)
    z = (1 - labels) * dz - 0.5 * dz               # RC-PPI-like residual with eta = 0.5
    z[:4] = 0.0                                    # sampled residuals of stratum 0 all zero
    rep = variance_report(z, audit, dz=dz, targets={"model": {"prior": "jeffreys"}})
    assert rep["var_model"] > rep["var"]
    # stratum 1 is well populated: its model target is close to its sample variance, so the
    # increase comes from stratum 0 alone
    unit0 = 800 ** 2 * (1 - 4 / 800) / 4
    s0 = np.r_[np.ones(4, bool), np.zeros(30, bool)]
    t0 = anticipated_s2(z, dz, labels, s0, 0.5 / 5)
    t1 = anticipated_s2(z, dz, labels, ~s0, (labels[4:].sum() + 0.5) / 31)
    s2_1 = z[4:].var(ddof=1)
    unit1 = 100 ** 2 * (1 - 30 / 100) / 30
    shr = lambda s2, t, m: ((m - 1) * s2 + t) / m  # noqa: E731  (kappa = 1)
    assert rep["var_model"] == pytest.approx(unit0 * shr(0.0, t0, 4) + unit1 * shr(s2_1, t1, 30))
