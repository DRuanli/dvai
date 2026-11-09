import numpy as np
import pytest

from d2vi.sampling.audit import PerfectVerifier, _capped_inclusion, make_design
from d2vi.sampling.mixture import IIDMixture

DESIGNS = ["srs", {"name": "proportional", "n_strata": 10}, "top_decile", "adaptive",
           {"name": "neyman", "n_strata": 5}, "pps"]


@pytest.mark.parametrize("spec", DESIGNS)
@pytest.mark.parametrize("m", [9, 45, 90])
def test_exact_budget_and_valid_probabilities(pool, spec, m):
    rng = np.random.default_rng(3)
    rs = IIDMixture().sample(pool, 900, 0.1, rng)
    audit = make_design(spec).draw(rs.observed, m, rng, PerfectVerifier(rs.a))
    assert audit.m == m
    assert np.unique(audit.idx).shape[0] == m
    assert np.all((audit.rho > 0) & (audit.rho <= 1))
    np.testing.assert_array_equal(audit.labels, rs.a[audit.idx])
    if audit.kind == "stratified":
        # stratified SRSWOR weights add up to the population size exactly
        assert audit.w.sum() == pytest.approx(rs.n)


def test_capped_inclusion_sums_to_budget():
    x = np.r_[np.ones(95), np.full(5, 100.0)]
    rho = _capped_inclusion(x, 20)
    assert rho.sum() == pytest.approx(20)
    assert rho.max() <= 1.0 + 1e-12


@pytest.mark.parametrize("spec", DESIGNS)
def test_horvitz_thompson_total_is_unbiased(pool, spec):
    """E[sum_s w x] = sum_U x for every design (checked by Monte Carlo)."""
    rng = np.random.default_rng(11)
    rs = IIDMixture().sample(pool, 300, 0.1, rng)
    x = rs.observed.outcome("sent") + 2.0
    true_total = x.sum()
    design = make_design(spec)
    totals = []
    for _ in range(1500):
        a = design.draw(rs.observed, 30, rng, PerfectVerifier(rs.a))
        totals.append(np.sum(a.w * x[a.idx]))
    totals = np.asarray(totals)
    mc_se = totals.std(ddof=1) / np.sqrt(totals.shape[0])
    assert abs(totals.mean() - true_total) < 4 * mc_se


def test_top_decile_oversamples_machines(pool):
    rng = np.random.default_rng(5)
    rs = IIDMixture().sample(pool, 900, 0.05, rng)
    srs, top = make_design("srs"), make_design("top_decile")
    n_srs = np.mean([srs.draw(rs.observed, 45, rng, PerfectVerifier(rs.a)).n_machine
                     for _ in range(300)])
    n_top = np.mean([top.draw(rs.observed, 45, rng, PerfectVerifier(rs.a)).n_machine
                     for _ in range(300)])
    assert n_srs == pytest.approx(45 * 0.05, rel=0.2)
    assert n_top > 2 * n_srs
