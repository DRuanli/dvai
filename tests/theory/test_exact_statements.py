"""Exact checks of the statements in paper/theory.tex (enumeration or algebra, no Monte Carlo)."""

import itertools

import numpy as np
import pytest

from d2vi.estimators import make_estimator
from d2vi.estimators.base import EstimationContext
from d2vi.inference.variance import anticipated_s2
from d2vi.sampling.audit import PerfectVerifier, make_design
from d2vi.sampling.mixture import IIDMixture


def _family(y, h, eta, lam, sample, rho):
    w = 1.0 / rho[sample]
    c = h[sample] - lam * (1 - eta[sample])
    num = lam * np.sum(y * (1 - eta)) + np.sum(w * y[sample] * c)
    den = lam * np.sum(1 - eta) + np.sum(w * c)
    return num, den


def test_theorem1_unbiased_numerator_and_denominator_by_enumeration():
    rng = np.random.default_rng(0)
    n, m = 7, 3
    y, eta = rng.normal(size=n), rng.random(n) * 1.5 - 0.2      # eta may be arbitrarily wrong
    h = np.array([1, 1, 0, 1, 0, 1, 1.0])
    rho = np.full(n, m / n)
    for lam in (0.0, 1.0, 0.37, -2.0):
        vals = [_family(y, h, eta, lam, np.array(s), rho)
                for s in itertools.combinations(range(n), m)]
        num, den = np.mean(vals, axis=0)
        assert num == pytest.approx(np.sum(h * y), abs=1e-12)
        assert den == pytest.approx(h.sum(), abs=1e-12)


@pytest.mark.parametrize("name, lam, use_flag", [("hajek", 0.0, False), ("flag_ppi", 1.0, True)])
def test_theorem1_error_representation_is_exact(diff_pool, name, lam, use_flag):
    rng = np.random.default_rng(5)
    rs = IIDMixture().sample(diff_pool, 400, 0.2, rng)
    audit = make_design("proportional").draw(rs.observed, 40, rng, PerfectVerifier(rs.a))
    res = make_estimator(name).estimate(EstimationContext(rs.observed, audit, rng), "sent")
    y, h = rs.observed.outcome("sent"), 1 - rs.a
    mu = rs.human_mean("sent")
    eta = rs.observed.flag.astype(float) if use_flag else np.ones_like(y)
    e = (h - lam * (1 - eta)) * (y - mu)
    t_hat = np.sum(audit.w * e[audit.idx])
    _, den = _family(y, h, eta, lam, audit.idx, _rho(audit, len(y)))
    assert res.point - mu == pytest.approx((t_hat - e.sum()) / den, abs=1e-12)


def _rho(audit, n):
    rho = np.ones(n)
    rho[audit.idx] = audit.rho
    return rho


def test_corollary1_clipped_tuning_never_exceeds_the_audit_only_variance():
    rng = np.random.default_rng(2)
    for _ in range(200):
        u, v = rng.normal(size=50), rng.normal(size=50) * rng.uniform(0.1, 3)
        v = v + rng.uniform(-2, 2) * u
        lam = np.clip(np.cov(u, v)[0, 1] / np.var(v, ddof=1), 0, 1)
        assert np.var(u - lam * v, ddof=1) <= np.var(u, ddof=1) + 1e-12


def test_proposition4_neyman_allocation_and_the_excess_of_proportional_allocation():
    rng = np.random.default_rng(3)
    n_h, s_h, m = rng.integers(50, 400, 6).astype(float), rng.uniform(0.1, 2, 6), 60.0

    def var(alloc):
        return np.sum(n_h ** 2 * s_h ** 2 / alloc) - np.sum(n_h * s_h ** 2)

    ney = m * n_h * s_h / np.sum(n_h * s_h)
    assert var(ney) == pytest.approx(np.sum(n_h * s_h) ** 2 / m - np.sum(n_h * s_h ** 2))
    for _ in range(300):
        assert var(m * rng.dirichlet(np.ones(6))) >= var(ney) - 1e-9
    prop = m * n_h / n_h.sum()
    s_bar = np.sum(n_h * s_h) / n_h.sum()
    assert var(prop) - var(ney) == pytest.approx(
        n_h.sum() / m * np.sum(n_h * (s_h - s_bar) ** 2))


def test_remark1_anticipated_variance_is_the_exact_expectation():
    rng = np.random.default_rng(4)
    m, p = 5, 0.3
    z0, dz = rng.normal(size=m), rng.normal(size=m)
    expect = 0.0
    for labels in itertools.product([0.0, 1.0], repeat=m):
        a = np.array(labels)
        prob = np.prod(np.where(a == 1, p, 1 - p))
        expect += prob * np.var(z0 - a * dz, ddof=1)
    a = np.array([0, 1, 0, 0, 1.0])
    got = anticipated_s2(z0 - a * dz, dz, a, np.ones(m, bool), p)
    assert got == pytest.approx(expect, abs=1e-12)


def test_lemma2_stagewise_weights_are_unbiased_and_single_stage_weights_are_not():
    g = np.array([1.0, 0.0, 0.0])              # one stratum; g = a
    stage, single = 0.0, 0.0
    for pilot in range(3):
        rest = [i for i in range(3) if i != pilot]
        m2 = 1 if g[pilot] == 1 else 2
        seconds = list(itertools.combinations(rest, m2))
        for s2 in seconds:
            prob = (1 / 3) / len(seconds)
            stage += prob * (g[pilot] + len(rest) / m2 * g[list(s2)].sum())
            single += prob * (3 / (1 + m2)) * (g[pilot] + g[list(s2)].sum())
    assert stage == pytest.approx(1.0, abs=1e-12)
    assert single == pytest.approx(7 / 6, abs=1e-12)


def test_proposition6_bias_formula_is_the_ratio_of_expectations():
    rng = np.random.default_rng(6)
    y, a = rng.normal(size=300), (rng.random(300) < 0.25).astype(float)
    y = y + 1.5 * a
    eps, mu_h, mu_a = a.mean(), y[a == 0].mean(), y[a == 1].mean()
    for phi0, phi1 in ((0.0, 0.1), (0.2, 0.0), (0.05, 0.03)):
        e_h = (1 - phi0) * (1 - a) + phi1 * a              # E[1 - a_tilde]
        target = np.sum(y * e_h) / np.sum(e_h)
        formula = phi1 * eps * (mu_a - mu_h) / ((1 - phi0) * (1 - eps) + phi1 * eps)
        assert target - mu_h == pytest.approx(formula, abs=1e-12)
        if phi1 == 0:
            assert abs(target - mu_h) < 1e-12
