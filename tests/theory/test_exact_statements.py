"""Checks of the statements in paper/theory.tex: enumeration or algebra where possible,
simulation (marked in the test name or docstring) for the asymptotic statements."""

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


# --- added after the independent review of the theory section (finding F18) ---------------

def test_theorem1_holds_for_stratified_audits_and_graded_origins():
    rng = np.random.default_rng(11)
    strata = [np.arange(0, 4), np.arange(4, 7)]
    sizes = (2, 1)
    y, eta = rng.normal(size=7), rng.random(7)
    h = np.array([1.0, 0.6, 0.0, 1.0, 0.25, 1.0, 0.0])            # hybrid authorship
    rho = np.concatenate([np.full(4, 2 / 4), np.full(3, 1 / 3)])
    samples = [np.array(s1 + s2) for s1 in itertools.combinations(strata[0], sizes[0])
               for s2 in itertools.combinations(strata[1], sizes[1])]
    mu = np.sum(h * y) / h.sum()
    for lam in (0.0, 1.0, 0.4, 1.7):
        vals = np.array([_family(y, h, eta, lam, s, rho) for s in samples])
        assert vals[:, 0].mean() == pytest.approx(np.sum(h * y), abs=1e-12)
        assert vals[:, 1].mean() == pytest.approx(h.sum(), abs=1e-12)
        e = (h - lam * (1 - eta)) * (y - mu)
        for s, (num, den) in zip(samples, vals, strict=True):
            assert num / den - mu == pytest.approx((np.sum(e[s] / rho[s]) - e.sum()) / den,
                                                   abs=1e-12)


def test_theorem1_representation_holds_at_the_tuned_coefficient(diff_pool):
    rng = np.random.default_rng(12)
    rs = IIDMixture().sample(diff_pool, 400, 0.2, rng)
    audit = make_design("proportional").draw(rs.observed, 40, rng, PerfectVerifier(rs.a))
    res = make_estimator("flag_ppi_pp").estimate(EstimationContext(rs.observed, audit, rng),
                                                 "sent")
    y, h, mu = rs.observed.outcome("sent"), 1 - rs.a, rs.human_mean("sent")
    lam, eta = res.extras["lam"], rs.observed.flag.astype(float)
    e = (h - lam * (1 - eta)) * (y - mu)
    _, den = _family(y, h, eta, lam, audit.idx, _rho(audit, len(y)))
    assert res.point - mu == pytest.approx((np.sum(audit.w * e[audit.idx]) - e.sum()) / den,
                                           abs=1e-12)


@pytest.mark.parametrize("name", ["flag_ppi", "rc_ppi_pp"])
def test_theorem2_linearised_normal_interval_covers_in_a_large_audit(diff_pool, name):
    """Equation (se) with the normal quantile, not the finite-sample rule of Remark 1."""
    cover, reps = [], 400
    for r in range(reps):
        rng = np.random.default_rng(7000 + r)
        rs = IIDMixture().sample(diff_pool, 3000, 0.2, rng)
        audit = make_design("proportional").draw(rs.observed, 600, rng, PerfectVerifier(rs.a))
        res = make_estimator(name).estimate(EstimationContext(rs.observed, audit, rng), "sent")
        assert res.feasible
        cover.append(abs(res.point - rs.human_mean("sent")) <= 1.959964 * res.extras["se_lin"])
    assert 0.92 <= np.mean(cover) <= 0.98, np.mean(cover)


def test_corollary1_package_tuning_never_raises_the_estimated_variance(diff_pool):
    from d2vi.estimators.design_based import tune_lambda
    from d2vi.inference.variance import ht_variance

    for r in range(60):
        rng = np.random.default_rng(8000 + r)
        rs = IIDMixture().sample(diff_pool, 900, 0.2, rng)
        audit = make_design("proportional").draw(rs.observed, 45, rng, PerfectVerifier(rs.a))
        y = rs.observed.outcome("sent")[audit.idx]
        mu = float(np.average(y, weights=1 - audit.labels + 1e-9))
        for eta in (rs.observed.flag.astype(float)[audit.idx], rng.random(audit.m)):
            lam = tune_lambda(y, audit.labels, eta, mu, audit)
            u, v = (1 - audit.labels) * (y - mu), (1 - eta) * (y - mu)
            assert 0.0 <= lam <= 1.0
            assert ht_variance(u - lam * v, audit) <= ht_variance(u, audit) * (1 + 1e-9) + 1e-12


def test_proposition3_starvation_probabilities_quoted_in_the_text():
    from scipy import stats

    k = stats.hypergeom(900, 45, 45)                 # N = 900, eps = 0.05, m = 45
    assert k.mean() == pytest.approx(45 * 0.05)
    assert round(float(k.sf(3)), 2) == 0.18          # P(K >= 4)
    assert round(float(k.sf(5)), 2) == 0.02          # P(K >= 6)


def _two_stage_moments(z, a, n1, rule):
    """Exact mean and variance of the stage-wise estimator, and the mean of the second-stage
    variance estimator, for one stratum; ``rule(pilot labels) -> m2``."""
    n, mean, second, v_hat = len(z), 0.0, 0.0, 0.0
    pilots = list(itertools.combinations(range(n), n1))
    for pilot in pilots:
        rest = [i for i in range(n) if i not in pilot]
        m2 = rule(a[list(pilot)])
        seconds = list(itertools.combinations(rest, m2))
        for s2 in seconds:
            prob = 1 / len(pilots) / len(seconds)
            zs = z[list(s2)]
            est = z[list(pilot)].sum() + len(rest) / m2 * zs.sum()
            mean += prob * est
            second += prob * est ** 2
            v_hat += prob * len(rest) ** 2 * (1 - m2 / len(rest)) * np.var(zs, ddof=1) / m2
    return mean, second - mean ** 2, v_hat


def test_lemma2_variance_estimator_is_unbiased_under_a_label_adaptive_second_stage():
    rng = np.random.default_rng(13)
    a = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    z = rng.normal(size=7) + a
    mean, var, v_hat = _two_stage_moments(z, a, 2, lambda lab: 2 if lab.sum() > 0 else 3)
    assert mean == pytest.approx(z.sum(), abs=1e-12)
    assert v_hat == pytest.approx(var, abs=1e-12)


def test_lemma2_variance_factor_of_a_fixed_two_stage_design():
    rng = np.random.default_rng(14)
    n, n1, m2 = 8, 2, 3
    z = rng.normal(size=n)
    _, var, _ = _two_stage_moments(z, np.zeros(n), n1, lambda lab: m2)
    single = n * (n - n1 - m2) * np.var(z, ddof=1) / (n1 + m2)
    assert var / single == pytest.approx((n - n1) * (n1 + m2) / (n * m2), abs=1e-12)


def test_proposition5_certification_rate_at_the_boundary_is_at_most_alpha(diff_pool):
    """Each response set is tested at tau = |b|, the least favourable point of the null."""
    from d2vi.decision.certify import TOSTCertifier

    certified, reps = [], 1000
    for r in range(reps):
        rng = np.random.default_rng(9000 + r)
        rs = IIDMixture().sample(diff_pool, 1500, 0.2, rng)
        y, d = rs.observed.outcome("sent"), rs.observed.flag.astype(bool)
        b = y[~d].mean() - rs.human_mean("sent")
        audit = make_design("proportional").draw(rs.observed, 300, rng, PerfectVerifier(rs.a))
        cert = TOSTCertifier(tau_sd=abs(b) / np.std(y, ddof=1))
        certified.append(cert.certify(EstimationContext(rs.observed, audit, rng), "sent")
                         .certified)
    rate = np.mean(certified)
    # one-sided 1% test of "rate <= alpha" (alpha = 0.05), and not vacuous
    assert 0.01 <= rate <= 0.05 + 2.326 * np.sqrt(0.05 * 0.95 / reps), rate


def test_proposition1_moment_inversion_recovers_the_human_mean_under_m1():
    rng = np.random.default_rng(15)
    y_h, y_a = rng.normal(size=20), rng.normal(1.0, 1.0, size=12)
    # every outcome value appears with the same flag pattern, so Cov(D, Y | A) = 0 exactly
    y = np.concatenate([np.repeat(y_h, 5), np.repeat(y_a, 4)])
    d = np.concatenate([np.tile([1, 0, 0, 0, 0], 20), np.tile([1, 1, 1, 0], 12)]).astype(float)
    a = np.concatenate([np.zeros(100), np.ones(48)])
    eps, f, s = a.mean(), d[a == 0].mean(), d[a == 1].mean()
    system = np.array([[(1 - eps) * f, eps * s], [(1 - eps) * (1 - f), eps * (1 - s)]])
    theta = np.linalg.solve(system, [np.mean(y * d), np.mean(y * (1 - d))])
    assert theta[0] == pytest.approx(y[a == 0].mean(), abs=1e-12)
    assert theta[1] == pytest.approx(y[a == 1].mean(), abs=1e-12)


def test_remark2_package_policy_follows_the_sqrt2_rule(diff_pool, pool):
    from d2vi.decision.policy import MSEPolicy

    seen = set()
    for r in range(300):
        rng = np.random.default_rng(9500 + r)
        src, eps = (diff_pool, 0.2) if r % 2 else (pool, 0.02)
        rs = IIDMixture().sample(src, 900, eps, rng)
        audit = make_design("proportional").draw(rs.observed, 45, rng, PerfectVerifier(rs.a))
        ctx = EstimationContext(rs.observed, audit, rng)
        anchor = make_estimator("rc_ppi_pp").estimate(ctx, "sent")
        if not anchor.feasible:
            anchor = make_estimator("flag_ppi").estimate(ctx, "sent")
        res = MSEPolicy().choose(ctx, "sent")
        b = {k: abs(make_estimator(k).estimate(ctx, "sent").point - anchor.point)
             for k in ("naive", "deletion")}
        limit = np.sqrt(2) * anchor.se
        seen.add(res.choice)
        mse = {k: max(v ** 2 - anchor.se ** 2, 0.0) for k, v in b.items()}
        if res.choice in b:
            assert b[res.choice] < limit and mse[res.choice] == min(mse.values())
            if mse["naive"] == mse["deletion"]:
                assert res.choice == "naive"                  # tie, stated in Remark 2
        else:
            assert min(b.values()) >= limit
    assert {"naive", "deletion"} & seen and len(seen) >= 2       # both branches exercised
