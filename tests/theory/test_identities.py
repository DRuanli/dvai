"""Exact finite-population checks of Lemma 1 and Proposition 2 (no Monte Carlo error)."""

import numpy as np
import pytest

from d2vi.estimators import EstimationContext, MomentInversion
from d2vi.sampling.audit import PerfectVerifier, make_design
from d2vi.sampling.mixture import IIDMixture


def _cov0(x, y):
    return float(np.mean((x - x.mean()) * (y - y.mean())))


@pytest.mark.parametrize("seed", range(5))
def test_lemma1_deletion_bias_identity(diff_pool, seed):
    """mean(Y | D=0) - mu_H = eps_res * Delta_surv - Cov(D_H, Y_H) / (1 - mean D_H)."""
    rng = np.random.default_rng(seed)
    rs = IIDMixture().sample(diff_pool, 900, 0.2, rng)
    y, d, a = rs.observed.outcome("sent"), rs.observed.flag, rs.a
    h, m = a == 0, a == 1
    lhs = y[d == 0].mean() - y[h].mean()
    eps_res = a[d == 0].mean()
    delta_surv = y[m & (d == 0)].mean() - y[h & (d == 0)].mean()
    rhs = eps_res * delta_surv - _cov0(d[h].astype(float), y[h]) / (1 - d[h].mean())
    assert lhs == pytest.approx(rhs, abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_proposition2_moment_bias_closed_form(diff_pool, seed):
    """Moment inversion with marginal (s, f) is off by -[Cov_H + eps/(1-eps) Cov_A]/(s - f)."""
    rng = np.random.default_rng(seed)
    rs = IIDMixture().sample(diff_pool, 900, 0.2, rng)
    y, d, a = rs.observed.outcome("sent"), rs.observed.flag.astype(float), rs.a
    h, m = a == 0, a == 1
    eps, s, f = a.mean(), d[m].mean(), d[h].mean()
    predicted = -(_cov0(d[h], y[h]) + eps / (1 - eps) * _cov0(d[m], y[m])) / (s - f)
    # a full audit makes the estimated (s, f, eps) equal to the realised ones
    audit = make_design("srs").draw(rs.observed, rs.n, rng, PerfectVerifier(rs.a))
    est = MomentInversion().estimate(EstimationContext(rs.observed, audit, rng), "sent")
    assert est.point - rs.human_mean("sent") == pytest.approx(predicted, abs=1e-10)
