"""Monte Carlo checks of Theorem 1 (consistency without M1), Theorem 2 (coverage),
Proposition 3 (audit starvation) and Proposition 6 (random label noise)."""

import numpy as np
import pytest
from scipy import stats

from d2vi.estimators import RCPPI, AuditHajek, EstimationContext, MomentInversion
from d2vi.sampling.audit import NoisyVerifier, PerfectVerifier, make_design
from d2vi.sampling.mixture import IIDMixture

R = 400


@pytest.fixture(scope="module")
def runs(diff_pool):
    """R replications on a population that violates non-differentiality."""
    out = {"rc": [], "rc_cover": [], "mom": [], "hajek_cover": []}
    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        rs = IIDMixture().sample(diff_pool, 900, 0.2, rng)
        audit = make_design("srs").draw(rs.observed, 90, rng, PerfectVerifier(rs.a))
        ctx = EstimationContext(rs.observed, audit, rng)
        truth = rs.human_mean("sent")
        rc = RCPPI(tune=True).estimate(ctx, "sent")
        if rc.feasible:
            out["rc"].append(rc.point - truth)
            out["rc_cover"].append(rc.ci_low <= truth <= rc.ci_high)
        hj = AuditHajek().estimate(ctx, "sent")
        out["hajek_cover"].append(hj.ci_low <= truth <= hj.ci_high)
        mo = MomentInversion().estimate(ctx, "sent")
        if mo.feasible:
            out["mom"].append(mo.point - truth)
    return {k: np.asarray(v) for k, v in out.items()}


def test_theorem1_rcppi_nearly_unbiased_when_m1_fails(runs):
    e = runs["rc"]
    rmse = np.sqrt(np.mean(e ** 2))
    assert abs(e.mean()) < 0.15 * rmse


def test_moment_inversion_is_biased_when_m1_fails(runs):
    e = runs["mom"]
    assert abs(np.median(e)) > 2 * np.sqrt(np.mean(runs["rc"] ** 2))


def test_theorem2_linearized_interval_coverage(runs):
    for key in ("rc_cover", "hajek_cover"):
        cov = runs[key].mean()
        assert 0.90 <= cov <= 0.99, (key, cov)


def test_proposition3_starvation_probability(pool):
    rng = np.random.default_rng(0)
    rs = IIDMixture().sample(pool, 900, 0.05, rng)
    counts = np.array([make_design("srs").draw(rs.observed, 45, rng,
                                               PerfectVerifier(rs.a)).n_machine
                       for _ in range(4000)])
    p_sim = np.mean(counts >= 6)
    p_hyp = stats.hypergeom(900, 45, 45).sf(5)  # N, K machines, m draws
    assert p_sim == pytest.approx(p_hyp, abs=0.012)
    assert p_hyp < 0.05


def test_proposition6_only_missed_machines_bias_the_estimate(pool):
    rng = np.random.default_rng(1)
    rs = IIDMixture().sample(pool, 4000, 0.3, rng)
    eps, mu_h, mu_a = rs.contamination(), rs.human_mean("sent"), rs.machine_mean("sent")
    for phi0, phi1 in [(0.10, 0.0), (0.0, 0.10), (0.05, 0.20)]:
        errs = []
        for _ in range(60):
            audit = make_design("srs").draw(rs.observed, 4000, rng,
                                            NoisyVerifier(rs.a, phi0, phi1,
                                                          seed=int(rng.integers(2**31))))
            errs.append(AuditHajek().estimate(EstimationContext(rs.observed, audit, rng),
                                              "sent").point - mu_h)
        predicted = phi1 * eps * (mu_a - mu_h) / ((1 - phi0) * (1 - eps) + phi1 * eps)
        assert np.mean(errs) == pytest.approx(predicted, abs=0.004)
