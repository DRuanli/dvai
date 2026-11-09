import numpy as np
import pytest

from d2vi.data.schema import ObservedData
from d2vi.estimators import (
    RCPPI,
    AuditHajek,
    CrossFitPropensity,
    Deletion,
    EstimationContext,
    FlagPPI,
    GREGRatio,
    Naive,
)
from d2vi.estimators.design_based import DifferenceRatioEstimator
from d2vi.sampling.audit import AuditSample, PerfectVerifier, make_design
from d2vi.sampling.mixture import IIDMixture


def _tiny():
    y = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 12.0])
    a = np.array([0, 0, 0, 0, 1, 1], dtype=float)
    d = np.array([0, 0, 1, 0, 1, 0])
    obs = ObservedData(y={"y": y}, score=np.array([.1, .2, .7, .3, .9, .4]), flag=d)
    audit = AuditSample(idx=np.array([0, 2, 4]), labels=a[[0, 2, 4]],
                        rho=np.full(3, 0.5), strata=np.zeros(3, dtype=int),
                        stratum_sizes={0: 6}, kind="stratified", design="manual")
    return obs, a, audit


def test_known_answer_hajek_and_flag_ppi():
    obs, a, audit = _tiny()
    ctx = EstimationContext(obs, audit)
    y, w = obs.y["y"], 2.0
    # Hajek: audited humans are records 0 and 2
    assert AuditHajek().estimate(ctx, "y").point == pytest.approx((1 + 3) / 2)
    # flag-PPI by hand: eta = D
    d = obs.flag.astype(float)
    num = np.sum(y * (1 - d)) + w * sum(y[i] * ((1 - a[i]) - (1 - d[i])) for i in (0, 2, 4))
    den = np.sum(1 - d) + w * sum((1 - a[i]) - (1 - d[i]) for i in (0, 2, 4))
    assert FlagPPI().estimate(ctx, "y").point == pytest.approx(num / den)


def test_naive_and_deletion():
    obs, _, audit = _tiny()
    ctx = EstimationContext(obs, audit)
    assert Naive().estimate(ctx, "y").point == pytest.approx(np.mean(obs.y["y"]))
    assert Deletion().estimate(ctx, "y").point == pytest.approx(np.mean([1, 2, 4, 12]))


class _EtaOne(DifferenceRatioEstimator):
    name = "eta_one"

    def _predictor(self, ctx, outcome):
        return np.ones(ctx.obs.n), np.ones(ctx.audit.m), ""


def test_eta_equal_one_reduces_to_hajek(pool):
    rng = np.random.default_rng(0)
    rs = IIDMixture().sample(pool, 900, 0.2, rng)
    audit = make_design("top_decile").draw(rs.observed, 60, rng, PerfectVerifier(rs.a))
    ctx = EstimationContext(rs.observed, audit, rng)
    h = AuditHajek().estimate(ctx, "sent")
    e = _EtaOne().estimate(ctx, "sent")
    assert e.point == pytest.approx(h.point)
    assert e.se == pytest.approx(h.se)


@pytest.mark.parametrize("est", [AuditHajek(), FlagPPI(), GREGRatio()])
def test_full_audit_recovers_truth_exactly(pool, est):
    rng = np.random.default_rng(2)
    rs = IIDMixture().sample(pool, 300, 0.2, rng)
    audit = make_design("srs").draw(rs.observed, 300, rng, PerfectVerifier(rs.a))
    res = est.estimate(EstimationContext(rs.observed, audit, rng), "sent")
    assert res.point == pytest.approx(rs.human_mean("sent"), abs=1e-10)
    assert res.se == pytest.approx(0.0, abs=1e-10)


def test_rcppi_reports_infeasible_without_fallback(pool):
    rng = np.random.default_rng(4)
    rs = IIDMixture().sample(pool, 900, 0.01, rng)
    audit = make_design("srs").draw(rs.observed, 20, rng, PerfectVerifier(rs.a))
    res = RCPPI().estimate(EstimationContext(rs.observed, audit, rng), "sent")
    assert not res.feasible
    assert np.isnan(res.point)
    assert "machine" in res.reason


def test_power_tuning_does_not_increase_estimated_variance(pool):
    rng = np.random.default_rng(7)
    rs = IIDMixture().sample(pool, 900, 0.2, rng)
    audit = make_design("srs").draw(rs.observed, 90, rng, PerfectVerifier(rs.a))
    ctx = EstimationContext(rs.observed, audit, rng)
    se_pp = RCPPI(tune=True).estimate(ctx, "sent").se
    se_rc = RCPPI(tune=False).estimate(ctx, "sent").se
    se_h = AuditHajek().estimate(ctx, "sent").se
    assert se_pp <= 1.02 * min(se_rc, se_h)


def test_graded_labels_are_accepted(pool):
    rng = np.random.default_rng(8)
    rs = IIDMixture().sample(pool, 600, 0.3, rng)
    a = np.clip(rs.a * rng.uniform(0.6, 1.0, rs.n), 0, 1)
    audit = make_design("srs").draw(rs.observed, 120, rng, PerfectVerifier(a))
    fit = CrossFitPropensity().fit(rs.observed, audit, rng)
    assert fit.feasible
    assert np.all((fit.eta_all >= 0) & (fit.eta_all <= 1))
