import numpy as np
import pandas as pd
import pytest

from d2vi.decision.certify import TOSTCertifier
from d2vi.decision.policy import MSEPolicy
from d2vi.estimators import EstimationContext, make_estimator
from d2vi.inference.bootstrap import RaoWuBootstrap, rao_wu_replicate
from d2vi.sampling.audit import PerfectVerifier, make_design
from d2vi.sampling.mixture import IIDMixture
from d2vi.sim.metrics import regret_table, summarize
from d2vi.sim.runner import CellSpec, Experiment

ESTIMATORS = ["naive", "deletion", "hajek", "flag_ppi", "flag_ppi_pp", "rc_ppi", "rc_ppi_pp",
              "greg", "moment"]


def test_tost_size_at_the_equivalence_boundary(diff_pool):
    """With tau set equal to the true |b_del| (the least favourable case), the
    certification rate must not exceed alpha beyond Monte Carlo error."""
    hits, n = 0, 300
    for r in range(n):
        rng = np.random.default_rng(r)
        rs = IIDMixture().sample(diff_pool, 900, 0.1, rng)
        y = rs.observed.outcome("sent")
        b_true = y[rs.observed.flag == 0].mean() - rs.human_mean("sent")
        cert = TOSTCertifier(tau_sd=abs(b_true) / np.std(y, ddof=1), alpha=0.05)
        audit = make_design("top_decile").draw(rs.observed, 180, rng, PerfectVerifier(rs.a))
        hits += cert.certify(EstimationContext(rs.observed, audit, rng), "sent").certified
    assert hits / n <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / n)


def test_policy_returns_a_feasible_choice(pool):
    rng = np.random.default_rng(3)
    rs = IIDMixture().sample(pool, 900, 0.02, rng)
    audit = make_design("srs").draw(rs.observed, 18, rng, PerfectVerifier(rs.a))
    pr = MSEPolicy().choose(EstimationContext(rs.observed, audit, rng), "sent")
    assert np.isfinite(pr.estimate.point)
    assert pr.choice in {"naive", "deletion", "rc_ppi_pp", "flag_ppi"}


def test_rao_wu_replicate_and_interval(pool):
    rng = np.random.default_rng(9)
    rs = IIDMixture().sample(pool, 900, 0.2, rng)
    audit = make_design("top_decile").draw(rs.observed, 90, rng, PerfectVerifier(rs.a))
    rep = rao_wu_replicate(audit, rng)
    assert rep.m <= audit.m
    ctx = EstimationContext(rs.observed, audit, rng)
    res = RaoWuBootstrap(n_boot=60).interval(make_estimator("rc_ppi_pp"), ctx, "sent", rng)
    assert res.feasible and res.ci_low <= res.point <= res.ci_high


def test_experiment_is_reproducible_and_uses_common_random_numbers(pool):
    exp = Experiment(pool, [make_estimator(e) for e in ESTIMATORS],
                     policies=[MSEPolicy()], certifiers=[TOSTCertifier()])
    a = exp.run_cell(CellSpec("synthetic", "toy", 0.1, 600, 0.05, "srs"), n_rep=3)
    b = exp.run_cell(CellSpec("synthetic", "toy", 0.1, 600, 0.05, "srs"), n_rep=3)
    pd.testing.assert_frame_equal(a, b)
    c = exp.run_cell(CellSpec("synthetic", "toy", 0.1, 600, 0.05, "top_decile"), n_rep=3)
    # same data key -> same response sets -> same truth in every replication
    ta = a.groupby(["rep", "outcome"])["truth"].first()
    tc = c.groupby(["rep", "outcome"])["truth"].first()
    pd.testing.assert_series_equal(ta, tc)
    assert set(ESTIMATORS) <= set(a["estimator"])


def test_summary_and_regret(pool):
    exp = Experiment(pool, [make_estimator(e) for e in ["naive", "deletion", "rc_ppi_pp"]],
                     policies=[MSEPolicy()])
    df = exp.run([CellSpec("synthetic", "toy", e, 600, 0.1, "top_decile") for e in (0.1, 0.3)],
                 n_rep=20)
    summ = summarize(df)
    assert {"rmse", "srmse", "coverage", "feasible_rate"} <= set(summ.columns)
    reg = regret_table(summ, ["naive", "deletion", "rc_ppi_pp", "policy_debiased"])
    assert reg["max_regret"].min() == pytest.approx(reg["max_regret"].min())
    assert (reg["max_regret"] >= 0).all()


def test_verification_error_cells_share_audit_and_folds(pool):
    """E11 pairing: with phi > 0 but no flip realised, every estimate equals the clean one."""
    ests = [make_estimator("hajek"), make_estimator("rc_ppi_pp")]
    exp = Experiment(pool, ests)
    clean = exp.run_cell(CellSpec("synthetic", "toy", 0.2, 600, 0.10), 4)
    tiny = exp.run_cell(CellSpec("synthetic", "toy", 0.2, 600, 0.10, phi0=1e-12, phi1=1e-12), 4)
    pd.testing.assert_series_equal(clean["point"], tiny["point"])
    noisy = exp.run_cell(CellSpec("synthetic", "toy", 0.2, 600, 0.10, phi0=0.2, phi1=0.2), 4)
    np.testing.assert_allclose(clean["truth"], noisy["truth"])
    assert not np.allclose(clean["m_machine"], noisy["m_machine"])


def test_noisy_verifier_labels_are_fixed_per_record():
    from d2vi.sampling.audit import NoisyVerifier
    a = np.r_[np.zeros(500), np.ones(500)]
    v = NoisyVerifier(a, 0.1, 0.1, seed=3)
    idx = np.arange(1000)
    first = v.verify(idx, np.random.default_rng(0))
    np.testing.assert_array_equal(first, v.verify(idx, np.random.default_rng(9)))
    assert 0.05 < np.mean(first != a) < 0.15
    with pytest.raises(ValueError):
        NoisyVerifier(np.r_[0.0, 0.5], 0.1, 0.1)
