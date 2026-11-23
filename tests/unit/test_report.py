"""``d2vi-report``: the protocol on an analyst's own response set."""

import json

import numpy as np
import pandas as pd
import pytest

from d2vi.decision.certify import TOSTCertifier
from d2vi.decision.policy import MSEPolicy
from d2vi.estimators.base import EstimationContext
from d2vi.estimators.design_based import RCPPI
from d2vi.report import analyse, load_audit, plan_audit, render_markdown
from d2vi.report.cli import main
from d2vi.sampling.audit import PerfectVerifier, make_design
from d2vi.sampling.mixture import make_mixture

SEED = 20261109


@pytest.fixture(scope="module")
def rs(pool):
    return make_mixture("iid").sample(pool, 900, 0.20, np.random.default_rng(5))


def _filled(rs, m=45, design="proportional", seed=7):
    plan = plan_audit(rs.observed.score, m, design, seed)
    plan["label"] = rs.a[plan["row"].to_numpy()]
    return plan


@pytest.mark.parametrize("design", ["proportional", "srs", "top_decile", "pps"])
def test_plan_is_the_package_design_with_exact_size_and_known_probabilities(rs, design):
    plan = plan_audit(rs.observed.score, 45, design, seed=3)
    ref = make_design(design).draw(rs.observed, 45, np.random.default_rng(3), PerfectVerifier(rs.a))
    assert len(plan) == 45 and plan["label"].isna().all()
    assert plan["row"].tolist() == ref.idx.tolist()
    assert np.allclose(plan["rho"], ref.rho)
    assert plan["rho"].between(0, 1, inclusive="right").all()
    audit = load_audit(plan.assign(label=rs.a[plan["row"].to_numpy()]), rs.n)
    assert audit.kind == ref.kind and audit.stratum_sizes == ref.stratum_sizes
    assert np.array_equal(audit.labels, ref.labels)


def test_label_adaptive_designs_cannot_be_planned(rs):
    with pytest.raises(ValueError, match="needs labels"):
        plan_audit(rs.observed.score, 45, "neyman")


def test_analysis_equals_the_estimators_policy_and_certifier_of_the_package(rs):
    plan = _filled(rs)
    obs = rs.observed
    res = analyse(obs.y, obs.score, obs.flag, plan, np.expm1(obs.covariates["loglen"]), seed=SEED)
    ctx = EstimationContext(obs, load_audit(plan, rs.n), np.random.default_rng(SEED))
    for o in obs.outcomes:
        out = res["outcomes"][o]
        rc = RCPPI(tune=True).estimate(ctx, o)
        assert out["audit_based"]["point"] == pytest.approx(rc.point, abs=1e-9)
        assert out["audit_based"]["ci_low"] == pytest.approx(rc.ci_low, abs=1e-9)
        pol = MSEPolicy().choose(ctx, o)
        assert out["reported"]["choice"] == pol.choice
        assert out["reported"]["point"] == pytest.approx(pol.estimate.point, abs=1e-9)
        cert = TOSTCertifier().certify(ctx, o)
        assert out["certification"]["certified"] == cert.certified
        assert out["certification"]["deletion_bias"] == pytest.approx(cert.b_hat, abs=1e-9)
    s = res["summary"]
    assert s["audited"] == 45 and s["audited_machine"] + s["audited_human"] == 45
    assert s["contamination_ci"][0] <= s["contamination"] <= s["contamination_ci"][1]


def test_operating_points_are_design_weighted(rs):
    # a top-decile audit over-represents flagged records; unweighted rates would be biased
    plan = _filled(rs, m=90, design="top_decile")
    res = analyse(rs.observed.y, rs.observed.score, rs.observed.flag, plan)
    w = 1 / plan["rho"].to_numpy()
    a, d = plan["label"].to_numpy(), rs.observed.flag[plan["row"].to_numpy()]
    assert res["summary"]["false_positive_rate"] == pytest.approx(
        np.sum(w * (1 - a) * d) / np.sum(w * (1 - a)))
    assert res["summary"]["sensitivity"] == pytest.approx(np.sum(w * a * d) / np.sum(w * a))


def test_infeasible_rc_ppi_is_reported_with_its_fallback_and_an_advisory(pool):
    low = make_mixture("iid").sample(pool, 900, 0.01, np.random.default_rng(1))
    plan = plan_audit(low.observed.score, 18, seed=2)
    plan["label"] = low.a[plan["row"].to_numpy()]
    res = analyse(low.observed.y, low.observed.score, low.observed.flag, plan)
    out = res["outcomes"]["sent"]
    assert out["audit_based"]["estimator"] == "flag_ppi"
    assert not out["estimates"][2]["feasible"]                 # the RC-PPI++ row
    assert any("flag-PPI on the same audit" in a for a in res["advisories"])
    text = render_markdown(res)
    assert "not feasible" in text and "## Advisories" in text


def test_unverified_or_foreign_plans_are_refused(rs):
    plan = plan_audit(rs.observed.score, 45, seed=1)
    with pytest.raises(ValueError, match="no verified label"):
        load_audit(plan, rs.n)
    filled = plan.assign(label=0.0)
    with pytest.raises(ValueError, match="another data file"):
        load_audit(filled, rs.n + 10)
    with pytest.raises(ValueError, match="machine share"):
        load_audit(plan.assign(label=2.0), rs.n)


def test_command_line_round_trip(rs, tmp_path, capsys):
    obs = rs.observed
    data = tmp_path / "responses.csv"
    pd.DataFrame({"rating": obs.y["sent"], "score": obs.score, "flagged": obs.flag,
                  "words": np.expm1(obs.covariates["loglen"])}).to_csv(data, index=False)
    plan_path = tmp_path / "audit.csv"
    main(["plan", str(data), "--score", "score", "--budget", "0.05", "--out", str(plan_path)])
    plan = pd.read_csv(plan_path)
    assert len(plan) == 45
    plan["label"] = rs.a[plan["row"].to_numpy()]
    plan.to_csv(plan_path, index=False)
    report, numbers = tmp_path / "report.md", tmp_path / "report.json"
    main(["estimate", str(data), str(plan_path), "--outcome", "rating", "--score", "score",
          "--flag", "flagged", "--length", "words", "--out", str(report), "--json", str(numbers)])
    res = json.loads(numbers.read_text())
    ref = analyse({"rating": obs.y["sent"]}, obs.score, obs.flag, plan,
                  np.expm1(obs.covariates["loglen"]))
    assert res["outcomes"]["rating"]["reported"]["point"] == pytest.approx(
        ref["outcomes"]["rating"]["reported"]["point"])
    text = report.read_text()
    assert "Disclosure paragraph" in text and "Is deletion safe?" in text
    with pytest.raises(SystemExit, match="exactly one of --flag"):
        main(["estimate", str(data), str(plan_path), "--outcome", "rating", "--score", "score"])
