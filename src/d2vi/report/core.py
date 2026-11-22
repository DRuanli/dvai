"""The D2VI protocol applied to an analyst's own response set (Section 6 of the paper).

Two steps, because verification happens outside the software:

1. :func:`plan_audit` draws the audit from the detector scores alone and returns the records
   to verify, each with its design stratum and inclusion probability;
2. once the verified origins are entered, :func:`analyse` computes the contamination rate,
   the detector's operating point, the naive, deletion and audit-based estimates of the
   human mean, the certification of deletion and the estimate the selection policy reports,
   and :func:`render_markdown` writes the report with a disclosure paragraph.

Every default is the frozen protocol's (``configs/protocol/frozen.yaml``): proportional
audit over ten score strata, RC-PPI++ with flag-PPI on the same audit when it is infeasible,
the interval rule of Remark 1, TOST with a tolerance of 0.2 SD, and the debiased selection
policy. Nothing here was used to produce the paper's results; it wraps the same estimator,
certifier and policy classes that the simulations call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import __version__
from ..data.schema import ObservedData
from ..decision.certify import TOSTCertifier
from ..decision.policy import MSEPolicy
from ..estimators.base import EstimationContext
from ..estimators.design_based import RCPPI, FlagPPI
from ..estimators.simple import Deletion, Naive
from ..inference.variance import degrees_of_freedom, ht_variance, t_interval
from ..sampling.audit import AuditSample, Verifier, make_design

__all__ = ["SINGLE_STAGE_DESIGNS", "plan_audit", "load_audit", "analyse", "render_markdown"]

# designs whose sample can be drawn before any record is verified
SINGLE_STAGE_DESIGNS = ("proportional", "srs", "top_decile", "pps")
PLAN_COLUMNS = ["row", "score", "stratum", "stratum_size", "rho", "kind", "design", "label"]
SMALLEST_EVALUATED_AUDIT = 18      # primary grid of the paper: m = 18, 45, 90 at N = 900


class _Pending(Verifier):
    """Stands in for verification while the audit is only being planned."""

    def verify(self, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.full(idx.shape[0], np.nan)


def plan_audit(score: np.ndarray, m: int, design: str | dict = "proportional",
               seed: int = 20261109, ids: np.ndarray | None = None) -> pd.DataFrame:
    """Draw an audit of exactly ``m`` records from the detector scores.

    Returns one row per record to verify: its position ``row`` in the response set, the
    design ``stratum`` with its population size, the inclusion probability ``rho`` and an
    empty ``label`` column to fill with the verified origin (0 = human, 1 = machine, or the
    machine share of a hybrid response).
    """
    name = design if isinstance(design, str) else design["name"]
    if name not in SINGLE_STAGE_DESIGNS:
        raise ValueError(f"design {name!r} needs labels while it draws; choose one of "
                         f"{SINGLE_STAGE_DESIGNS}")
    score = np.asarray(score, dtype=float)
    if not np.all(np.isfinite(score)):
        raise ValueError("every record needs a finite detector score")
    obs = ObservedData(y={}, score=score, flag=np.zeros(score.shape[0], dtype=int))
    audit = make_design(design).draw(obs, int(m), np.random.default_rng(seed), _Pending())
    plan = pd.DataFrame({
        "row": audit.idx, "score": score[audit.idx], "stratum": audit.strata,
        "stratum_size": [audit.stratum_sizes[int(h)] for h in audit.strata],
        "rho": audit.rho, "kind": audit.kind, "design": audit.design, "label": np.nan})
    if ids is not None:
        plan.insert(1, "id", np.asarray(ids)[audit.idx])
    return plan


def load_audit(plan: pd.DataFrame, n: int) -> AuditSample:
    """Rebuild the audit from a plan whose ``label`` column has been filled in."""
    missing = set(PLAN_COLUMNS) - set(plan.columns)
    if missing:
        raise ValueError(f"audit file lacks columns {sorted(missing)}")
    plan = plan.sort_values("row")
    labels = pd.to_numeric(plan["label"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(labels).any():
        raise ValueError(f"{int(np.isnan(labels).sum())} audited records have no verified label; "
                         "every record of the plan must be verified (a record that cannot be "
                         "verified is a selective failure the method does not correct)")
    if np.any((labels < 0) | (labels > 1)):
        raise ValueError("labels must be 0 (human), 1 (machine) or a machine share in [0, 1]")
    idx = plan["row"].to_numpy(dtype=int)
    if np.unique(idx).shape[0] != idx.shape[0] or idx.min() < 0 or idx.max() >= n:
        raise ValueError("audit rows must be distinct positions of the response set")
    rho = plan["rho"].to_numpy(dtype=float)
    if np.any((rho <= 0) | (rho > 1)):
        raise ValueError("inclusion probabilities must lie in (0, 1]")
    kinds, designs = plan["kind"].unique(), plan["design"].unique()
    if len(kinds) != 1 or len(designs) != 1:
        raise ValueError("the audit file mixes several designs")
    sizes = plan.groupby("stratum")["stratum_size"].agg(["min", "max"])
    if (sizes["min"] != sizes["max"]).any():
        raise ValueError("a stratum has two different sizes in the audit file")
    stratum_sizes = {int(h): int(v) for h, v in sizes["min"].items()}
    if kinds[0] == "stratified" and sum(stratum_sizes.values()) != n:
        raise ValueError(f"stratum sizes sum to {sum(stratum_sizes.values())}, but the response "
                         f"set has {n} records: the plan belongs to another data file")
    return AuditSample(idx=idx, labels=labels, rho=rho, strata=plan["stratum"].to_numpy(dtype=int),
                       stratum_sizes=stratum_sizes, kind=str(kinds[0]), design=str(designs[0]))


def _design_label(name: str) -> str:
    import re
    if m := re.fullmatch(r"prop(\d+)", name):
        return f"proportional allocation over {m.group(1)} detector-score strata"
    if m := re.fullmatch(r"top(\d+)split", name):
        return f"top-{m.group(1)}% score split"
    if m := re.fullmatch(r"pps(\d+)", name):
        return "unequal-probability sampling on the score rank"
    return {"srs": "simple random sampling"}.get(name, name)


def _weighted_rate(w: np.ndarray, part: np.ndarray, flag: np.ndarray) -> float:
    total = float(np.sum(w * part))
    return float(np.sum(w * part * flag) / total) if total > 0 else float("nan")


@dataclass(frozen=True)
class _Row:
    name: str
    point: float
    low: float = float("nan")
    high: float = float("nan")
    feasible: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        return {"estimate": self.name, "point": self.point, "ci_low": self.low,
                "ci_high": self.high, "feasible": self.feasible, "note": self.note}


def analyse(y: dict[str, np.ndarray], score: np.ndarray, flag: np.ndarray, plan: pd.DataFrame,
            length: np.ndarray | None = None, tau_sd: float = 0.2, alpha: float = 0.05,
            seed: int = 20261109) -> dict:
    """Run the protocol on one response set and its verified audit."""
    score = np.asarray(score, dtype=float)
    flag = np.asarray(flag).astype(int)
    cov = {} if length is None else {"loglen": np.log1p(np.asarray(length, dtype=float))}
    obs = ObservedData(y={k: np.asarray(v, dtype=float) for k, v in y.items()}, score=score,
                       flag=flag, covariates=cov)
    for name, values in obs.y.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"outcome {name!r} has missing values; the estimand is the human "
                             "mean of a complete response set (assumption D3)")
    audit = load_audit(plan, obs.n)
    ctx = EstimationContext(obs, audit, np.random.default_rng(seed))
    w, a, d_aud = audit.w, audit.labels, flag[audit.idx].astype(float)

    eps_hat = float(np.sum(w * a) / np.sum(w))
    se_eps = float(np.sqrt(max(ht_variance(a - eps_hat, audit), 0.0)) / np.sum(w))
    lo, hi = t_interval(eps_hat, se_eps, degrees_of_freedom(audit))
    summary = {
        "records": obs.n, "audited": audit.m, "design": audit.design,
        "design_label": _design_label(audit.design),
        "strata": len(audit.stratum_sizes), "audited_machine": audit.n_machine,
        "audited_human": audit.n_human, "graded_labels": bool(np.any((a > 0) & (a < 1))),
        "contamination": eps_hat, "contamination_ci": [max(lo, 0.0), min(hi, 1.0)],
        "flag_rate": float(flag.mean()),
        "sensitivity": _weighted_rate(w, a, d_aud),
        "false_positive_rate": _weighted_rate(w, 1 - a, d_aud),
    }

    policy, certifier = MSEPolicy(), TOSTCertifier(tau_sd=tau_sd, alpha=alpha)
    outcomes, advisories = {}, []
    for o in obs.outcomes:
        naive, deletion = Naive().estimate(ctx, o), Deletion().estimate(ctx, o)
        rc, fl = RCPPI(tune=True).estimate(ctx, o), FlagPPI().estimate(ctx, o)
        anchor = rc if rc.feasible else fl
        choice = policy.choose(ctx, o)
        cert = certifier.certify(ctx, o)
        rows = [
            _Row("naive (all records)", naive.point),
            _Row("deletion (unflagged records)", deletion.point, feasible=deletion.feasible,
                 note=deletion.reason),
            _Row("RC-PPI++", rc.point, rc.ci_low, rc.ci_high, rc.feasible, rc.reason),
            _Row("flag-PPI", fl.point, fl.ci_low, fl.ci_high, fl.feasible, fl.reason),
        ]
        outcomes[o] = {
            "sd_all_records": float(np.std(obs.outcome(o), ddof=1)),
            "estimates": [r.as_dict() for r in rows],
            "audit_based": {"estimator": anchor.estimator, "point": anchor.point, "se": anchor.se,
                            "ci_low": anchor.ci_low, "ci_high": anchor.ci_high,
                            "feasible": anchor.feasible},
            "reported": {"choice": choice.choice, "point": choice.estimate.point,
                         "ci_low": choice.estimate.ci_low, "ci_high": choice.estimate.ci_high,
                         "note": choice.reason},
            "certification": {"certified": cert.certified, "deletion_bias": cert.b_hat,
                              "se": cert.se, "tolerance": cert.tau, "tau_sd": tau_sd,
                              "alpha": alpha, "note": cert.reason},
        }
        if not rc.feasible and rc.reason not in advisories:
            advisories.append(rc.reason)
    if advisories:
        advisories = ["RC-PPI++ is not feasible (" + "; ".join(advisories) + "): the audit-based "
                      "estimate is flag-PPI on the same audit. A larger audit, or one that "
                      "contains more machine-written records, makes RC-PPI++ feasible."]
    if audit.m < SMALLEST_EVALUATED_AUDIT:
        advisories.append(f"The audit has {audit.m} records. The paper's primary evaluation uses "
                          f"audits of {SMALLEST_EVALUATED_AUDIT} to 90 records; with 9 records "
                          "the audit-based estimates were on average less accurate than deletion.")
    if summary["graded_labels"]:
        advisories.append("Some labels are machine shares between 0 and 1. The estimators accept "
                          "them, but the paper's evaluation uses binary origins only.")
    return {"d2vi": __version__, "seed": seed, "summary": summary, "outcomes": outcomes,
            "advisories": advisories}


# -------------------------------------------------------------------------------- report
def _n(x: float, nd: int = 4) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _ci(lo: float, hi: float, nd: int = 4) -> str:
    return "" if not (np.isfinite(lo) and np.isfinite(hi)) else f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def disclosure(res: dict, outcome: str) -> str:
    s, o = res["summary"], res["outcomes"][outcome]
    rep, ab, cert = o["reported"], o["audit_based"], o["certification"]
    name = {"rc_ppi_pp": "RC-PPI++", "flag_ppi": "flag-PPI"}.get(ab["estimator"], ab["estimator"])
    rule = {"naive": "the mean of all records", "deletion": "the mean of unflagged records",
            "rc_ppi_pp": "the RC-PPI++ estimate",
            "flag_ppi": "the flag-PPI estimate"}[rep["choice"]]
    verdict = ("was certified as equivalent to the audit-based estimate" if cert["certified"]
               else "was not certified as equivalent to the audit-based estimate")
    return (
        f"All {s['records']} responses were scored by [detector name and version]; scores and "
        f"flags were kept for every record. The origin of {s['audited']} responses was verified "
        f"by [verification mechanism] in a probability audit ({s['design_label']}; known "
        f"inclusion probabilities), which found "
        f"{_n(s['audited_machine'], 0)} machine-written responses. The estimated contamination "
        f"rate is {_n(100 * s['contamination'], 1)}% (95% interval "
        f"{_n(100 * s['contamination_ci'][0], 1)}% to {_n(100 * s['contamination_ci'][1], 1)}%); "
        f"the detector's estimated sensitivity is {_n(s['sensitivity'], 2)} and its "
        f"false-positive rate {_n(s['false_positive_rate'], 2)}. For {outcome}, we report "
        f"{rule}, {_n(rep['point'])}, selected by the audit-anchored policy, with the 95% "
        f"design-based interval of the {name} estimate, {_ci(rep['ci_low'], rep['ci_high'])}. "
        f"Deleting flagged responses {verdict} within {cert['tau_sd']:g} standard deviations of "
        f"the outcome (two one-sided tests, level {cert['alpha']:g}). Analysis: d2vi "
        f"{res['d2vi']}.")


def render_markdown(res: dict) -> str:
    s = res["summary"]
    out = ["# D2VI report", "",
           f"- Records: {s['records']}; audited: {s['audited']} ({s['design_label']})",
           f"- Verified machine-written: {_n(s['audited_machine'], 1)}; verified human: "
           f"{_n(s['audited_human'], 1)}",
           f"- Estimated contamination rate: {_n(s['contamination'], 3)} (approximate 95% "
           f"interval {_ci(*s['contamination_ci'], nd=3)}; from the audit alone)",
           f"- Detector flag rate: {_n(s['flag_rate'], 3)}; estimated sensitivity "
           f"{_n(s['sensitivity'], 3)}; estimated false-positive rate "
           f"{_n(s['false_positive_rate'], 3)} (design-weighted, from the audit)", ""]
    for name, o in res["outcomes"].items():
        rep, cert = o["reported"], o["certification"]
        out += [f"## Outcome: {name}", "",
                "| Estimate of the human mean | Value | 95% interval | Note |", "|---|---|---|---|"]
        for r in o["estimates"]:
            value = _n(r["point"]) if r["feasible"] else "not feasible"
            out.append(f"| {r['estimate']} | {value} | {_ci(r['ci_low'], r['ci_high'])} | "
                       f"{r['note']} |")
        label = {"naive": "naive", "deletion": "deletion", "rc_ppi_pp": "RC-PPI++",
                 "flag_ppi": "flag-PPI"}[rep["choice"]]
        out += ["", f"**Reported by the selection policy:** {_n(rep['point'])} ({label}), with the "
                    f"audit-based interval {_ci(rep['ci_low'], rep['ci_high'])}.", ""]
        if np.isfinite(cert["deletion_bias"]):
            verdict = "certified" if cert["certified"] else "not certified"
            out.append(f"**Is deletion safe?** Estimated deletion bias {_n(cert['deletion_bias'])} "
                       f"(SE {_n(cert['se'])}); tolerance {_n(cert['tolerance'])} "
                       f"({cert['tau_sd']:g} SD). Deletion is **{verdict}** at level "
                       f"{cert['alpha']:g}.")
        else:
            out.append(f"**Is deletion safe?** Not certified: {cert['note']}.")
        out += ["", "**Disclosure paragraph.** " + disclosure(res, name), ""]
    if res["advisories"]:
        out += ["## Advisories", ""] + [f"- {a}" for a in res["advisories"]] + [""]
    out.append(f"_d2vi {res['d2vi']}; cross-fitting seed {res['seed']}._")
    return "\n".join(out) + "\n"
