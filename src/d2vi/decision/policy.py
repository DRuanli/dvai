"""Audit-anchored selection among naive, deletion and RC-PPI++ (Algorithm 3).

1. Compute the anchor (RC-PPI++; if infeasible, flag-PPI on the same audit; if that is
   also infeasible, report the naive mean and say so). Flag-PPI replaced the two-phase
   Hajek estimator as fallback in stage S4: with 45 audited records the Hajek estimator
   has about five times its error at 1-2% contamination, where RC-PPI++ is mostly
   infeasible (``results/S4_policy_dev_regret_hajekfb.csv``).
2. Estimate the bias of naive and deletion against the anchor, ``b_k = mu_k - mu_anchor``.
3. Estimate each candidate's MSE: ``bias^2`` for naive and deletion (fixed numbers for
   the finite-population estimand), the anchor's variance for the anchor. With
   ``bias_sq="plugin"`` the squared bias is ``b_k^2``; with ``"debiased"`` it is
   ``max(b_k^2 - V_anchor, 0)``.
4. Choose the smallest estimated MSE. With ``require_certification=True`` deletion is a
   candidate only when the TOST certifier certifies it.

The configuration used in the paper is chosen on the development corpora and frozen
before the held-out evaluation (``configs/protocol``). The policy picks a point estimate;
the uncertainty statement reported with it is the anchor's interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..estimators.base import EstimateResult, EstimationContext
from ..estimators.design_based import RCPPI, FlagPPI
from ..estimators.simple import Deletion, Naive
from .certify import TOSTCertifier

__all__ = ["PolicyResult", "MSEPolicy"]


@dataclass
class PolicyResult:
    choice: str
    estimate: EstimateResult
    mse: dict
    certified: bool | None
    reason: str = ""


class MSEPolicy:
    def __init__(self, bias_sq: str = "plugin", require_certification: bool = False,
                 certifier: TOSTCertifier | None = None) -> None:
        if bias_sq not in ("plugin", "debiased"):
            raise ValueError("bias_sq must be 'plugin' or 'debiased'")
        self.bias_sq = bias_sq
        self.require_certification = bool(require_certification)
        self.anchor = RCPPI(tune=True)
        self.fallback = FlagPPI()
        self.certifier = certifier or TOSTCertifier(anchor=self.anchor)
        self.name = "policy_" + bias_sq + ("_cert" if self.require_certification else "")

    def choose(self, ctx: EstimationContext, outcome: str) -> PolicyResult:
        naive = Naive().estimate(ctx, outcome)
        deletion = Deletion().estimate(ctx, outcome)
        anchor = self.anchor.estimate(ctx, outcome)
        note = ""
        if not anchor.feasible:
            anchor = self.fallback.estimate(ctx, outcome)
            note = "anchor fell back to flag_ppi"
        if not anchor.feasible or not np.isfinite(anchor.se):
            res = EstimateResult(self.name, outcome, naive.point, feasible=True,
                                 reason="no feasible audit estimator; naive reported",
                                 extras={"choice": "naive"})
            return PolicyResult("naive", res, {}, None, res.reason)
        v = anchor.se ** 2
        mse = {anchor.estimator: v}
        for cand in (naive, deletion):
            if not cand.feasible:
                continue
            b2 = (cand.point - anchor.point) ** 2
            mse[cand.estimator] = b2 if self.bias_sq == "plugin" else max(b2 - v, 0.0)
        certified = None
        if self.require_certification:
            certified = self.certifier.certify(ctx, outcome).certified
            if not certified:
                mse.pop("deletion", None)
        choice = min(mse, key=mse.get)
        chosen = {naive.estimator: naive, deletion.estimator: deletion,
                  anchor.estimator: anchor}[choice]
        res = EstimateResult(self.name, outcome, chosen.point, anchor.se, anchor.ci_low,
                             anchor.ci_high, True, note, extras={"choice": choice})
        return PolicyResult(choice, res, mse, certified, note)
