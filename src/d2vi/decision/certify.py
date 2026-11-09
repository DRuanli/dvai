"""Certifying detect-and-delete with an equivalence test (Proposition 5).

The deletion bias is ``b_del = mean(y | D = 0) - mu_H``. For the finite-population
estimand the deletion mean is a fixed number, so ``b_hat = ybar_{D=0} - mu_hat_anchor`` has
the anchor's standard error. Deletion is certified when the ``1 - 2 alpha`` interval for
``b_del`` lies inside ``(-tau, tau)``: the two one-sided tests (TOST) of Schuirmann (1987).
Asymptotically the probability of certifying when ``|b_del| >= tau`` is at most ``alpha``.

``tau`` is the smallest bias of interest, declared before looking at the data, in units of
the outcome's standard deviation over all records (default 0.2, Cohen's small effect).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..estimators.base import EstimationContext, Estimator
from ..estimators.design_based import RCPPI
from ..estimators.simple import Deletion

__all__ = ["CertificationResult", "TOSTCertifier", "LegacyCertifier"]


@dataclass(frozen=True)
class CertificationResult:
    certified: bool
    b_hat: float
    se: float
    tau: float
    reason: str = ""


class TOSTCertifier:
    def __init__(self, tau_sd: float = 0.2, alpha: float = 0.05,
                 anchor: Estimator | None = None) -> None:
        self.tau_sd = float(tau_sd)
        self.alpha = float(alpha)
        self.anchor = anchor or RCPPI(tune=True)

    def certify(self, ctx: EstimationContext, outcome: str) -> CertificationResult:
        y = ctx.obs.outcome(outcome)
        tau = self.tau_sd * float(np.std(y, ddof=1))
        anchor = self.anchor.estimate(ctx, outcome)
        if not anchor.feasible or not np.isfinite(anchor.se):
            return CertificationResult(False, float("nan"), float("nan"), tau,
                                       f"anchor infeasible: {anchor.reason}")
        deletion = Deletion().estimate(ctx, outcome)
        if not deletion.feasible:
            return CertificationResult(False, float("nan"), anchor.se, tau, deletion.reason)
        b_hat = deletion.point - anchor.point
        z = stats.norm.ppf(1 - self.alpha)
        ok = abs(b_hat) + z * anchor.se < tau
        return CertificationResult(bool(ok), float(b_hat), float(anchor.se), tau,
                                   "" if ok else "equivalence not shown")


class LegacyCertifier:
    """v1 rule (Step 4b): certify if ``|b_hat| < 2 sd`` and ``2 sd < 0.15 SD(Y)``.

    ``b_hat`` and ``sd`` come from a nonparametric bootstrap of the audited humans'
    design-weighted mean. Kept only to measure its false-certification rate (E5).
    """

    def __init__(self, n_boot: int = 200, tau_sd: float = 0.15) -> None:
        self.n_boot = int(n_boot)
        self.tau_sd = float(tau_sd)

    def certify(self, ctx: EstimationContext, outcome: str) -> CertificationResult:
        audit, y = ctx.audit, ctx.obs.outcome(outcome)
        deletion = float(np.mean(y[ctx.obs.flag == 0]))
        tau = self.tau_sd * float(np.std(y, ddof=1))
        diffs = []
        for _ in range(self.n_boot):
            b = ctx.rng.integers(0, audit.m, audit.m)
            h = audit.labels[b] < 0.5
            if h.sum() < 5:
                continue
            wh = audit.w[b][h]
            diffs.append(deletion - float(np.sum(wh * y[audit.idx[b][h]]) / np.sum(wh)))
        if len(diffs) < 10:
            return CertificationResult(False, float("nan"), float("nan"), tau, "too few humans")
        b_hat, sd = float(np.mean(diffs)), float(np.std(diffs))
        ok = abs(b_hat) < 2 * sd and 2 * sd < tau
        return CertificationResult(bool(ok), b_hat, sd, tau, "")
