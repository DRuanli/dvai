"""Estimators that ignore or only partly use the audit: naive, deletion, moment inversion."""

from __future__ import annotations

import numpy as np

from .base import EstimateResult, EstimationContext, Estimator

__all__ = ["Naive", "Deletion", "MomentInversion"]


class Naive(Estimator):
    """Mean of every record, contamination ignored."""

    name = "naive"
    needs_audit = False

    def estimate(self, ctx: EstimationContext, outcome: str) -> EstimateResult:
        return EstimateResult(self.name, outcome, float(np.mean(ctx.obs.outcome(outcome))))


class Deletion(Estimator):
    """Mean of the records the detector does not flag (detect-and-delete)."""

    name = "deletion"
    needs_audit = False

    def estimate(self, ctx: EstimationContext, outcome: str) -> EstimateResult:
        keep = ctx.obs.flag == 0
        if not keep.any():
            return EstimateResult.infeasible(self.name, outcome, "every record is flagged")
        return EstimateResult(self.name, outcome, float(np.mean(ctx.obs.outcome(outcome)[keep])))


class MomentInversion(Estimator):
    """Audit-light moment inversion (Proposition 1), stratified by a style covariate.

    Within each stratum ``g`` the audit gives design-weighted estimates of the
    sensitivity ``s_g`` and false-positive rate ``f_g``; the contamination rate follows
    from the Rogan-Gladen inversion ``eps_g = (p_g - f_g) / (s_g - f_g)``, and
    ``(mu_H, mu_A)`` solve the 2x2 system in the flagged and unflagged means. Strata are
    combined with weights ``N_g (1 - eps_g)``. Valid only under non-differentiality (M1)
    and ``s_g != f_g`` (M2); Proposition 2 gives its bias when M1 fails.
    """

    name = "moment"

    def __init__(self, stratum: str | None = None, min_per_class: float = 3.0) -> None:
        self.stratum = stratum
        self.min_per_class = float(min_per_class)
        if stratum:
            self.name = f"moment_{stratum}"

    def estimate(self, ctx: EstimationContext, outcome: str) -> EstimateResult:
        audit = ctx.audit
        if audit is None:
            return EstimateResult.infeasible(self.name, outcome, "no audit")
        obs = ctx.obs
        y, d = obs.outcome(outcome), obs.flag.astype(float)
        g_all = (obs.covariates[self.stratum] if self.stratum else np.zeros(obs.n))
        g_aud = g_all[audit.idx]
        parts, weights = [], []
        for g in np.unique(g_all):
            in_g = g_all == g
            in_ga = g_aud == g
            w, a, d_a = audit.w[in_ga], audit.labels[in_ga], d[audit.idx][in_ga]
            if np.sum(a) < self.min_per_class or np.sum(1 - a) < self.min_per_class:
                return EstimateResult.infeasible(self.name, outcome,
                                                 f"stratum {g}: too few audited records")
            s = np.sum(w * a * d_a) / np.sum(w * a)
            f = np.sum(w * (1 - a) * d_a) / np.sum(w * (1 - a))
            if abs(s - f) < 1e-6:
                return EstimateResult.infeasible(self.name, outcome, f"stratum {g}: s == f")
            p = d[in_g].mean()
            eps = float(np.clip((p - f) / (s - f), 0.0, 1.0))
            n_g = int(in_g.sum())
            if eps <= 1e-9:
                mu_h = y[in_g].mean()
            elif eps >= 1 - 1e-9:
                return EstimateResult.infeasible(self.name, outcome,
                                                 f"stratum {g}: estimated eps = 1")
            else:
                if p <= 0 or p >= 1:
                    return EstimateResult.infeasible(self.name, outcome,
                                                     f"stratum {g}: flag rate is 0 or 1")
                m1 = y[in_g & (d == 1)].mean()
                m0 = y[in_g & (d == 0)].mean()
                mat = np.array([[(1 - eps) * f / p, eps * s / p],
                                [(1 - eps) * (1 - f) / (1 - p), eps * (1 - s) / (1 - p)]])
                mu_h = float(np.linalg.solve(mat, np.array([m1, m0]))[0])
            parts.append(mu_h)
            weights.append(n_g * (1 - eps))
        weights = np.asarray(weights)
        return EstimateResult(self.name, outcome, float(np.dot(weights, parts) / weights.sum()))
