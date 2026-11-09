"""Rao-Wu rescaling bootstrap for stratified audits, refitting the propensity each time.

Within each design stratum ``h`` with ``m_h >= 2`` audited records, a replicate draws
``m_h - 1`` records with replacement and multiplies the weight of record ``i`` by
``m_h / (m_h - 1) * c_i``, where ``c_i`` is its multiplicity (Rao and Wu, 1988). The
estimator, including the cross-fitted propensity model, is recomputed on every replicate.
Strata with a single record, and certainty strata (inclusion probability 1, such as the
pilot of a two-stage design), keep their records and weights unchanged. The finite-population
correction is ignored, which makes the interval slightly conservative.
"""

from __future__ import annotations

import numpy as np

from ..estimators.base import EstimateResult, EstimationContext, Estimator
from ..inference.variance import collapse_strata
from ..sampling.audit import AuditSample

__all__ = ["rao_wu_replicate", "RaoWuBootstrap"]


def rao_wu_replicate(audit: AuditSample, rng: np.random.Generator) -> AuditSample:
    strata = collapse_strata(audit.strata)
    mult = np.zeros(audit.m)
    scale = np.ones(audit.m)
    for h in np.unique(strata):
        members = np.flatnonzero(strata == h)
        m_h = members.shape[0]
        if m_h < 2 or np.all(audit.rho[members] >= 1.0):
            mult[members] = 1.0  # single record, or a certainty stratum (two-stage pilot)
            continue
        draws = rng.choice(members, size=m_h - 1, replace=True)
        np.add.at(mult, draws, 1.0)
        scale[members] = m_h / (m_h - 1)
    weights = audit.w * scale * mult
    keep = mult > 0
    return audit.subset(keep, weights=weights)


class RaoWuBootstrap:
    """Percentile interval from ``n_boot`` Rao-Wu replicates of an estimator."""

    def __init__(self, n_boot: int = 200, level: float = 0.95) -> None:
        self.n_boot = int(n_boot)
        self.level = float(level)

    def interval(self, estimator: Estimator, ctx: EstimationContext, outcome: str,
                 rng: np.random.Generator) -> EstimateResult:
        base = estimator.estimate(ctx, outcome)
        if not base.feasible:
            return base
        values = []
        for _ in range(self.n_boot):
            rep = rao_wu_replicate(ctx.audit, rng)
            res = estimator.estimate(ctx.with_audit(rep, rng), outcome)
            if res.feasible:
                values.append(res.point)
        values = np.asarray(values)
        if values.shape[0] < max(20, self.n_boot // 2):
            return EstimateResult(base.estimator + "+raowu", outcome, base.point,
                                  feasible=True, reason="too few feasible replicates",
                                  extras={"n_ok": int(values.shape[0])})
        alpha = 1 - self.level
        lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
        return EstimateResult(base.estimator + "+raowu", outcome, base.point,
                              float(values.std(ddof=1)), float(lo), float(hi), True, "",
                              extras={"n_ok": int(values.shape[0])})
