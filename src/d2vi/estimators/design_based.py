"""Audit-based ratio estimators of the human mean: one family, several predictors.

All estimators here have the form (Section 3.5, Corollary 1)

    N_hat = lam * sum_U y_j (1 - eta_j)    + sum_{audit} w_i y_i [(1 - a_i) - lam (1 - eta_i)]
    D_hat = lam * sum_U     (1 - eta_j)    + sum_{audit} w_i     [(1 - a_i) - lam (1 - eta_i)]
    mu_hat = N_hat / D_hat,

where ``eta`` is a predictor of the machine share and ``lam`` a power-tuning coefficient.
Special cases:

* ``lam = 0`` (or ``eta = 1``): the two-phase Hajek estimator (validation sampling);
* ``eta = D`` (the detector flag), ``lam = 1``: flag-PPI, the difference estimator;
* ``eta`` = cross-fitted logistic propensity, ``lam = 1``: RC-PPI;
* RC-PPI with ``lam`` chosen to minimise the estimated variance: RC-PPI++;
* ``eta`` from a weighted linear working model: GREG (model-assisted estimation).

The linearised residual on an audited record is ``z_i = [(1 - a_i) - lam (1 - eta_i)]
(y_i - mu_hat)``, and the standard error is ``sqrt(ht_variance(z)) / D_hat``.
"""

from __future__ import annotations

import numpy as np

from ..inference.variance import ht_covariance, ht_variance, t_interval, variance_report
from .base import EstimateResult, EstimationContext, Estimator
from .propensity import CrossFitPropensity

__all__ = [
    "DifferenceRatioEstimator",
    "AuditHajek",
    "FlagPPI",
    "RCPPI",
    "GREGRatio",
    "tune_lambda",
]


def _ratio(y, a_aud, w, eta_all, eta_aud, idx, lam):
    num = lam * np.sum(y * (1 - eta_all)) + np.sum(w * y[idx] * ((1 - a_aud) - lam * (1 - eta_aud)))
    den = lam * np.sum(1 - eta_all) + np.sum(w * ((1 - a_aud) - lam * (1 - eta_aud)))
    return num, den


def tune_lambda(y_aud, a_aud, eta_aud, mu, audit, clip: tuple[float, float] = (0.0, 1.0)) -> float:
    """Variance-minimising ``lam`` (PPI++-style) for the ratio's linearised residual."""
    u = (1 - a_aud) * (y_aud - mu)
    v = (1 - eta_aud) * (y_aud - mu)
    vv = ht_variance(v, audit)
    if vv <= 0:
        return 1.0
    return float(np.clip(ht_covariance(u, v, audit) / vv, *clip))


class DifferenceRatioEstimator(Estimator):
    """Base class: subclasses supply the predictor ``eta``."""

    tune: bool = False
    lam: float = 1.0
    level: float = 0.95
    min_human: float = 2.0
    # interval rule chosen on development data in S4: shrunk stratum variances with
    # Satterthwaite degrees of freedom ("lin" = plain linearisation, df = m - H)
    variance_rule: str = "shrink_satt"

    def _predictor(self, ctx: EstimationContext, outcome: str):
        """Return ``(eta_all, eta_aud, reason)``; ``eta_all is None`` means infeasible."""
        raise NotImplementedError

    def estimate(self, ctx: EstimationContext, outcome: str) -> EstimateResult:
        audit = ctx.audit
        if audit is None:
            return EstimateResult.infeasible(self.name, outcome, "no audit")
        if audit.n_human < self.min_human:
            return EstimateResult.infeasible(self.name, outcome, "too few audited humans")
        eta_all, eta_aud, reason = self._predictor(ctx, outcome)
        if eta_all is None:
            return EstimateResult.infeasible(self.name, outcome, reason)
        y = ctx.obs.outcome(outcome)
        a_aud, w, idx = audit.labels, audit.w, audit.idx
        lam = self.lam
        num, den = _ratio(y, a_aud, w, eta_all, eta_aud, idx, lam)
        if den <= 0:
            return EstimateResult.infeasible(self.name, outcome, "non-positive denominator")
        mu = num / den
        if self.tune:
            lam = tune_lambda(y[idx], a_aud, eta_aud, mu, audit)
            num, den = _ratio(y, a_aud, w, eta_all, eta_aud, idx, lam)
            if den <= 0:
                return EstimateResult.infeasible(self.name, outcome, "non-positive denominator")
            mu = num / den
        z = ((1 - a_aud) - lam * (1 - eta_aud)) * (y[idx] - mu)
        vr = variance_report(z, audit)
        se_lin = float(np.sqrt(max(vr["var"], 0.0)) / den)
        se_shrink = float(np.sqrt(max(vr["var_shrink"], 0.0)) / den)
        if self.variance_rule == "lin":
            se, df = se_lin, vr["df"]
        else:
            se, df = se_shrink, vr["df_satt"]
        lo, hi = t_interval(mu, se, df, self.level)
        return EstimateResult(self.name, outcome, float(mu), se, lo, hi, True, "",
                              extras={"lam": lam, "den": float(den),
                                      "n_machine_audit": audit.n_machine, "df": vr["df"],
                                      "df_satt": vr["df_satt"], "se_lin": se_lin,
                                      "se_shrink": se_shrink})


class AuditHajek(DifferenceRatioEstimator):
    """Two-phase Hajek estimator: the audited humans alone, design-weighted."""

    name = "hajek"
    lam = 0.0

    def _predictor(self, ctx, outcome):
        n = ctx.obs.n
        return np.ones(n), np.ones(ctx.audit.m), ""


class FlagPPI(DifferenceRatioEstimator):
    """Prediction-powered estimator with the detector flag as predictor."""

    def __init__(self, tune: bool = False) -> None:
        self.tune = bool(tune)
        self.name = "flag_ppi_pp" if self.tune else "flag_ppi"

    def _predictor(self, ctx, outcome):
        d = ctx.obs.flag.astype(float)
        return d, d[ctx.audit.idx], ""


class RCPPI(DifferenceRatioEstimator):
    """RC-PPI: cross-fitted machine-origin propensity as predictor (RC-PPI++ if tuned)."""

    def __init__(self, tune: bool = False, propensity: CrossFitPropensity | None = None) -> None:
        self.tune = bool(tune)
        self.propensity = propensity or CrossFitPropensity()
        self.name = "rc_ppi_pp" if self.tune else "rc_ppi"

    def _predictor(self, ctx, outcome):
        fit = ctx.cached(self.propensity.key,
                         lambda: self.propensity.fit(ctx.obs, ctx.audit, ctx.rng))
        if not fit.feasible:
            return None, None, fit.reason
        return fit.eta_all, fit.eta_oof, ""


class GREGRatio(DifferenceRatioEstimator):
    """Generalised regression (model-assisted) estimator with a linear working model.

    Weighted least squares of ``(1 - a) y`` and of ``(1 - a)`` on the features, fitted on
    the audit with design weights, gives predictions on every record. It is the
    difference estimator with those predictions; there is no cross-fitting, as in
    classical GREG.
    """

    name = "greg"

    def __init__(self, features: tuple[str, ...] = ("score", "loglen", "flag")) -> None:
        self.features = tuple(features)

    def estimate(self, ctx: EstimationContext, outcome: str) -> EstimateResult:
        audit = ctx.audit
        if audit is None or audit.n_human < self.min_human:
            return EstimateResult.infeasible(self.name, outcome, "too few audited humans")
        names = tuple(f for f in self.features if f in ("score", "flag")
                      or f in ctx.obs.covariates)
        x = np.column_stack([np.ones(ctx.obs.n), ctx.obs.design_matrix(names)])
        y = ctx.obs.outcome(outcome)
        idx, w, a = audit.idx, audit.w, audit.labels
        xa = x[idx]
        if np.linalg.matrix_rank(xa) < x.shape[1]:
            return EstimateResult.infeasible(self.name, outcome, "rank-deficient audit design")
        wsqrt = np.sqrt(w)[:, None]
        t, d = (1 - a) * y[idx], (1 - a)
        beta_t = np.linalg.lstsq(wsqrt * xa, np.sqrt(w) * t, rcond=None)[0]
        beta_d = np.linalg.lstsq(wsqrt * xa, np.sqrt(w) * d, rcond=None)[0]
        pt, pd_ = x @ beta_t, x @ beta_d
        num = pt.sum() + np.sum(w * (t - pt[idx]))
        den = pd_.sum() + np.sum(w * (d - pd_[idx]))
        if den <= 0:
            return EstimateResult.infeasible(self.name, outcome, "non-positive denominator")
        mu = num / den
        z = (t - pt[idx]) - mu * (d - pd_[idx])
        vr = variance_report(z, audit)
        se = float(np.sqrt(max(vr["var"], 0.0)) / den)
        lo, hi = t_interval(mu, se, vr["df"], self.level)
        return EstimateResult(self.name, outcome, float(mu), se, lo, hi, True, "",
                              extras={"den": float(den)})
