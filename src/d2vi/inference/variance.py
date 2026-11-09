"""Design-based variance for audit estimators (Theorem 2).

Every audit estimator in :mod:`d2vi.estimators` is a ratio of two Horvitz-Thompson-type
totals. After linearisation its error is the HT error of a single residual total,

    mu_hat - mu_H  ~  (sum_{i in audit} w_i z_i - sum_{i in U} z_i) / D_hat,

so its variance is the design variance of an HT total of ``z`` divided by ``D_hat**2``.

:func:`ht_covariance` estimates that design (co)variance with one formula that covers
stratified simple random sampling without replacement and fixed-size unequal-probability
sampling. Within each stratum ``h`` it uses the Hajek approximation

    V_h = m_h / (m_h - 1) * sum_{i in h} (1 - rho_i) (w_i a_i - A_h)(w_i b_i - B_h),

with ``A_h`` the ``(1 - rho)``-weighted mean of ``w_i a_i``. For equal probabilities
within a stratum this reduces exactly to ``N_h^2 (1 - m_h/N_h) s_ab,h / m_h``, the textbook
stratified formula with finite-population correction. Strata with a single audited record
are collapsed into a neighbouring stratum (the collapsed-strata technique), which is
conservative.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..sampling.audit import AuditSample

__all__ = ["ht_covariance", "ht_variance", "collapse_strata", "t_interval", "degrees_of_freedom",
           "variance_report"]


def collapse_strata(strata: np.ndarray) -> np.ndarray:
    """Merge every stratum holding fewer than two audited records into a neighbour."""
    strata = np.asarray(strata).copy()
    while True:
        keys, counts = np.unique(strata, return_counts=True)
        if keys.shape[0] <= 1 or counts.min() >= 2:
            return strata
        j = int(np.argmin(counts))
        neighbour = keys[j - 1] if j > 0 else keys[j + 1]
        strata[strata == keys[j]] = neighbour


def _components(a: np.ndarray, b: np.ndarray, audit: AuditSample) -> list[tuple[float, int, float]]:
    """Per-stratum ``(contribution, m_h, scale_h)``; ``contribution / scale_h`` is the
    within-stratum sample covariance of ``a`` and ``b`` for equal-probability strata."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = 1.0 / audit.rho
    one_minus = 1.0 - audit.rho
    strata = collapse_strata(audit.strata)
    out = []
    for h in np.unique(strata):
        sel = strata == h
        m_h = int(sel.sum())
        c = one_minus[sel]
        if m_h < 2 or c.sum() <= 0:  # single record, or stratum fully enumerated
            continue
        wa, wb = w[sel] * a[sel], w[sel] * b[sel]
        abar = np.sum(c * wa) / c.sum()
        bbar = np.sum(c * wb) / c.sum()
        contrib = m_h / (m_h - 1) * np.sum(c * (wa - abar) * (wb - bbar))
        scale = float(np.sum(c * w[sel] ** 2) / (m_h - 1))  # = N_h^2 (1 - f_h) / m_h
        out.append((float(contrib), m_h, scale))
    return out


def ht_covariance(a: np.ndarray, b: np.ndarray, audit: AuditSample) -> float:
    """Estimated design covariance of the HT totals ``sum w a`` and ``sum w b``."""
    return float(sum(c for c, _, _ in _components(a, b, audit)))


def variance_report(z: np.ndarray, audit: AuditSample, kappa: float = 1.0) -> dict:
    """Variance of the HT total of ``z`` with two small-sample refinements (stage S4).

    * ``df_satt`` - Satterthwaite degrees of freedom for a sum of stratum variances; small
      when one stratum with few records dominates the variance.
    * ``var_shrink`` - each stratum's residual variance is shrunk towards the pooled
      within-stratum variance with weight ``kappa`` pseudo-records, which stops a stratum
      whose sampled residuals happen to be all zero (sparse misclassification) from
      contributing no variance.
    """
    comps = _components(z, z, audit)
    var = sum(c for c, _, _ in comps)
    if not comps:
        return {"var": 0.0, "df": degrees_of_freedom(audit), "df_satt": 1.0, "var_shrink": 0.0}
    s2 = np.array([c / sc if sc > 0 else 0.0 for c, _, sc in comps])
    m = np.array([mh for _, mh, _ in comps], dtype=float)
    pooled = float(np.sum((m - 1) * s2) / np.sum(m - 1))
    s2_shrunk = ((m - 1) * s2 + kappa * pooled) / (m - 1 + kappa)
    var_shrink = float(sum(sc * s for (_, _, sc), s in zip(comps, s2_shrunk, strict=True)))
    contrib = np.array([c for c, _, _ in comps])
    denom = float(np.sum(contrib ** 2 / (m - 1)))
    df_satt = float(var ** 2 / denom) if denom > 0 else float(np.sum(m - 1))
    return {"var": float(var), "df": degrees_of_freedom(audit), "df_satt": max(df_satt, 1.0),
            "var_shrink": var_shrink}


def ht_variance(a: np.ndarray, audit: AuditSample) -> float:
    return ht_covariance(a, a, audit)


def degrees_of_freedom(audit: AuditSample) -> int:
    """Audited records minus strata, at least 1."""
    strata = collapse_strata(audit.strata)
    return max(int(audit.m - np.unique(strata).shape[0]), 1)


def t_interval(point: float, se: float, df: float, level: float = 0.95) -> tuple[float, float]:
    if not np.isfinite(se):
        return float("nan"), float("nan")
    q = stats.t.ppf(0.5 + level / 2, df)
    return point - q * se, point + q * se
