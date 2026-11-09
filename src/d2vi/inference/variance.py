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

__all__ = ["ht_covariance", "ht_variance", "collapse_strata", "t_interval", "degrees_of_freedom"]


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


def ht_covariance(a: np.ndarray, b: np.ndarray, audit: AuditSample) -> float:
    """Estimated design covariance of the HT totals ``sum w a`` and ``sum w b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = 1.0 / audit.rho
    one_minus = 1.0 - audit.rho
    strata = collapse_strata(audit.strata)
    total = 0.0
    for h in np.unique(strata):
        sel = strata == h
        m_h = int(sel.sum())
        if m_h < 2:
            continue
        c = one_minus[sel]
        wa, wb = w[sel] * a[sel], w[sel] * b[sel]
        if c.sum() <= 0:  # stratum fully enumerated: no sampling variance
            continue
        abar = np.sum(c * wa) / c.sum()
        bbar = np.sum(c * wb) / c.sum()
        total += m_h / (m_h - 1) * np.sum(c * (wa - abar) * (wb - bbar))
    return float(total)


def ht_variance(a: np.ndarray, audit: AuditSample) -> float:
    return ht_covariance(a, a, audit)


def degrees_of_freedom(audit: AuditSample) -> int:
    """Audited records minus strata, at least 1."""
    strata = collapse_strata(audit.strata)
    return max(int(audit.m - np.unique(strata).shape[0]), 1)


def t_interval(point: float, se: float, df: int, level: float = 0.95) -> tuple[float, float]:
    if not np.isfinite(se):
        return float("nan"), float("nan")
    q = stats.t.ppf(0.5 + level / 2, df)
    return point - q * se, point + q * se
