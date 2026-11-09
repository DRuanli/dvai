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

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..sampling.audit import AuditSample

__all__ = ["ht_covariance", "ht_variance", "collapse_strata", "t_interval", "degrees_of_freedom",
           "variance_report", "anticipated_s2", "stratum_rate",
           "DEFAULT_TARGETS"]


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


@dataclass(frozen=True)
class _Stratum:
    """One (collapsed) stratum of the audit, for variance computations."""

    sel: np.ndarray        # mask of its audited records
    m: int                 # audited records
    contrib: float         # Hajek-approximation variance contribution
    unit: float            # contrib = unit * s2 for equal probabilities: N_h^2 (1 - f_h) / m_h
    equal: bool = True     # all records of the stratum share one inclusion probability

    @property
    def s2(self) -> float:
        """Within-stratum sample variance (equal-probability case)."""
        return self.contrib / self.unit if self.unit > 0 else 0.0


def _strata(a: np.ndarray, b: np.ndarray, audit: AuditSample) -> list[_Stratum]:
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
        unit = float(np.sum(c * w[sel] ** 2))  # = N_h^2 (1 - f_h) / m_h
        rho_h = audit.rho[sel]
        out.append(_Stratum(sel, m_h, float(contrib), unit,
                            bool(np.ptp(rho_h) <= 1e-12 * max(rho_h.max(), 1e-300))))
    return out


def ht_covariance(a: np.ndarray, b: np.ndarray, audit: AuditSample) -> float:
    """Estimated design covariance of the HT totals ``sum w a`` and ``sum w b``."""
    return float(sum(s.contrib for s in _strata(a, b, audit)))


def _satterthwaite(v: np.ndarray, m: np.ndarray) -> float:
    denom = float(np.sum(v ** 2 / (m - 1)))
    return max(float(v.sum() ** 2 / denom), 1.0) if denom > 0 else float(np.sum(m - 1))


def stratum_rate(labels: np.ndarray, sel: np.ndarray, prior: str, alpha: float,
                 eps_hat: float) -> float:
    """Smoothed machine share of one stratum's audited records.

    ``"jeffreys"``: ``(k + 1/2) / (m + 1)``. ``"eb"``: ``(k + alpha * eps_hat) / (m + alpha)``,
    a beta-binomial (empirical-Bayes) estimate shrunk towards the design-weighted machine
    share of the whole audit with ``alpha`` pseudo-records.
    """
    k, m = float(labels[sel].sum()), float(sel.sum())
    if prior == "jeffreys":
        return (k + 0.5) / (m + 1.0)
    if prior == "eb":
        return (k + alpha * eps_hat) / (m + alpha)
    raise ValueError(f"unknown prior {prior!r}")


def anticipated_s2(z: np.ndarray, dz: np.ndarray, labels: np.ndarray, sel: np.ndarray,
                   p: float) -> float:
    """Within-stratum variance of ``z`` anticipated under a Bernoulli origin model.

    ``z`` is linear in the verified label, ``z = z0 - a * dz`` (``dz = y - mu`` for every
    estimator of the difference-ratio family and for GREG). Treating the labels of the
    stratum as Bernoulli with a smoothed rate ``p`` (:func:`stratum_rate`) gives
    ``E[Var] = mean(dz^2) p (1 - p) + var(z0 - p dz)``. Unlike the sample variance it is
    not zero when the audited records of a stratum happen to be all human.
    """
    zs, ds, a = z[sel], dz[sel], labels[sel]
    z0 = zs + a * ds
    return float(np.mean(ds ** 2) * p * (1 - p) + np.var(z0 - p * ds, ddof=1))


# chosen on the development corpora in S4 (results/S4_variance_dev_summary.csv)
DEFAULT_TARGETS = {"model": {"prior": "jeffreys"}}


def variance_report(z: np.ndarray, audit: AuditSample, kappa: float = 1.0,
                    dz: np.ndarray | None = None, targets: dict | None = None) -> dict:
    """Variance of the HT total of ``z`` with small-sample refinements (stage S4).

    Keys
    ----
    ``var``, ``df``, ``df_satt``
        Linearised variance; ``m - H`` and Satterthwaite degrees of freedom.
    ``var_pool``, ``df_pool``
        Each stratum's sample variance shrunk towards the pooled within-stratum variance
        with ``kappa`` pseudo-records.
    ``var_<name>``, ``df_<name>`` for every entry of ``targets`` (when ``dz`` is given)
        Shrunk with ``kappa`` pseudo-records towards the variance anticipated under a
        Bernoulli origin model (:func:`anticipated_s2`) whose stratum rates are smoothed
        as the entry says (:func:`stratum_rate`). Default: ``DEFAULT_TARGETS``.

    The ``df_*`` entries are Satterthwaite degrees of freedom of the respective sums.
    """
    z = np.asarray(z, dtype=float)
    targets = DEFAULT_TARGETS if targets is None else targets
    st = _strata(z, z, audit)
    base_df = degrees_of_freedom(audit)
    if not st:
        zero = {"var": 0.0, "df": base_df, "df_satt": 1.0, "var_pool": 0.0, "df_pool": 1.0}
        if dz is not None:
            for name in targets:
                zero |= {f"var_{name}": 0.0, f"df_{name}": 1.0}
        return zero
    m = np.array([s.m for s in st], dtype=float)
    unit = np.array([s.unit for s in st])
    s2 = np.array([s.s2 for s in st])
    contrib = np.array([s.contrib for s in st])
    out = {"var": float(contrib.sum()), "df": base_df, "df_satt": _satterthwaite(contrib, m)}

    # shrinkage only where s2 is a sample variance of z: strata with one inclusion
    # probability (collapsed strata with mixed probabilities and PPS keep their s2)
    equal = np.array([s.equal for s in st])

    def shrunk(target: np.ndarray) -> np.ndarray:
        return np.where(equal, ((m - 1) * s2 + kappa * target) / (m - 1 + kappa), s2)

    pooled = float(np.sum((m - 1) * s2) / np.sum(m - 1))
    v = unit * shrunk(np.full_like(s2, pooled))
    out |= {"var_pool": float(v.sum()), "df_pool": _satterthwaite(v, m)}
    if dz is not None:
        dz = np.asarray(dz, dtype=float)
        w, a = audit.w, audit.labels
        eps_hat = float(np.sum(w * a) / np.sum(w))
        for name, spec in targets.items():
            t = np.array([anticipated_s2(z, dz, a, s.sel,
                                         stratum_rate(a, s.sel, spec["prior"],
                                                      spec.get("alpha", 1.0), eps_hat))
                          for s in st])
            v = unit * shrunk(t)
            out |= {f"var_{name}": float(v.sum()), f"df_{name}": _satterthwaite(v, m)}
    return out


def ht_variance(a: np.ndarray, audit: AuditSample) -> float:
    return ht_covariance(a, a, audit)


def degrees_of_freedom(audit: AuditSample) -> int:
    """Audited records minus strata, at least 1, counting only strata that contribute
    variance (records with inclusion probability 1, such as the pilot of a two-stage
    design, carry none)."""
    strata = collapse_strata(audit.strata)
    df = 0
    for h in np.unique(strata):
        sel = strata == h
        if np.any(audit.rho[sel] < 1.0):
            df += int(sel.sum()) - 1
    return max(df, 1)


def t_interval(point: float, se: float, df: float, level: float = 0.95) -> tuple[float, float]:
    if not np.isfinite(se):
        return float("nan"), float("nan")
    q = stats.t.ppf(0.5 + level / 2, df)
    return point - q * se, point + q * se
