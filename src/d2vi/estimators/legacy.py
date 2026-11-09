"""Faithful re-implementation of the original (v1) experiment code, for reproduction only.

This module reproduces the estimators and audits of the four Colab notebooks shipped
with the first submission, including their known problems, so that Stage S3 (experiment
E0) can measure how far the corrected pipeline moves the published numbers:

* F2 - when stratified RC-PPI abstains, the v1 code reported the random-audit flag-PPI
  value in its place (``rc_strat = rs if not isnan(rs) else p``);
* F3 - flag-PPI used a Bernoulli audit of random size while stratified RC-PPI used a
  separate fixed-size audit (``max(round(frac * n), 20)``) from another random stream;
* F4 - unaudited records were predicted by a model fitted on the whole audit while the
  rectifier used fold models;
* F13 - with a random audit, ``ppi_py.ppi_mean_pointestimate`` was called with its
  default ``lam=None``, i.e. power-tuned PPI++, whereas the stratified branch used
  ``lam = 1``; the paper describes both as standard PPI.

Do not use these functions for new results. The random-number scheme mirrors the
notebooks (``default_rng(1000 * seed + int(eps * 100))`` for the mixture and the random
audit, ``default_rng(7000 + seed)`` for the stratified audit, ``default_rng(seed)`` for
the cross-fitting folds).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "legacy_stratified_audit",
    "legacy_flag_ppi",
    "legacy_rc_ppi",
    "legacy_replication",
]


def _ppi_mean(y_lab, yhat_lab, yhat_unlab, lam=None):
    """``ppi_py.ppi_mean_pointestimate`` (power-tuned when ``lam is None``)."""
    try:
        from ppi_py import ppi_mean_pointestimate
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("legacy reproduction needs `pip install ppi-python`") from exc
    return float(np.asarray(ppi_mean_pointestimate(np.asarray(y_lab, float),
                                                   np.asarray(yhat_lab, float),
                                                   np.asarray(yhat_unlab, float),
                                                   lam=lam)).ravel()[0])


def _wppi(vh_all, v_aud, vh_aud, w):
    return float(np.mean(vh_all) + np.sum(w * (v_aud - vh_aud)) / len(vh_all))


def legacy_stratified_audit(score: np.ndarray, frac: float, rng: np.random.Generator):
    """v1 stratified audit: half from the top score decile, half from the rest."""
    n = score.shape[0]
    m = max(int(round(frac * n)), 20)
    thr = np.quantile(score, 0.90)
    top = np.flatnonzero(score >= thr)
    rest = np.setdiff1d(np.arange(n), top)
    m_top = min(m // 2, top.shape[0])
    m_rest = m - m_top
    idx = np.concatenate([rng.choice(top, m_top, replace=False),
                          rng.choice(rest, m_rest, replace=False)])
    w = np.zeros(n)
    w[np.intersect1d(idx, top)] = top.shape[0] / m_top
    w[np.intersect1d(idx, rest)] = rest.shape[0] / m_rest
    return idx, w


def legacy_flag_ppi(y, d, a, aidx) -> float:
    """v1 flag-PPI on a random audit (power-tuned through ppi_py)."""
    mask = np.zeros(y.shape[0], dtype=bool)
    mask[aidx] = True
    yl, dl, al = y[mask], d[mask], a[mask]
    num = _ppi_mean(yl * (1 - al), yl * (1 - dl), y[~mask] * (1 - d[~mask]))
    den = _ppi_mean(1 - al, 1 - dl, 1 - d[~mask])
    return num / den if den > 1e-6 else float("nan")


def _fit_pi(x, aidx, ya, wa, seed):
    from sklearn.linear_model import LogisticRegression

    pi = np.zeros(x.shape[0])
    fold = np.random.default_rng(seed).random(aidx.shape[0]) < 0.5
    for tr in (fold, ~fold):
        if ya[tr].sum() < 3 or (1 - ya[tr]).sum() < 3:
            return None
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(x[aidx[tr]], ya[tr], sample_weight=wa[tr])
        pi[aidx[~tr]] = model.predict_proba(x[aidx[~tr]])[:, 1]
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(x[aidx], ya, sample_weight=wa)
    rest = np.setdiff1d(np.arange(x.shape[0]), aidx)
    pi[rest] = model.predict_proba(x[rest])[:, 1]
    return pi


def legacy_rc_ppi(y, a, score, length, aidx, seed, w_full=None) -> float:
    """v1 RC-PPI. ``w_full`` is the length-``N`` HT weight vector of a stratified audit."""
    aidx = np.asarray(aidx)
    ya = a[aidx]
    if ya.sum() < 6 or (1 - ya).sum() < 6:
        return float("nan")
    x = np.column_stack([score, np.log1p(length)])
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    wa = np.ones(aidx.shape[0]) if w_full is None else w_full[aidx]
    pi = _fit_pi(x, aidx, ya, wa, seed)
    if pi is None:
        return float("nan")
    soft = 1 - pi
    if w_full is None:
        mask = np.zeros(y.shape[0], dtype=bool)
        mask[aidx] = True
        num = _ppi_mean((y * (1 - a))[mask], (y * soft)[mask], (y * soft)[~mask])
        den = _ppi_mean((1 - a)[mask], soft[mask], soft[~mask])
    else:
        num = _wppi(y * soft, y[aidx] * (1 - ya), (y * soft)[aidx], wa)
        den = _wppi(soft, 1 - ya, soft[aidx], wa)
    return num / den if den > 1e-6 else float("nan")


def legacy_replication(human: dict, machine: dict, n: int, eps: float, seed: int,
                       frac: float, outcomes: list[str]) -> list[dict]:
    """One replication of the v1 main grid for one (corpus, generator, eps, seed).

    ``human`` and ``machine`` map column names (``score``, ``flag``, ``length`` and
    ``y_<outcome>``) to arrays in pool order. Returns one row per outcome with the v1
    estimates, the stratified RC-PPI value with and without the F2 fallback, and the
    finite-population truth.
    """
    rng = np.random.default_rng(1000 * seed + int(eps * 100))
    n_a = int(round(n * eps))
    n_h = n - n_a
    ih = rng.choice(human["score"].shape[0], n_h, replace=False)
    ia = rng.choice(machine["score"].shape[0], n_a, replace=False)

    def cat(col):
        return np.concatenate([human[col][ih], machine[col][ia]])

    a = np.concatenate([np.zeros(n_h), np.ones(n_a)])
    score, d, length = cat("score"), cat("flag").astype(float), cat("length")
    aidx = np.flatnonzero(rng.random(n) < frac)
    rows = []
    for o in outcomes:
        y = cat("y_" + o)
        sidx, w = legacy_stratified_audit(score, frac, np.random.default_rng(7000 + seed))
        rs = legacy_rc_ppi(y, a, score, length, sidx, seed, w_full=w)
        p = legacy_flag_ppi(y, d, a, aidx)
        rows.append({
            "outcome": o, "eps": eps, "seed": seed,
            "truth": float(y[a == 0].mean()),
            "naive": float(y.mean()),
            "deletion": float(y[d == 0].mean()),
            "ppi": p,
            "rc_ppi": legacy_rc_ppi(y, a, score, length, aidx, seed),
            "rc_strat": rs if not np.isnan(rs) else p,  # F2: silent fallback
            "rc_strat_raw": rs,
            "m_random": int(aidx.shape[0]),
            "m_strat": int(sidx.shape[0]),
        })
    return rows
