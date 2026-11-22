"""Descriptive summaries of the held-out results that are not endpoints.

These analyses were defined on 22-23 November 2026, *after* the held-out results had been
seen (the endpoints PE1-PE4 and the secondary grids were frozen on 9 November). They add no
simulation: they re-read the per-cell tables of the primary grid and the feature tables.
The paper labels them as post hoc descriptions.

* :func:`best_single_estimator` - which single estimator has the lowest MSE in a cell;
* :func:`excess_summary` - distribution over cells of a rule's excess standardised RMSE
  (the regret of Definition 2 in ``paper/theory.tex``);
* :func:`rule_summary` - standardised RMSE of each rule under the two ways of pooling cells;
* :func:`lemma1_terms` - the two terms of the deletion bias (Lemma 1): residual
  contamination and the selection of humans.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["SINGLES", "RULES", "best_single_estimator", "excess_summary", "rule_summary",
           "lemma1_terms"]

SINGLES = ("naive", "deletion", "hajek", "flag_ppi", "rc_ppi_pp")
RULES = ("policy_debiased", *SINGLES)
TIE = "flag_ppi=rc_ppi_pp"


def _srmse(cells: pd.DataFrame, rules=RULES) -> pd.DataFrame:
    return np.sqrt(cells[[f"mse_{r}" for r in rules]]).set_axis(list(rules), axis=1)


def best_single_estimator(cells: pd.DataFrame, singles=SINGLES) -> pd.Series:
    """Name of the single estimator with the lowest MSE in every cell.

    ``rc_ppi_pp`` is scored intention-to-treat (flag-PPI on the same audit when RC-PPI++ is
    infeasible), so in a cell where RC-PPI++ is never feasible the two have the same MSE.
    When that common value is the minimum the cell is labelled ``flag_ppi=rc_ppi_pp``
    instead of being credited to either.
    """
    mse = cells[[f"mse_{r}" for r in singles]].to_numpy()
    lowest = mse.min(axis=1)
    best = np.array(singles, dtype=object)[mse.argmin(axis=1)]
    if {"flag_ppi", "rc_ppi_pp"} <= set(singles):
        i, j = singles.index("flag_ppi"), singles.index("rc_ppi_pp")
        tie = (mse[:, i] == lowest) & (mse[:, j] == lowest)
        best[tie] = TIE
    return pd.Series(best, index=cells.index, name="best")


def excess_summary(cells: pd.DataFrame, rules=RULES, singles=SINGLES,
                   thresholds=(0.05, 0.10, 0.20)) -> dict:
    """Per rule: quantiles of the excess sRMSE over cells, the share of cells above each
    threshold (human-SD units) and the cell with the largest excess."""
    sr = _srmse(cells, rules)
    excess = sr.sub(sr[list(singles)].min(axis=1), axis=0)
    out = {}
    for r in rules:
        e = excess[r]
        worst = cells.loc[e.idxmax()]
        out[r] = {
            "mean": float(e.mean()),
            "quantiles": {f"{q:g}": float(e.quantile(q)) for q in (0.5, 0.9, 0.95, 0.99)},
            "max": float(e.max()),
            "share_above": {f"{t:g}": float((e > t).mean()) for t in thresholds},
            "share_negative": float((e < 0).mean()),
            "worst_cell": {k: (worst[k].item() if hasattr(worst[k], "item") else worst[k])
                           for k in ("pool", "budget", "eps", "outcome") if k in worst.index},
        }
    return out


def rule_summary(cells: pd.DataFrame, rules=RULES) -> dict:
    """Standardised RMSE of each rule, pooled in two ways.

    ``pooled``: square root of the mean cell MSE (the convention of PE4's ``pooled_srmse``;
    dominated by the cells where a rule fails). ``mean_cell``: mean over cells of the cell
    sRMSE (the convention of the secondary summaries; describes a typical cell).
    """
    sr = _srmse(cells, rules)
    return {r: {"pooled": float(np.sqrt(cells[f"mse_{r}"].mean())),
                "mean_cell": float(sr[r].mean())} for r in rules}


def lemma1_terms(y: np.ndarray, a: np.ndarray, d: np.ndarray, eps: float | None = None) -> dict:
    """Terms of the deletion bias of Lemma 1 for binary origins ``a`` and flags ``d``.

    With ``eps=None`` the contamination rate is that of the records given, and
    ``selection + residual`` equals ``mean(y | d = 0) - mean(y | a = 0)`` exactly. With a
    value of ``eps`` the human and machine records are treated as two pools mixed at that
    rate: the selection term is unchanged (it involves humans only) and the residual term
    uses ``eps_res(eps)``.

    Returns the raw (unstandardised) terms; ``naive`` is ``eps (mu_A - mu_H)``.
    """
    y, a, d = (np.asarray(v, dtype=float) for v in (y, a, d))
    if not np.all((a == 0) | (a == 1)):
        raise ValueError("lemma1_terms is defined for binary origins")
    hum, mac = a == 0, a == 1
    if not hum.any():
        raise ValueError("no human record")
    eps = float(mac.mean()) if eps is None else float(eps)
    f = float(d[hum].mean())
    s = float(d[mac].mean()) if mac.any() else float("nan")
    mu_h = float(y[hum].mean())
    kept_h, kept_a = hum & (d == 0), mac & (d == 0)
    if not kept_h.any():
        raise ValueError("every human record is flagged (f = 1): Lemma 1 needs f < 1")
    mu_h_kept = float(y[kept_h].mean())
    selection = mu_h_kept - mu_h                     # = -C_0 / (1 - f)
    if mac.any() and kept_a.any() and eps > 0:
        eps_res = eps * (1 - s) / ((1 - eps) * (1 - f) + eps * (1 - s))
        delta_surv = float(y[kept_a].mean()) - mu_h_kept
    else:
        eps_res, delta_surv = 0.0, 0.0
    naive = eps * (float(y[mac].mean()) - mu_h) if mac.any() else 0.0
    return {"f": f, "s": s, "eps": eps, "eps_res": float(eps_res), "delta_surv": delta_surv,
            "selection": selection, "residual": float(eps_res * delta_surv),
            "deletion_bias": selection + float(eps_res * delta_surv), "naive_bias": naive}
