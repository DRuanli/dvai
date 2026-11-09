"""Cell-level summaries: bias, RMSE, standardised RMSE, coverage, feasibility, regret.

Definitions used in the paper (Section 4 and R1-M5):

* error ``e = point - truth`` per replication, over feasible replications only;
* ``rmse = sqrt(mean(e^2))`` and ``srmse = rmse / sd_h`` with ``sd_h`` the outcome's
  standard deviation among the pool's human documents;
* ``coverage`` = share of feasible replications whose interval contains the truth, with
  Monte Carlo standard error ``sqrt(c (1 - c) / R)``;
* ``regret`` of method ``k`` in cell ``c`` = ``srmse_c(k) - min_j srmse_c(j)`` over a
  stated candidate set that excludes oracles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["CELL_KEYS", "summarize", "regret_table"]

CELL_KEYS = ["corpus", "detector", "eps", "n", "budget", "design", "mixture", "outcome"]


def _summ(g: pd.DataFrame) -> pd.Series:
    feas = g[g["feasible"].astype(bool)]
    e = (feas["point"] - feas["truth"]).to_numpy(dtype=float)
    r = e.shape[0]
    out = {"n_rep": len(g), "feasible_rate": r / len(g) if len(g) else np.nan}
    if r == 0:
        return pd.Series(out)
    rmse = float(np.sqrt(np.mean(e ** 2)))
    sd_h = float(feas["sd_h"].iloc[0])
    out.update(
        bias=float(e.mean()),
        rmse=rmse,
        srmse=rmse / sd_h if sd_h > 0 else np.nan,
        mc_se_rmse=float(np.std(e ** 2, ddof=1) / (2 * rmse * np.sqrt(r))) if r > 1 and rmse > 0
        else np.nan,
    )
    if {"ci_low", "ci_high"} <= set(feas.columns):
        ok = np.isfinite(feas["ci_low"]) & np.isfinite(feas["ci_high"])
        if ok.any():
            f2 = feas[ok]
            cov = float(np.mean((f2["ci_low"] <= f2["truth"]) & (f2["truth"] <= f2["ci_high"])))
            out.update(coverage=cov, coverage_mc_se=float(np.sqrt(cov * (1 - cov) / ok.sum())),
                       width=float(np.mean(f2["ci_high"] - f2["ci_low"])))
    return pd.Series(out)


def summarize(df: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    keys = [k for k in (keys or CELL_KEYS) if k in df.columns] + ["estimator"]
    est = df[~df["estimator"].astype(str).str.startswith("cert_")]
    rows = []
    for key, g in est.groupby(keys, dropna=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        row.update(_summ(g).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def regret_table(summary: pd.DataFrame, candidates: list[str], metric: str = "srmse",
                 keys: list[str] | None = None) -> pd.DataFrame:
    """Per-method regret: max, mean and 90th percentile over cells."""
    keys = [k for k in (keys or CELL_KEYS) if k in summary.columns]
    s = summary[summary["estimator"].isin(candidates)]
    wide = s.pivot_table(index=keys, columns="estimator", values=metric)
    best = wide.min(axis=1)
    reg = wide.sub(best, axis=0)
    ratio = wide.div(best, axis=0)
    return pd.DataFrame({
        "max_regret": reg.max(),
        "mean_regret": reg.mean(),
        "p90_regret": reg.quantile(0.9),
        "max_ratio": ratio.max(),
        "n_cells": reg.count(),
    }).sort_values("max_regret")
