"""Primary endpoints PE1-PE4 of the frozen protocol (docs/EXPERIMENT_PROTOCOL.md, section 6).

Written and tested on development data before any held-out corpus was run. The analysis
has two steps, so that raw replication files can be reduced one pool at a time:

1. :func:`cell_stats` and :func:`cert_tables` reduce the raw replications of one pool to
   one row per cell (and, for certification, one row per replication cluster);
2. :func:`evaluate` computes the endpoints from the concatenated reductions.

Definitions (protocol sections 3, 5, 6):

* standardised error = (estimate - human mean of the response set) / human SD of the pool;
* intention-to-treat (ITT) RC-PPI++ = RC-PPI++ where feasible, otherwise flag-PPI on the
  same audit, point estimate and interval together;
* per protocol = replications in which RC-PPI++ is feasible, both estimators on those same
  replications, cells with at least ``min_feasible`` such replications;
* regret of a rule in a cell = its standardised RMSE minus the lowest standardised RMSE
  among the single estimators (RC-PPI++ scored ITT) in that cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["EndpointSpec", "cell_stats", "cert_tables", "evaluate", "bca_interval"]


@dataclass(frozen=True)
class EndpointSpec:
    cell: tuple[str, ...] = ("corpus", "pool", "budget", "eps", "outcome")
    unit: tuple[str, ...] = ("corpus",)          # cluster for PE1 and PE4 (subset of cell)
    rc: str = "rc_ppi_pp"
    fallback: str = "flag_ppi"
    policy: str = "policy_debiased"
    certifier: str = "cert_tost0.2"
    singles: tuple[str, ...] = ("naive", "deletion", "hajek", "flag_ppi", "rc_ppi_pp")
    margin: float = 1.10
    alpha: float = 0.05
    min_feasible: int = 50
    coverage_mean_min: float = 0.93
    coverage_cell_min: float = 0.90
    coverage_share_min: float = 0.90
    false_cert_max: float = 0.075
    near_boundary: tuple[float, float] = (1.0, 1.1)

    def __post_init__(self) -> None:
        if not set(self.unit) <= set(self.cell) or "outcome" not in self.cell:
            raise ValueError("unit columns must be cell columns, and cells need 'outcome'")


DEFAULT = EndpointSpec()


# --------------------------------------------------------------------------- reductions
def cell_stats(raw: pd.DataFrame, spec: EndpointSpec = DEFAULT) -> pd.DataFrame:
    """One row per cell: standardised MSE of every rule, ITT coverage and width,
    feasibility, and the per-protocol MSEs of RC-PPI++ and the fallback."""
    cell = list(spec.cell)
    key = cell + ["rep"]
    est = raw[~raw["estimator"].str.startswith("cert_")]
    rules = [r for r in dict.fromkeys((*spec.singles, spec.policy)) if r in set(est["estimator"])]
    missing = {spec.rc, spec.fallback} - set(rules)
    if missing:
        raise ValueError(f"raw results lack estimators {sorted(missing)}")
    pt = est[est["estimator"].isin(rules)].pivot(index=key, columns="estimator", values="point")
    rc = est[est["estimator"] == spec.rc].set_index(key).reindex(pt.index)
    fb = est[est["estimator"] == spec.fallback].set_index(key).reindex(pt.index)
    ok = rc["feasible"].astype(bool)
    err = pt.sub(rc["truth"], axis=0).div(rc["sd_h"], axis=0)
    itt = err[spec.rc].where(ok, err[spec.fallback])
    lo = rc["ci_low"].where(ok, fb["ci_low"])
    hi = rc["ci_high"].where(ok, fb["ci_high"])
    d = pd.DataFrame({f"mse_{r}": err[r] ** 2 for r in rules if r != spec.rc})
    d["mse_" + spec.rc] = itt ** 2                       # RC-PPI++ is always scored ITT
    d["err_itt"] = itt
    d["cover_itt"] = ((lo <= rc["truth"]) & (rc["truth"] <= hi)).astype(float)
    d["width_itt"] = (hi - lo) / rc["sd_h"]
    d["feasible"] = ok.astype(float)
    d["pp_rc"] = (err[spec.rc] ** 2).where(ok)
    d["pp_fb"] = (err[spec.fallback] ** 2).where(ok)
    g = d.groupby(level=cell)
    out = g[[c for c in d if c.startswith("mse_")]].mean()
    out["n_rep"] = g.size()
    out["sd_itt"] = g["err_itt"].std()
    out["bias_itt"] = g["err_itt"].mean()
    out[["cover_itt", "width_itt", "feasible"]] = g[["cover_itt", "width_itt", "feasible"]].mean()
    out["n_feasible"] = g["feasible"].sum().astype(int)
    out["mse_pp_rc"] = g["pp_rc"].mean()
    out["mse_pp_fb"] = g["pp_fb"].mean()
    return out.reset_index()


def cert_tables(raw: pd.DataFrame, spec: EndpointSpec = DEFAULT
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Certification counts: per replication cluster (the outcomes of one replication share
    an audit) and per cell. ``bad``: |b_del| >= tau; ``near``: also < 1.1 tau; ``small``:
    |b_del| < tau / 2."""
    c = raw[raw["estimator"] == spec.certifier]
    lo, hi = spec.near_boundary
    b, tau = c["deletion_bias"].abs(), c["tau"]
    cert = c["certified"].astype(bool)
    f = pd.DataFrame({col: c[col] for col in (*spec.cell, "rep")})
    f["n_bad"] = (b >= lo * tau).astype(int)
    f["k_bad"] = ((b >= lo * tau) & cert).astype(int)
    f["n_near"] = ((b >= lo * tau) & (b < hi * tau)).astype(int)
    f["k_near"] = ((b >= lo * tau) & (b < hi * tau) & cert).astype(int)
    f["n_small"] = (b < tau / 2).astype(int)
    f["k_small"] = ((b < tau / 2) & cert).astype(int)
    counts = ["n_bad", "k_bad", "n_near", "k_near", "n_small", "k_small"]
    cluster_key = [k for k in spec.cell if k != "outcome"] + ["rep"]
    clusters = f.groupby(cluster_key)[counts].sum().reset_index(drop=True)
    cells = f.groupby(list(spec.cell))[counts].sum().reset_index()
    return clusters, cells


# ---------------------------------------------------------------------------- inference
def bca_interval(stat, n: int, level: float = 0.90, n_boot: int = 4000,
                 seed: int = 1) -> tuple[float, float]:
    """BCa bootstrap interval of ``stat(index array)`` over ``n`` resampled clusters."""
    if n < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    theta = stat(np.arange(n))
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    boots = boots[np.isfinite(boots)]
    p0 = np.clip(np.mean(boots < theta), 1 / (len(boots) + 1), 1 - 1 / (len(boots) + 1))
    z0 = stats.norm.ppf(p0)
    jack = np.array([stat(np.delete(np.arange(n), i)) for i in range(n)])
    dj = jack.mean() - jack
    denom = 6.0 * np.sum(dj ** 2) ** 1.5
    acc = float(np.sum(dj ** 3) / denom) if denom > 0 else 0.0
    out = []
    for q in ((1 - level) / 2, 1 - (1 - level) / 2):
        z = stats.norm.ppf(q)
        out.append(float(np.quantile(boots, stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))))))
    return out[0], out[1]


def _pattern_bootstrap_upper(n: np.ndarray, k: np.ndarray, n_boot: int, seed: int) -> float:
    """One-sided 95% upper bound of sum(k) / sum(n) by a cluster bootstrap. Resampling the
    clusters with replacement is a multinomial draw over the distinct (n, k) patterns."""
    keep = n > 0
    if not keep.any():
        return float("nan")
    pat, cnt = np.unique(np.c_[n[keep], k[keep]], axis=0, return_counts=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(int(cnt.sum()), cnt / cnt.sum(), size=n_boot)
    rates = (draws @ pat[:, 1]) / (draws @ pat[:, 0])
    return float(np.quantile(rates, 0.95))


def _unit_id(df: pd.DataFrame, spec: EndpointSpec) -> pd.Series:
    return df[list(spec.unit)].astype(str).agg("|".join, axis=1)


def _pe1(cells: pd.DataFrame, spec: EndpointSpec, n_boot: int, seed: int) -> dict:
    rc, fb = "mse_" + spec.rc, "mse_" + spec.fallback
    out = {}
    views = {"itt": (cells, rc, fb),
             "per_protocol": (cells[cells["n_feasible"] >= spec.min_feasible],
                              "mse_pp_rc", "mse_pp_fb")}
    for name, (d, a, b) in views.items():
        if d.empty:
            out[name] = {"units": 0, "passes": False, "note": "no cells"}
            continue
        per = d.groupby("_unit")[[a, b]].sum()
        ratio = per[a] / per[b]
        num, den = per[a].to_numpy(), per[b].to_numpy()
        lo, hi = bca_interval(lambda i, num=num, den=den: num[i].sum() / den[i].sum(),
                              len(per), n_boot=n_boot, seed=seed)
        diff = np.log(ratio.to_numpy()) - np.log(spec.margin)
        if len(per) >= 2 and np.any(diff != 0):
            p = float(stats.wilcoxon(diff, alternative="less").pvalue)
        else:
            p = float("nan")
        out[name] = {"units": int(len(per)), "cells": int(len(d)),
                     "pooled_ratio": float(num.sum() / den.sum()),
                     "pooled_ratio_bca90": [lo, hi],
                     "per_unit_ratio": {u: float(v) for u, v in ratio.items()},
                     "wilcoxon_p_one_sided": p,
                     "passes": bool(np.isfinite(p) and p < spec.alpha)}
    out["feasible_share_of_replications"] = float(
        (cells["feasible"] * cells["n_rep"]).sum() / cells["n_rep"].sum())
    out["cells_never_feasible"] = float((cells["n_feasible"] == 0).mean())
    out["passes"] = bool(out["itt"]["passes"] and out["per_protocol"]["passes"])
    return out


def _pe2(cells: pd.DataFrame, spec: EndpointSpec) -> dict:
    cov = cells["cover_itt"]
    wr = cells["width_itt"] / (2 * 1.96 * cells["sd_itt"])
    mcse = np.sqrt(cov * (1 - cov) / cells["n_rep"])
    out = {"cells": int(len(cells)), "mean_coverage": float(cov.mean()),
           "share_cells_ge_090": float((cov >= spec.coverage_cell_min).mean()),
           "min_coverage": float(cov.min()), "median_mc_se_per_cell": float(mcse.median()),
           "median_width_ratio": float(wr.median())}
    if "budget" in cells:
        out["by_budget"] = {
            f"{b:g}": {"mean_coverage": float(g["cover_itt"].mean()),
                       "share_cells_ge_090": float((g["cover_itt"] >= spec.coverage_cell_min)
                                                   .mean()),
                       "median_width_ratio": float((g["width_itt"] / (2 * 1.96 * g["sd_itt"]))
                                                   .median())}
            for b, g in cells.groupby("budget")}
    out["passes"] = bool(out["mean_coverage"] >= spec.coverage_mean_min
                         and out["share_cells_ge_090"] >= spec.coverage_share_min)
    return out


def _pe3(clusters: pd.DataFrame, cert_cells: pd.DataFrame, spec: EndpointSpec, n_boot: int,
         seed: int) -> dict:
    out = {}
    for name in ("bad", "near"):
        n, k = clusters[f"n_{name}"].to_numpy(), clusters[f"k_{name}"].to_numpy()
        tot, hit = int(n.sum()), int(k.sum())
        out[name] = {"replications": tot, "rate": hit / tot if tot else float("nan"),
                     "upper95_cluster_bootstrap": _pattern_bootstrap_upper(n, k, n_boot, seed),
                     "upper95_clopper_pearson": float(stats.beta.ppf(0.95, hit + 1, tot - hit))
                     if tot > hit else (1.0 if tot else float("nan"))}
    big = cert_cells[cert_cells["n_bad"] >= 30]
    out["worst_cell_rate"] = float((big["k_bad"] / big["n_bad"]).max()) if len(big) else None
    ns, ks = int(clusters["n_small"].sum()), int(clusters["k_small"].sum())
    out["power_small_bias"] = ks / ns if ns else float("nan")
    # a category without replications cannot be evaluated; PE3 needs the overall bound and
    # every evaluable bound to be within the limit
    ups = [out[n]["upper95_cluster_bootstrap"] for n in ("bad", "near")
           if out[n]["replications"] > 0]
    out["evaluable"] = bool(out["bad"]["replications"] > 0)
    out["passes"] = bool(out["evaluable"] and all(u <= spec.false_cert_max for u in ups))
    return out


def _pe4(cells: pd.DataFrame, spec: EndpointSpec, n_boot: int, seed: int) -> dict:
    rules = [r for r in dict.fromkeys((spec.policy, *spec.singles)) if f"mse_{r}" in cells]
    srmse = np.sqrt(cells[[f"mse_{r}" for r in rules]]).set_axis(rules, axis=1)
    best = srmse[[r for r in spec.singles if r in rules]].min(axis=1)
    excess = srmse.sub(best, axis=0)
    excess["_unit"] = cells["_unit"].to_numpy()

    def table(ex: pd.DataFrame) -> dict:
        per_unit = ex.groupby("_unit")[rules].max().to_numpy()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(per_unit), size=(n_boot, len(per_unit)))
        boot = per_unit[idx].max(axis=1)                # (n_boot, rules)
        res = {}
        for j, r in enumerate(rules):
            res[r] = {"max_excess_srmse": float(ex[r].max()),
                      "boot90": [float(np.quantile(boot[:, j], 0.05)),
                                 float(np.quantile(boot[:, j], 0.95))],
                      "mean_excess_srmse": float(ex[r].mean()),
                      "pooled_srmse": float(np.sqrt((srmse.loc[ex.index, r] ** 2).mean()))}
        lowest = min(rules, key=lambda r: res[r]["max_excess_srmse"])
        return {"rules": res, "lowest_worst_case": lowest,
                "policy_is_lowest": bool(lowest == spec.policy)}

    out = {"pooled": table(excess)}
    if "budget" in cells:
        out["by_budget"] = {f"{b:g}": table(excess.loc[g.index])
                            for b, g in cells.groupby("budget")}
    return out


def evaluate(cells: pd.DataFrame, clusters: pd.DataFrame, cert_cells: pd.DataFrame,
             spec: EndpointSpec = DEFAULT, n_boot: int = 4000, seed: int = 1) -> dict:
    """Endpoints PE1-PE4 from the reductions of all pools."""
    cells = cells.reset_index(drop=True).copy()
    cells["_unit"] = _unit_id(cells, spec)
    return {"units": sorted(cells["_unit"].unique()), "cells": int(len(cells)),
            "PE1": _pe1(cells, spec, n_boot, seed), "PE2": _pe2(cells, spec),
            "PE3": _pe3(clusters, cert_cells, spec, n_boot, seed),
            "PE4": _pe4(cells, spec, n_boot, seed)}
