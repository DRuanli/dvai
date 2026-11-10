"""S4, final round: policy, certifier and feasibility threshold on the chosen design.

Runs after ``run_s4b_design_dev.py`` has chosen the audit design (``--design``), on the
regime of the primary endpoints: both development corpora (15 pools), N = 900, iid
mixture, contamination 0.01-0.45, audit budget 2%, 5%, 10%. Candidates:

* policies - squared bias ``plugin`` or ``debiased``, with or without requiring TOST
  certification before deletion may be chosen;
* certifiers - TOST with tau = 0.1, 0.2, 0.3 SD (Student quantile, flag-PPI fallback),
  TOST with the normal quantile and no fallback, and the v1 rule;
* feasibility threshold of RC-PPI++ - 4 (2 per fold, current) and 6 (3 per fold).

Selection rules, fixed before this run: the policy with the lowest worst-cell excess
standardised RMSE over the best single estimator of the cell (PE4 metric), ties broken by
pooled standardised RMSE; the feasibility threshold with the lower mean cell
intention-to-treat standardised RMSE. The certifier is reported, not chosen: tau = 0.2 SD
is a substantive choice.

Writes one raw file per pool to ``results/raw/S4c/`` (resumable) and
``results/S4c_regret.csv``, ``results/S4c_regret_by_budget.csv``,
``results/S4c_certification.csv``, ``results/S4c_feasibility.csv``,
``results/S4c_interval_rules.csv`` (linearised, Satterthwaite, pooled and model-based
shrinkage on the chosen design), ``results/S4c_fallback_cells.csv`` (flag-PPI vs Hajek
fallback) and ``results/S4c_manifest.json`` (design and grid of the run).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_s4_policy_dev as pol
import run_s4_variance_dev as base

from d2vi.decision.certify import LegacyCertifier, TOSTCertifier
from d2vi.decision.policy import MSEPolicy
from d2vi.estimators import RCPPI, CrossFitPropensity, make_estimator
from d2vi.sampling.mixture import FeaturePool
from d2vi.sim.runner import CellSpec, Experiment

OUTCOMES = ["read", "sent", "trust", "subj"]
SINGLE = ["naive", "deletion", "hajek", "flag_ppi", "rc_ppi_pp"]
EPS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.45)
BUDGETS = (0.02, 0.05, 0.10)
CELL = ["det", "budget", "eps", "domain", "generator", "outcome"]
RAW = Path("results/raw/S4c")
KEEP = CELL + ["design", "rep", "estimator", "point", "ci_low", "ci_high", "feasible", "truth",
               "sd_h", "deletion_bias", "certified", "tau", "choice", "x_se_lin", "x_df_lin",
               "x_df_satt", "x_se_pool", "x_df_pool", "x_se_model", "x_df_model"]
RULES = {"lin": ("x_se_lin", "x_df_lin"), "satt": ("x_se_lin", "x_df_satt"),
         "pool": ("x_se_pool", "x_df_pool"), "model": ("x_se_model", "x_df_model")}


def experiment(sub: pd.DataFrame) -> Experiment:
    mc6 = RCPPI(tune=True, propensity=CrossFitPropensity(min_class=6, min_class_fold=3))
    mc6.name = "rc_ppi_pp_mc6"
    policies = [MSEPolicy(b, c) for b in ("plugin", "debiased") for c in (False, True)]
    certifiers = [TOSTCertifier(tau_sd=t) for t in (0.1, 0.2, 0.3)]
    certifiers += [TOSTCertifier(tau_sd=0.2, quantile="z", fallback=False),
                   LegacyCertifier()]
    return Experiment(FeaturePool(sub), [make_estimator(e) for e in SINGLE] + [mc6],
                      policies=policies, certifiers=certifiers, outcomes=OUTCOMES)


def run(design, n_rep: int, n_jobs: int) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for det, path in base.POOLS.items():
        for corpus, label, tags, sub in base._pools(det, path):
            out = RAW / f"{det}__{corpus}__{label.replace('/', '-')}.parquet"
            if out.exists():
                continue
            exp = experiment(sub)
            specs = [CellSpec(corpus, label, e, 900, b, design, tags=tags)
                     for b in BUDGETS for e in EPS]
            frames = [exp.run_cell(sp, n_rep, n_jobs).reindex(columns=KEEP) for sp in specs]
            pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
            print("done", out.name, flush=True)


def _itt(est: pd.DataFrame, name: str) -> pd.DataFrame:
    """Point and interval of RC-PPI++ variant ``name``, flag-PPI where it is infeasible."""
    key = CELL + ["rep"]
    cols = ["point", "ci_low", "ci_high"]
    rc = est[est.estimator == name].set_index(key)
    fb = est[est.estimator == "flag_ppi"].set_index(key)[cols].reindex(rc.index)
    ok = rc["feasible"].astype(bool)
    return rc[cols].where(ok, fb, axis=0)


def cell_errors(raw: pd.DataFrame) -> pd.DataFrame:
    """Per cell: standardised RMSE of every rule (RC-PPI++ variants intention-to-treat)
    and coverage of the interval each rule reports."""
    est = raw[~raw.estimator.str.startswith("cert_")].copy()
    key = CELL + ["rep"]
    for name in ("rc_ppi_pp", "rc_ppi_pp_mc6"):
        sel = est.estimator == name
        vals = _itt(est, name).reindex(est.loc[sel].set_index(key).index)
        est.loc[sel, ["point", "ci_low", "ci_high"]] = vals.to_numpy()
    est["se2"] = ((est["point"] - est["truth"]) / est["sd_h"]) ** 2
    est["cover"] = ((est["ci_low"] <= est["truth"]) & (est["truth"] <= est["ci_high"]))
    g = est.groupby(CELL + ["estimator"])
    out = g["se2"].mean().pow(0.5).unstack("estimator")
    cov = g["cover"].mean().unstack("estimator").add_prefix("cover_")
    return out.join(cov)


def interval_rules(raw: pd.DataFrame) -> pd.DataFrame:
    """Coverage and width of each interval rule, per cell, for flag-PPI and feasible
    RC-PPI++ (the interval rule was chosen in round one; this checks it on the design)."""
    from scipy import stats

    d = raw[raw.estimator.isin(["flag_ppi", "rc_ppi_pp"]) & raw.feasible.astype(bool)].copy()
    d["err"] = (d.point - d.truth) / d.sd_h
    out = []
    for name, (se, df) in RULES.items():
        half = stats.t.ppf(0.975, d[df].clip(lower=1)) * d[se]
        d[f"cov_{name}"] = (d.point - half <= d.truth) & (d.truth <= d.point + half)
        d[f"w_{name}"] = 2 * half / d.sd_h
    g = d.groupby(CELL + ["estimator"])
    agg = {f"cov_{r}": (f"cov_{r}", "mean") for r in RULES}
    agg |= {f"w_{r}": (f"w_{r}", "mean") for r in RULES}
    cells = g.agg(n=("err", "size"), sd_err=("err", "std"), **agg).reset_index()
    cells = cells[cells.n >= 50]
    for r in RULES:
        cells[f"wr_{r}"] = cells[f"w_{r}"] / (2 * 1.96 * cells["sd_err"])
    out.append(cells)
    return pd.concat(out, ignore_index=True)


def fallback_compare(raw: pd.DataFrame) -> pd.DataFrame:
    """Per cell: ITT standardised RMSE of RC-PPI++ with the flag-PPI and with the Hajek
    fallback, and the feasibility rate (evidence for the frozen fallback)."""
    key = CELL + ["rep"]
    d = raw[raw.estimator.isin(["rc_ppi_pp", "flag_ppi", "hajek"])]
    w = d.pivot_table(index=key, columns="estimator", values="point")
    rc = d[d.estimator == "rc_ppi_pp"].set_index(key).reindex(w.index)
    ok = rc["feasible"].astype(bool)
    out = {}
    for fb in ("flag_ppi", "hajek"):
        e2 = ((w["rc_ppi_pp"].where(ok, w[fb]) - rc["truth"]) / rc["sd_h"]) ** 2
        out[f"srmse_{fb}_fallback"] = e2.groupby(level=CELL).mean().pow(0.5)
    out["feasible"] = ok.astype(float).groupby(level=CELL).mean()
    return pd.DataFrame(out).reset_index()


def regret(cell: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    rows = []
    best = cell[SINGLE].min(axis=1)
    pols = [c for c in cell.columns if c.startswith("policy_")]
    groups = [((), cell.index)] if not by else list(cell.groupby(level=by).groups.items())
    for gkey, idx in groups:
        for name in pols + SINGLE:
            d = (cell.loc[idx, name] - best.loc[idx])
            rows.append({**dict(zip(by or [], gkey if isinstance(gkey, tuple) else (gkey,),
                                    strict=False)),
                         "rule": name, "max_excess_srmse": d.max(),
                         "p90_excess_srmse": d.quantile(0.9), "mean_excess_srmse": d.mean(),
                         "pooled_srmse": float(np.sqrt((cell.loc[idx, name] ** 2).mean())),
                         "max_ratio_regret": float((cell.loc[idx, name] / best.loc[idx]).max()),
                         "cover_mean": float(cell.loc[idx, f"cover_{name}"].mean())
                         if f"cover_{name}" in cell else np.nan,
                         "cover_cells_ge90": float((cell.loc[idx, f"cover_{name}"] >= 0.90).mean())
                         if f"cover_{name}" in cell else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, help="design name or JSON spec")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--tag", default="S4c", help="prefix of the raw directory and outputs")
    args = ap.parse_args()
    global RAW
    RAW = Path("results/raw") / args.tag
    design = json.loads(args.design) if args.design.startswith("{") else args.design
    t0 = time.time()
    run(design, args.reps, args.jobs)
    files = sorted(RAW.glob("*.parquet"))
    cell = pd.concat([cell_errors(pd.read_parquet(f)) for f in files])
    cert = pol.certification(pd.concat(
        [d[d.estimator.str.startswith("cert_")] for d in map(pd.read_parquet, files)],
        ignore_index=True))
    reg = regret(cell).sort_values(["max_excess_srmse", "pooled_srmse"])
    reg.to_csv(f"results/{args.tag}_regret.csv", index=False)
    regret(cell, ["budget"]).to_csv(f"results/{args.tag}_regret_by_budget.csv", index=False)
    cert.to_csv(f"results/{args.tag}_certification.csv", index=False)
    feas = pd.DataFrame({
        name: {"mean_cell_srmse": cell[name].mean(), "max_cell_srmse": cell[name].max(),
               "cells_better_than_other": float((cell[name] < cell[other]).mean())}
        for name, other in (("rc_ppi_pp", "rc_ppi_pp_mc6"), ("rc_ppi_pp_mc6", "rc_ppi_pp"))}).T
    feas.to_csv(f"results/{args.tag}_feasibility.csv")
    fbc = pd.concat([fallback_compare(pd.read_parquet(f)) for f in files], ignore_index=True)
    fbc.to_csv(f"results/{args.tag}_fallback_cells.csv", index=False)
    print(fbc.groupby("eps")[["srmse_flag_ppi_fallback", "srmse_hajek_fallback"]].mean()
          .round(4))
    Path(f"results/{args.tag}_manifest.json").write_text(json.dumps(
        {"design": design, "reps": args.reps, "budgets": BUDGETS, "eps": EPS,
         "pools": [f.name for f in files]}, indent=2))
    ir = pd.concat([interval_rules(pd.read_parquet(f)) for f in files], ignore_index=True)
    irs = ir.groupby(["estimator", "budget"]).agg(
        **{f"mean_{r}": (f"cov_{r}", "mean") for r in RULES},
        **{f"ge90_{r}": (f"cov_{r}", lambda v: float(np.mean(v >= 0.90))) for r in RULES},
        **{f"wr_{r}": (f"wr_{r}", "median") for r in RULES}).reset_index()
    irs.to_csv(f"results/{args.tag}_interval_rules.csv", index=False)
    print(irs.round(3).to_string(index=False))
    pd.set_option("display.width", 250)
    print(reg.round(4).to_string(index=False))
    print(cert.round(4).to_string(index=False))
    print(feas.round(4))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
