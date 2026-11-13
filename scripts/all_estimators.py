"""All nine estimators of the primary grid on the corpora whose raw replication files are
kept here (everything except RAID): pooled MSE relative to flag-PPI, mean coverage of the
95% interval and feasibility. An estimator that is infeasible in a replication is replaced by
flag-PPI on the same audit (the intention-to-treat convention of the endpoints).

    python scripts/all_estimators.py results/raw/S5/primary --out results/S5_all_estimators.csv
"""
import argparse
import glob

import numpy as np
import pandas as pd

EST = ["naive", "deletion", "moment", "hajek", "flag_ppi", "flag_ppi_pp", "rc_ppi", "rc_ppi_pp",
       "greg", "policy_debiased"]
ap = argparse.ArgumentParser()
ap.add_argument("raw_dir")
ap.add_argument("--out", default="results/S5_all_estimators.csv")
args = ap.parse_args()
parts = []
for f in sorted(glob.glob(f"{args.raw_dir}/*.parquet")):
    d = pd.read_parquet(f, columns=["corpus", "pool", "budget", "eps", "outcome", "rep", "estimator",
                                    "point", "ci_low", "ci_high", "feasible", "truth", "sd_h"])
    d = d[d.estimator.isin(EST)]
    key = ["pool", "budget", "eps", "outcome", "rep"]
    fb = d[d.estimator == "flag_ppi"].set_index(key)[["point", "ci_low", "ci_high"]]
    d = d.join(fb.add_suffix("_fb"), on=key)
    bad = ~d.feasible.astype(bool) | d.point.isna()
    for c in ("point", "ci_low", "ci_high"):
        d.loc[bad, c] = d.loc[bad, c + "_fb"]
    d["se2"] = ((d.point - d.truth) / d.sd_h) ** 2
    d["cover"] = (d.ci_low <= d.truth) & (d.truth <= d.ci_high)
    parts.append(d.groupby(["corpus", "pool", "budget", "eps", "outcome", "estimator"])
                 .agg(mse=("se2", "mean"), cover=("cover", "mean"), feasible=("feasible", "mean"))
                 .reset_index())
cells = pd.concat(parts)
ref = cells[cells.estimator == "flag_ppi"].set_index(["pool", "budget", "eps", "outcome"]).mse
cells = cells.join(ref.rename("mse_flag"), on=["pool", "budget", "eps", "outcome"])
rows = []
for (corpus, est), g in pd.concat([cells, cells.assign(corpus="all five")]).groupby(["corpus", "estimator"]):
    rows.append({"corpus": corpus, "estimator": est, "cells": len(g),
                 "mse_vs_flag_ppi": g.mse.sum() / g.mse_flag.sum(),
                 "srmse": float(np.sqrt(g.mse.mean())), "coverage": g.cover.mean(),
                 "feasible": g.feasible.mean()})
out = pd.DataFrame(rows)
out.to_csv(args.out, index=False)
print(out[out.corpus == "all five"].round(3).to_string())
