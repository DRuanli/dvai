"""S4 / E3 on development corpora: which audit design to freeze (equal budgets).

Same grid as ``run_s4_variance_dev.py`` (Ghostbuster x TF-IDF and Fast-DetectGPT,
3 domains x 2 generators, contamination 0.05-0.30, N = 900, audit of exactly 45), with the
designs SRS, top-decile split, two-stage Neyman and the adaptive constrained Neyman design.
Writes ``results/raw/S4_design_dev.parquet`` and ``results/S4_design_dev_summary.csv``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import run_s4_variance_dev as base

DESIGNS = ["srs", "top_decile", {"name": "neyman", "n_strata": 5}, "adaptive"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    base.DESIGNS = DESIGNS
    t0 = time.time()
    raw = base.run(args.reps, args.jobs)
    raw.to_parquet("results/raw/S4_design_dev.parquet", index=False)
    d = base.add_intervals(raw)
    d["err2"] = (d["point"] - d["truth"]) ** 2
    d["serr2"] = d["err2"] / d["sd_h"] ** 2
    feas = (raw.groupby(["det", "design", "estimator", "eps"])["feasible"].mean()
            .rename("feasible").reset_index())
    acc = (d.groupby(["det", "design", "estimator", "eps"])
           .agg(srmse=("serr2", lambda v: float(np.sqrt(v.mean()))),
                cov_lin=("cov_lin", "mean"), cov_shrink_satt=("cov_shrink_satt", "mean"),
                m_ai=("m_machine", "mean"))
           .reset_index())
    out = feas.merge(acc, on=["det", "design", "estimator", "eps"], how="left")
    out.to_csv("results/S4_design_dev_summary.csv", index=False)
    pd.set_option("display.width", 250)
    r = out[out.estimator == "rc_ppi_pp"]
    for col in ("feasible", "srmse", "cov_shrink_satt", "m_ai"):
        print(col)
        print(r.pivot_table(index=["det", "design"], columns="eps", values=col).round(3))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
