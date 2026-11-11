"""Secondary grids of S5: per-factor summaries of the intention-to-treat rule and the
other rules (frozen protocol, section 4, secondary grids).

    python scripts/summarise_secondary.py results/raw/S5 --corpus anes2024 --tag interim_anes

Several roots may be given (e.g. raw files here and compact files from Kaggle).

For every grid directory present, the varying factor of the grid (budget, N, mixture,
design or verification error) is crossed with the budget. Reported per factor level, as
means over cells (corpus x pool x eps x outcome): standardised RMSE of each rule; coverage
(mean and lowest cell) and width of the 95% interval of the ITT rule (RC-PPI++ when
feasible, flag-PPI on the same audit otherwise); feasibility of RC-PPI++.
Writes ``results/S5_<tag>_secondary.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GRIDS = {"small_budget": "budget", "large_n": "n", "mixtures": "mixture",
         "designs": "design", "verification_error": "phi"}
RULES = ["naive", "deletion", "hajek", "flag_ppi", "rc_ppi_pp", "policy_debiased"]
COLS = ["corpus", "pool", "budget", "eps", "n", "design", "mixture", "phi0", "phi1",
        "outcome", "rep", "estimator", "point", "ci_low", "ci_high", "feasible", "truth",
        "sd_h"]


KEYS = ["corpus", "pool", "budget", "eps", "n", "design", "mixture", "phi", "outcome"]


def cell_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per cell: ITT error, coverage, width and feasibility; MSE of the other rules."""
    raw = raw.copy()
    raw["phi"] = "phi0=" + raw["phi0"].astype(str) + ",phi1=" + raw["phi1"].astype(str)
    raw["design"] = raw["design"].astype(str)
    raw["mixture"] = raw["mixture"].astype(str)
    raw["e2"] = ((raw["point"] - raw["truth"]) / raw["sd_h"]) ** 2
    raw["cov"] = (raw["ci_low"] <= raw["truth"]) & (raw["truth"] <= raw["ci_high"])
    raw["width"] = (raw["ci_high"] - raw["ci_low"]) / raw["sd_h"]
    sub = raw[raw["estimator"].isin(RULES)]
    rc = sub[sub["estimator"].eq("rc_ppi_pp")].set_index([*KEYS, "rep"])
    fb = sub[sub["estimator"].eq("flag_ppi")].set_index([*KEYS, "rep"]).reindex(rc.index)
    use = rc["feasible"].astype(bool)
    itt = pd.DataFrame({"e2": np.where(use, rc["e2"], fb["e2"]),
                        "cov": np.where(use, rc["cov"], fb["cov"]),
                        "width": np.where(use, rc["width"], fb["width"]),
                        "feas": use}, index=rc.index).reset_index()
    cell = itt.groupby(KEYS).agg(itt_mse=("e2", "mean"), cover=("cov", "mean"),
                                 width=("width", "mean"), feasible=("feas", "mean"))
    # every rule except RC-PPI++ is reported for every replication
    other = (sub[~sub["estimator"].eq("rc_ppi_pp")].groupby([*KEYS, "estimator"])["e2"]
             .mean().unstack("estimator"))
    return cell.join(other).reset_index()


def summarise(cell: pd.DataFrame, factor: str) -> pd.DataFrame:
    out = []
    for (level, budget), g in cell.groupby([factor, "budget"]):
        row = {"grid_factor": factor, "level": level, "budget": budget, "cells": len(g),
               "srmse_itt": float(np.sqrt(g["itt_mse"]).mean()),
               "cover_mean": float(g["cover"].mean()), "cover_min": float(g["cover"].min()),
               "width_mean_sd": float(g["width"].mean()), "feasible": float(g["feasible"].mean())}
        for r in RULES:
            if r in g and r != "rc_ppi_pp":
                row[f"srmse_{r}"] = float(np.sqrt(g[r]).mean())
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="+", help="roots holding one directory per grid")
    ap.add_argument("--corpus")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    frames = []
    for grid, factor in GRIDS.items():
        tables = {}
        for root in args.root:
            d = Path(root, grid)
            for f in sorted(d.glob("*.sec.parquet")):           # compact (compact_s5.py)
                tables[f.name[: -len(".sec.parquet")]] = pd.read_parquet(f)
            for f in sorted(d.glob("*.parquet")):
                if f.name.count(".") == 1 and f.stem not in tables:
                    tables[f.stem] = cell_table(pd.read_parquet(f, columns=COLS))
        if args.corpus:
            tables = {k: v for k, v in tables.items() if k.startswith(f"{args.corpus}__")}
        if not tables:
            continue
        files = list(tables)
        cells = pd.concat(tables.values(), ignore_index=True)
        s = summarise(cells, factor)
        s.insert(0, "grid", grid)
        s.insert(1, "pools", len(files))
        frames.append(s)
        print(grid, len(files), "pools")
    out = pd.concat(frames, ignore_index=True)
    path = Path(f"results/S5_{args.tag}_secondary.csv")
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(out.round(3).to_string(index=False))
    print("written", path)


if __name__ == "__main__":
    main()
