"""Post hoc descriptions of the held-out results (not endpoints; no new simulation).

    python scripts/posthoc_descriptives.py results/compact/S5 \
        --out results/S5_posthoc_descriptives.json --channels results/S5_deletion_channels.csv

Defined on 22-23 November 2026, after the held-out results had been seen; the paper labels
them as post hoc. Definitions are in ``d2vi.analysis.descriptives``.

1. Which single estimator has the lowest MSE in a cell (overall, per corpus, per budget and
   contamination rate).
2. Excess standardised RMSE of every rule over the best single estimator of the cell:
   quantiles, share of cells above 0.05, 0.10 and 0.20 human SD, worst cell.
3. Standardised RMSE of every rule under both pooling conventions (overall, per corpus,
   per budget, per contamination rate), and MSE ratios per contamination rate.
4. The two terms of the deletion bias (Lemma 1) for every evaluated pool whose feature
   table is in the repository (all corpora except ANES, whose records cannot be
   redistributed; its selection term is in ``results/S5_anes_descriptives.json``). Human
   and machine records of a pool are treated as mixed at the contamination rates of the
   primary grid; terms are in units of the human SD of the pool. The predicted bias is
   compared with the simulated deletion error of the same cells as a consistency check.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from d2vi.analysis.descriptives import (
    RULES,
    best_single_estimator,
    excess_summary,
    lemma1_terms,
    rule_summary,
)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_s5", ROOT / "scripts" / "run_s5.py")
run_s5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_s5)

BIAS_THRESHOLD = 0.10      # human SD: a deletion bias at least this large is "material"


def shares(best: pd.Series) -> dict:
    return {k: float(v) for k, v in best.value_counts(normalize=True).sort_index().items()}


def ratio(d: pd.DataFrame, a: str, b: str) -> float:
    return float(d[f"mse_{a}"].sum() / d[f"mse_{b}"].sum())


def cell_descriptives(cells: pd.DataFrame) -> dict:
    best = best_single_estimator(cells)
    res = {
        "cells": int(len(cells)), "pools": int(cells.pool.nunique()),
        "best_single": {
            "overall": shares(best),
            "by_corpus": {c: shares(best[g.index]) for c, g in cells.groupby("corpus")},
            "by_budget_eps": {f"{b:g}|{e:g}": shares(best[g.index])
                              for (b, e), g in cells.groupby(["budget", "eps"])},
        },
        "excess_srmse": {
            "overall": excess_summary(cells),
            "by_budget": {f"{b:g}": excess_summary(g) for b, g in cells.groupby("budget")},
        },
        "srmse": {
            "overall": rule_summary(cells),
            "by_corpus": {c: rule_summary(g) for c, g in cells.groupby("corpus")},
            "by_budget": {f"{b:g}": rule_summary(g) for b, g in cells.groupby("budget")},
            "by_eps": {f"{e:g}": rule_summary(g) for e, g in cells.groupby("eps")},
        },
        "mse_ratios": {
            "by_eps": {f"{e:g}": {"itt_vs_flag_ppi": ratio(g, "rc_ppi_pp", "flag_ppi"),
                                  "deletion_vs_itt": ratio(g, "deletion", "rc_ppi_pp"),
                                  "naive_vs_itt": ratio(g, "naive", "rc_ppi_pp"),
                                  "policy_vs_itt": ratio(g, "policy_debiased", "rc_ppi_pp"),
                                  "feasible": float(g.feasible.mean())}
                       for e, g in cells.groupby("eps")},
            "by_corpus": {c: {f"{r}_vs_flag_ppi": ratio(g, r, "flag_ppi") for r in RULES}
                          | {"pools": int(g.pool.nunique()), "cells": int(len(g)),
                             "coverage_mean": float(g.cover_itt.mean()),
                             "coverage_share_ge_090": float((g.cover_itt >= 0.90).mean())}
                          for c, g in cells.groupby("corpus")},
        },
    }
    return res


def deletion_channels(pools: set[str], eps_grid: list[float]) -> pd.DataFrame:
    """One row per evaluated pool x outcome x contamination rate, in human-SD units."""
    rows = []
    for path in sorted(glob.glob(str(ROOT / "configs" / "s5" / "*.yaml"))):
        cfg = yaml.safe_load(Path(path).read_text())
        feat_path = ROOT / cfg["features"]
        if not feat_path.exists():          # ANES: records are not redistributed
            continue
        feat = pd.read_parquet(feat_path)
        cfg = run_s5.resolve(cfg, feat)
        for name, dom, group, det in run_s5.pool_plan(cfg):
            if name not in pools:           # empty or too small: no cell was run
                continue
            rows_dom = feat if dom is None else feat[feat["domain"].eq(dom)]
            pool, _ = run_s5.build_pool(rows_dom, cfg, group, det)
            a, d = pool["a"].to_numpy(), pool["flag"].to_numpy()
            for o in cfg["outcomes"]:
                y = pool[f"y_{o}"].to_numpy(dtype=float)
                sd = float(y[a == 0].std(ddof=1))
                for eps in eps_grid:
                    t = lemma1_terms(y, a, d, eps)
                    rows.append({"corpus": cfg["corpus"], "pool": name, "detector": det,
                                 "outcome": o, "eps": eps, "f": t["f"], "s": t["s"],
                                 "selection": t["selection"] / sd,
                                 "residual": t["residual"] / sd,
                                 "deletion_bias": t["deletion_bias"] / sd,
                                 "naive_bias": t["naive_bias"] / sd})
    out = pd.DataFrame(rows)
    out["dominant"] = np.where(out.selection.abs() > out.residual.abs(), "selection", "residual")
    return out


def channel_summary(ch: pd.DataFrame, cells: pd.DataFrame) -> dict:
    res = {"threshold_sd": BIAS_THRESHOLD, "pools": int(ch.pool.nunique()),
           "corpora": sorted(ch.corpus.unique()), "by_eps": {}}
    for eps, g in ch.groupby("eps"):
        big = g[g.deletion_bias.abs() >= BIAS_THRESHOLD]
        tab = big.groupby(["corpus", "detector"]).dominant.value_counts().unstack(fill_value=0)
        res["by_eps"][f"{eps:g}"] = {
            "pool_outcomes": int(len(g)), "material": int(len(big)),
            "dominant": {k: int(v) for k, v in big.dominant.value_counts().items()},
            "dominant_by_corpus_detector": {
                f"{c}|{d}": {k: int(v) for k, v in row.items() if v}
                for (c, d), row in tab.iterrows()},
            "deletion_worse_than_naive": float((g.deletion_bias.abs() > g.naive_bias.abs()).mean()),
        }
    sel = ch.drop_duplicates(["pool", "outcome"])           # the selection term has no eps
    worst = sel.loc[sel.selection.abs().idxmax()]
    res["selection_term"] = {
        "max_abs": float(sel.selection.abs().max()),
        "max_at": {"pool": worst.pool, "outcome": worst.outcome, "f": float(worst.f)},
        "share_ge_threshold": float((sel.selection.abs() >= BIAS_THRESHOLD).mean()),
        "max_abs_by_corpus_detector": {
            f"{c}|{d}": float(v) for (c, d), v in
            sel.assign(x=sel.selection.abs()).groupby(["corpus", "detector"]).x.max().items()},
    }
    # consistency: Lemma 1 at the pool level against the simulated deletion error
    sim = cells.drop_duplicates(["pool", "outcome", "eps"])[["pool", "outcome", "eps",
                                                             "mse_deletion"]]
    both = ch.merge(sim, on=["pool", "outcome", "eps"], how="inner")
    rmse = np.sqrt(both.mse_deletion)
    res["check_against_simulation"] = {
        "rows": int(len(both)),
        "correlation": float(np.corrcoef(both.deletion_bias.abs(), rmse)[0, 1]),
        "median_abs_difference_sd": float((both.deletion_bias.abs() - rmse).abs().median()),
        "max_abs_difference_sd": float((both.deletion_bias.abs() - rmse).abs().max()),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="results/S5_posthoc_descriptives.json")
    ap.add_argument("--channels", default="results/S5_deletion_channels.csv")
    args = ap.parse_args()
    files = sorted(glob.glob(str(Path(args.root) / "primary" / "*.cells.parquet")))
    cells = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    res = {"status": "post hoc descriptive analyses, defined after the held-out results "
                     "were seen (22-23 November 2026); not endpoints",
           **cell_descriptives(cells)}
    ch = deletion_channels(set(cells.pool), sorted(float(e) for e in cells.eps.unique()))
    ch.round(6).to_csv(args.channels, index=False)
    res["deletion_channels"] = channel_summary(ch, cells)
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    print("written", args.out, args.channels)


if __name__ == "__main__":
    main()
