"""Secondary comparisons as frozen in the protocol (section 4): within each family, a
Friedman test across corpora, then Wilcoxon signed-rank tests of each level against the
reference level with Holm correction, and effect sizes with bootstrap intervals.

    python scripts/secondary_tests.py results/compact/S5 --out results/S5_secondary_tests.csv

Units are corpora (blocks). The response is the corpus mean, over cells, of the ITT
standardised RMSE and of ITT coverage. With six corpora the smallest two-sided Wilcoxon
p-value is 1/32, so these tests can only detect differences that are consistent in every
corpus; the effect sizes are the informative part.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FAMILIES = {  # grid: (column, reference level)
    "designs": ("design", "proportional"),
    "mixtures": ("mixture", "clustered"),
    "verification_error": ("phi", "phi0=0.0,phi1=0.0"),
}


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def boot_ci(x: np.ndarray, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="results/S5_secondary_tests.csv")
    args = ap.parse_args()
    rows = []
    for grid, (col, ref) in FAMILIES.items():
        sec = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{args.root}/{grid}/*.sec.parquet")])
        sec["level"] = sec[col].astype(str)
        sec["level"] = sec["level"].where(~sec["level"].str.startswith(ref), ref)
        sec["srmse"] = np.sqrt(sec["itt_mse"])
        for metric in ("srmse", "cover"):
            tab = sec.groupby(["corpus", "level"])[metric].mean().unstack()
            levels = [c for c in tab.columns if c != ref]
            fried = stats.friedmanchisquare(*[tab[c].to_numpy() for c in tab.columns]).pvalue
            diffs = {lv: (tab[lv] - tab[ref]).to_numpy() for lv in levels}
            p = np.array([stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0 for d in diffs.values()])
            for (lv, d), padj, praw in zip(diffs.items(), holm(p), p):
                lo, hi = boot_ci(d)
                rows.append({"family": grid, "metric": metric, "reference": ref, "level": lv[:40],
                             "corpora": len(d), "mean_diff": d.mean(), "ci90_low": lo,
                             "ci90_high": hi, "corpora_worse": int((d > 0).sum() if metric == "srmse" else (d < 0).sum()),
                             "wilcoxon_p": praw, "holm_p": padj, "friedman_p": fried})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
