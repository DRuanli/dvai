"""S4 / E4 on development corpora: which interval method to freeze.

Runs Ghostbuster with both detectors (TF-IDF, Fast-DetectGPT), 3 domains x 2 generators,
contamination 0.05-0.30, N = 900, audit 5%, designs SRS / top-decile / two-stage Neyman,
and records for every design-based estimator the linearised SE, the Satterthwaite degrees
of freedom and the shrunk SE, so that four interval rules can be compared post hoc:

* ``lin``          - t interval, df = m - H (current default);
* ``satt``         - t interval, Satterthwaite df;
* ``shrink``       - shrunk stratum variances, df = m - H;
* ``shrink_satt``  - both.

Output: ``results/raw/S4_variance_dev.parquet`` and ``results/S4_variance_dev_summary.csv``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats

from d2vi.estimators import make_estimator
from d2vi.sampling.mixture import FeaturePool
from d2vi.sim.runner import CellSpec, Experiment

POOLS = {"tfidf_lr": "data/features/ghostbuster_tfidf.parquet",
         "fast_detectgpt": "data/features/ghostbuster_fdgpt.parquet"}
OUTCOMES = ["read", "sent", "trust", "subj"]
ESTIMATORS = ["hajek", "flag_ppi", "rc_ppi", "rc_ppi_pp"]
DESIGNS = ["srs", "top_decile", {"name": "neyman", "n_strata": 5}]
EPS = (0.05, 0.10, 0.20, 0.30)


def run(n_rep: int, n_jobs: int) -> pd.DataFrame:
    frames = []
    for det, path in POOLS.items():
        feat = pd.read_parquet(path)
        keep = ["doc_id", "domain", "source", "a", "generator", "score", "flag", "length"]
        feat = feat[keep + [f"y_{o}" for o in OUTCOMES]]
        for dom in ("essay", "reuter", "wp"):
            for gen in ("gpt", "claude"):
                sub = feat[(feat.domain == dom) & feat.source.isin(["human", gen])]
                exp = Experiment(FeaturePool(sub), [make_estimator(e) for e in ESTIMATORS],
                                 outcomes=OUTCOMES)
                specs = [CellSpec(f"ghostbuster-{dom}", f"{det}/{gen}", e, 900, 0.05, d,
                                  tags={"domain": dom, "generator": gen, "det": det})
                         for e in EPS for d in DESIGNS]
                frames.append(exp.run(specs, n_rep, n_jobs))
    return pd.concat(frames, ignore_index=True)


def add_intervals(df: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    d = df[df.feasible.astype(bool)].copy()
    q = 0.5 + level / 2
    rules = {"lin": ("se", "x_df"), "satt": ("se", "x_df_satt"),
             "shrink": ("x_se_shrink", "x_df"), "shrink_satt": ("x_se_shrink", "x_df_satt")}
    for name, (se_col, df_col) in rules.items():
        half = stats.t.ppf(q, d[df_col].clip(lower=1)) * d[se_col]
        d[f"cov_{name}"] = (d["point"] - half <= d["truth"]) & (d["truth"] <= d["point"] + half)
        d[f"w_{name}"] = 2 * half
    return d


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    cell = ["det", "design", "estimator", "eps", "domain", "generator", "outcome"]
    rules = ["lin", "satt", "shrink", "shrink_satt"]
    per = d.groupby(cell).agg(n=("point", "size"),
                              **{f"cov_{r}": (f"cov_{r}", "mean") for r in rules},
                              **{f"w_{r}": (f"w_{r}", "mean") for r in rules}).reset_index()
    per = per[per.n >= 50]
    out = per.groupby(["det", "design", "estimator"]).agg(
        cells=("n", "size"),
        **{f"mean_{r}": (f"cov_{r}", "mean") for r in rules},
        **{f"p90cells_{r}": (f"cov_{r}", lambda v: float(np.mean(v >= 0.90))) for r in rules},
        **{f"wratio_{r}": (f"w_{r}", "mean") for r in rules}).reset_index()
    for r in rules[1:]:
        out[f"wratio_{r}"] = out[f"wratio_{r}"] / out["wratio_lin"]
    out["wratio_lin"] = 1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    raw = run(args.reps, args.jobs)
    raw.to_parquet("results/raw/S4_variance_dev.parquet", index=False)
    d = add_intervals(raw[raw.estimator.isin(ESTIMATORS)])
    s = summarize(d)
    s.to_csv("results/S4_variance_dev_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
