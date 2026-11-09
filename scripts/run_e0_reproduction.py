"""E0 (stage S3): reproduce the first submission's Study 1 tables and measure the change.

Three runs on the Ghostbuster TF-IDF feature table, for 3 domains x 2 generators x
4 contamination rates, N = 900, audit budget 5%:

1. ``legacy8``   - the v1 code path (``d2vi.estimators.legacy``), 8 seeds as in the paper;
2. ``legacy200`` - the same code path with 200 seeds, to separate Monte Carlo noise from
   genuine differences;
3. ``corrected`` - the corrected pipeline (budget-matched shared audits, no fallback,
   cross-PPI, design-based standard errors), 200 replications.

Writes raw results to ``results/raw/E0_*.parquet`` and a comparison table to
``results/E0_comparison.csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from d2vi.estimators import make_estimator
from d2vi.estimators.legacy import legacy_replication
from d2vi.sampling.mixture import FeaturePool
from d2vi.sim.metrics import summarize
from d2vi.sim.runner import CellSpec, Experiment

DOMAINS = ("essay", "reuter", "wp")
GENERATORS = ("gpt", "claude")
EPS = (0.05, 0.10, 0.20, 0.30)
OUTCOMES = ["read", "sent"]


def _arrays(df):
    return {"score": df["score"].to_numpy(), "flag": df["flag"].to_numpy(),
            "length": df["length"].to_numpy(),
            **{f"y_{o}": df[f"y_{o}"].to_numpy() for o in OUTCOMES}}


def run_legacy(feat: pd.DataFrame, n_seeds: int, n_jobs: int) -> pd.DataFrame:
    jobs = []
    for dom in DOMAINS:
        h = _arrays(feat[(feat.domain == dom) & (feat.source == "human")])
        for gen in GENERATORS:
            m = _arrays(feat[(feat.domain == dom) & (feat.source == gen)])
            for eps in EPS:
                for seed in range(n_seeds):
                    jobs.append((dom, gen, h, m, eps, seed))

    def one(dom, gen, h, m, eps, seed):
        rows = legacy_replication(h, m, 900, eps, seed, 0.05, OUTCOMES)
        for r in rows:
            r.update(domain=dom, generator=gen)
        return rows

    out = Parallel(n_jobs=n_jobs)(delayed(one)(*j) for j in jobs)
    return pd.DataFrame([r for chunk in out for r in chunk])


def legacy_rmse(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    cols = ["naive", "deletion", "ppi", "rc_strat", "rc_strat_raw"]
    d = df.copy()
    for c in cols:
        d[c] = d[c] - d["truth"]
    agg = d.groupby(by).agg(**{c: (c, lambda e: float(np.sqrt(np.nanmean(e ** 2))))
                               for c in cols},
                            fallback_share=("rc_strat_raw", lambda v: float(np.isnan(v).mean())))
    return agg.reset_index()


def run_corrected(feat: pd.DataFrame, n_rep: int, n_jobs: int) -> pd.DataFrame:
    frames = []
    for dom in DOMAINS:
        for gen in GENERATORS:
            sub = feat[(feat.domain == dom) & feat.source.isin(["human", gen])]
            pool = FeaturePool(sub)
            exp = Experiment(pool, [make_estimator(e) for e in
                                    ["naive", "deletion", "hajek", "flag_ppi", "flag_ppi_pp",
                                     "rc_ppi", "rc_ppi_pp"]], outcomes=OUTCOMES)
            specs = [CellSpec(f"ghostbuster-{dom}", f"tfidf_lr/{gen}", e, 900, 0.05, d,
                              tags={"domain": dom, "generator": gen})
                     for e in EPS for d in ("srs", "top_decile")]
            frames.append(exp.run(specs, n_rep, n_jobs))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/ghostbuster_tfidf.parquet")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    feat = pd.read_parquet(args.features)
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    l8 = run_legacy(feat, 8, args.jobs)
    l8.to_parquet("results/raw/E0_legacy8.parquet", index=False)
    lmany = run_legacy(feat, args.seeds, args.jobs)
    lmany.to_parquet(f"results/raw/E0_legacy{args.seeds}.parquet", index=False)
    corr = run_corrected(feat, args.reps, args.jobs)
    corr.to_parquet("results/raw/E0_corrected.parquet", index=False)

    paper = pd.read_csv("docs/paper_v1_tables.csv")
    est_cols = ["naive", "deletion", "flag_ppi", "strat_rc_ppi"]
    paper = paper.rename(columns={c: f"paper_{c}" for c in est_cols})

    def tag(df, keys, prefix):
        return df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c not in keys})

    k2 = ["outcome", "eps"]
    kd = ["outcome", "domain", "generator", "eps"]
    t2 = (paper[paper.table == "T2"].drop(columns=["domain", "generator"])
          .merge(tag(legacy_rmse(l8, k2), k2, "l8"), on=k2)
          .merge(tag(legacy_rmse(lmany, k2), k2, "lm"), on=k2))
    t2.to_csv("results/E0_table2_comparison.csv", index=False)
    d2 = (paper[paper.table == "D2"]
          .merge(tag(legacy_rmse(l8, kd), kd, "l8"), on=kd)
          .merge(tag(legacy_rmse(lmany, kd), kd, "lm"), on=kd))
    d2.to_csv("results/E0_tableD2_comparison.csv", index=False)

    # corrected pipeline pooled over domains x generators (same pooling as Table 2)
    corr_pool = corr.copy()
    corr_pool["corpus"] = "ghostbuster"
    corr_pool["detector"] = "tfidf_lr"
    s = summarize(corr_pool, ["corpus", "detector", "eps", "design", "outcome"])
    s.to_csv("results/E0_corrected_summary.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
