"""S4 / E4 on development corpora: which interval method to freeze.

Runs both development corpora - Ghostbuster with two detectors (TF-IDF, Fast-DetectGPT;
3 domains x 2 generators) and HC3 with the RoBERTa detector (3 sources) - at contamination
0.05-0.30, N = 900, audit 5%, designs SRS / top-decile / two-stage Neyman / adaptive, and
records for every design-based estimator the SE and degrees of freedom of five interval
rules, compared post hoc:

* ``lin``    - linearised variance, t interval with df = m - H;
* ``satt``   - linearised variance, Satterthwaite df;
* ``pool``   - stratum variances shrunk to the pooled within-stratum variance, Satterthwaite;
* ``jeffreys`` - stratum variances shrunk (one pseudo-record) to the variance anticipated
  under a Bernoulli origin model with Jeffreys-smoothed stratum rates, Satterthwaite;
* ``eb05``, ``eb1``, ``eb2``, ``eb4`` - the same with empirical-Bayes stratum rates shrunk
  towards the audit's machine share with 0.5, 1, 2 or 4 pseudo-records.

A floor at the anticipated variance (instead of shrinkage) was also tried on 9 Nov 2026
and dropped: median interval width 1.4-2.3 times the oracle width.

The first version of this script (9 Nov 2026, morning) compared a ``shrink`` rule whose
implementation shrank towards a pooled sum of squares instead of a pooled variance; it
inflated small strata and its results are superseded (docs/FINDINGS_V1.md, F14).

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
         "fast_detectgpt": "data/features/ghostbuster_fdgpt.parquet",
         "roberta_hc3": "data/features/hc3_roberta.parquet"}
OUTCOMES = ["read", "sent", "trust", "subj"]
ESTIMATORS = ["hajek", "flag_ppi", "rc_ppi_pp"]
DESIGNS = ["srs", "top_decile", {"name": "neyman", "n_strata": 5}, "adaptive"]
TARGETS = {"jeffreys": {"prior": "jeffreys"}, "eb05": {"prior": "eb", "alpha": 0.5},
           "eb1": {"prior": "eb", "alpha": 1.0}, "eb2": {"prior": "eb", "alpha": 2.0},
           "eb4": {"prior": "eb", "alpha": 4.0}}
RULES = {"lin": ("x_se_lin", "x_df_lin"), "satt": ("x_se_lin", "x_df_satt"),
         "pool": ("x_se_pool", "x_df_pool"), "model": ("x_se_model", "x_df_model"),
         **{t: (f"x_se_{t}", f"x_df_{t}") for t in TARGETS}}
# "model" is the default rule of the estimators (Jeffreys target); runs of this script
# record it under the name "jeffreys", other scripts under "model"


def estimators():
    out = []
    for name in ESTIMATORS:
        e = make_estimator(name)
        e.variance_targets, e.variance_rule = TARGETS, "jeffreys"
        out.append(e)
    return out
EPS = (0.05, 0.10, 0.20, 0.30)


def _pools(det: str, path: str):
    """Yield ``(corpus, detector label, tags, sub-table)`` for one feature table."""
    feat = pd.read_parquet(path)
    ys = [f"y_{o}" for o in OUTCOMES]
    if det == "roberta_hc3":
        for dom in ("reddit_eli5", "finance", "open_qa"):
            sub = feat[feat.domain == dom][["doc_id", "a", "score", "flag", "length"] + ys]
            yield f"hc3-{dom}", det, {"domain": dom, "generator": "chatgpt", "det": det}, sub
        return
    keep = ["doc_id", "domain", "source", "a", "generator", "score", "flag", "length"] + ys
    feat = feat[keep]
    for dom in ("essay", "reuter", "wp"):
        for gen in ("gpt", "claude"):
            sub = feat[(feat.domain == dom) & feat.source.isin(["human", gen])]
            yield (f"ghostbuster-{dom}", f"{det}/{gen}",
                   {"domain": dom, "generator": gen, "det": det}, sub)


def run(n_rep: int, n_jobs: int, dets=tuple(POOLS)) -> pd.DataFrame:
    frames = []
    for det in dets:
        for corpus, label, tags, sub in _pools(det, POOLS[det]):
            exp = Experiment(FeaturePool(sub), estimators(), outcomes=OUTCOMES)
            specs = [CellSpec(corpus, label, e, 900, 0.05, d, tags=tags)
                     for e in EPS for d in DESIGNS]
            frames.append(exp.run(specs, n_rep, n_jobs))
    return pd.concat(frames, ignore_index=True)


def add_intervals(df: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    d = df[df.feasible.astype(bool)].copy()
    q = 0.5 + level / 2
    for name, (se_col, df_col) in RULES.items():
        if se_col not in d:
            continue
        half = stats.t.ppf(q, d[df_col].clip(lower=1)) * d[se_col]
        d[f"cov_{name}"] = (d["point"] - half <= d["truth"]) & (d["truth"] <= d["point"] + half)
        d[f"w_{name}"] = 2 * half / d["sd_h"]
    return d


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """Per detector x design x estimator: mean cell coverage, share of cells >= 0.90 and
    mean interval width (in human-SD units) relative to the empirical oracle width
    ``2 * 1.96 * SD(point)`` of the cell."""
    cell = ["det", "design", "estimator", "eps", "domain", "generator", "outcome"]
    rules = [r for r in RULES if f"cov_{r}" in d]
    d = d.assign(err=(d["point"] - d["truth"]) / d["sd_h"])
    per = d.groupby(cell).agg(n=("point", "size"), sd_err=("err", "std"),
                              **{f"cov_{r}": (f"cov_{r}", "mean") for r in rules},
                              **{f"w_{r}": (f"w_{r}", "mean") for r in rules}).reset_index()
    per = per[per.n >= 50]
    for r in rules:
        per[f"wr_{r}"] = per[f"w_{r}"] / (2 * 1.96 * per["sd_err"])
    out = per.groupby(["det", "design", "estimator"]).agg(
        cells=("n", "size"),
        **{f"mean_{r}": (f"cov_{r}", "mean") for r in rules},
        **{f"min_{r}": (f"cov_{r}", "min") for r in rules},
        **{f"p90cells_{r}": (f"cov_{r}", lambda v: float(np.mean(v >= 0.90))) for r in rules},
        **{f"wr_{r}": (f"wr_{r}", "median") for r in rules}).reset_index()
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
