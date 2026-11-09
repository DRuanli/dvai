"""S4 exploratory (development data only): does propensity-fit instability explain the
remaining undercoverage of RC-PPI++ under the top-decile audit?

Cells: Ghostbuster x Fast-DetectGPT, top-decile audit of 45, N = 900, contamination
0.10-0.30, essay/reuter/wp x GPT, readability. Variants of RC-PPI++:

* ``k2``      - 2 folds, C = 1 (current);
* ``k5``      - 5 folds, C = 1 (larger training folds, more stable fits);
* ``k2_c01``  - 2 folds, C = 0.1 (stronger regularisation);
* ``k5_c01``  - 5 folds, C = 0.1;
* ``k2_raowu``- current estimator with a Rao-Wu bootstrap percentile interval (refits
  the propensity in every replicate).

Writes ``results/S4_explore_propensity_variance.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from d2vi.estimators import RCPPI, CrossFitPropensity, EstimationContext
from d2vi.inference.bootstrap import RaoWuBootstrap
from d2vi.sampling.audit import PerfectVerifier, make_design
from d2vi.sampling.mixture import FeaturePool, IIDMixture

VARIANTS = {"k2": (2, 1.0), "k5": (5, 1.0), "k2_c01": (2, 0.1), "k5_c01": (5, 0.1)}


def one(pool, eps, rep, n_boot):
    rng = np.random.default_rng([20261109, int(eps * 100), rep])
    rs = IIDMixture().sample(pool, 900, eps, rng)
    audit = make_design("top_decile").draw(rs.observed, 45, rng, PerfectVerifier(rs.a))
    truth = rs.human_mean("read")
    rows = []
    for name, (k, c) in VARIANTS.items():
        est = RCPPI(tune=True, propensity=CrossFitPropensity(n_folds=k, c=c))
        ctx = EstimationContext(rs.observed, audit, np.random.default_rng(rep))
        res = est.estimate(ctx, "read")
        if not res.feasible:
            continue
        q = stats.t.ppf(0.975, max(res.extras["df_model"], 1))
        half = q * res.extras["se_model"]
        rows.append(dict(variant=name, err=res.point - truth,
                         cover=abs(res.point - truth) <= half, width=2 * half))
    if n_boot:
        est = RCPPI(tune=True)
        ctx = EstimationContext(rs.observed, audit, np.random.default_rng(rep))
        boot = RaoWuBootstrap(n_boot=n_boot)
        res = boot.interval(est, ctx, "read", np.random.default_rng(rep + 1))
        if res.feasible and np.isfinite(res.ci_low):
            rows.append(dict(variant="k2_raowu", err=res.point - truth,
                             cover=res.ci_low <= truth <= res.ci_high,
                             width=res.ci_high - res.ci_low))
    return rows


def main(n_rep: int = 200, n_boot: int = 150, n_jobs: int = 2) -> None:
    feat = pd.read_parquet("data/features/ghostbuster_fdgpt.parquet")
    out = []
    for dom in ("essay", "reuter", "wp"):
        sub = feat[(feat.domain == dom) & feat.source.isin(["human", "gpt"])]
        pool = FeaturePool(sub[["doc_id", "a", "score", "flag", "length", "y_read"]])
        for eps in (0.10, 0.20, 0.30):
            chunks = Parallel(n_jobs=n_jobs)(delayed(one)(pool, eps, r, n_boot)
                                             for r in range(n_rep))
            for rows in chunks:
                for row in rows:
                    out.append(dict(row, domain=dom, eps=eps))
    df = pd.DataFrame(out)
    s = df.groupby(["variant", "eps"]).agg(coverage=("cover", "mean"),
                                           rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                                           width=("width", "mean"), n=("cover", "size"))
    s.reset_index().to_csv("results/S4_explore_propensity_variance.csv", index=False)
    print(s.round(4).unstack("eps").to_string())


if __name__ == "__main__":
    main()
