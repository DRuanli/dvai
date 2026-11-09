"""S4 tuning on development corpora: audit-design target and feasibility threshold.

Grid: adaptive design with k_target in {12, 16, 20} and the top-decile split, crossed
with RC-PPI++ feasibility thresholds min_class in {4, 6} (training-fold minimum 2 and 3).
Accuracy is reported intention-to-treat: when RC-PPI++ is infeasible the two-phase Hajek
estimate is used, as the policy does. Writes ``results/S4_tuning_dev_summary.csv``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import run_s4_variance_dev as base

from d2vi.estimators import RCPPI, AuditHajek, CrossFitPropensity
from d2vi.sampling.mixture import FeaturePool
from d2vi.sim.runner import CellSpec, Experiment

DESIGNS = [{"name": "adaptive", "k_target": 12}, {"name": "adaptive", "k_target": 16},
           {"name": "adaptive", "k_target": 20}, "top_decile"]
OUTCOMES = ["read", "sent"]


def estimators():
    out = [AuditHajek()]
    for k, kf in ((6, 3), (4, 2)):
        e = RCPPI(tune=True, propensity=CrossFitPropensity(min_class=k, min_class_fold=kf))
        e.name = f"rc_ppi_pp_mc{k}"
        out.append(e)
    return out


def main(n_rep: int = 200, n_jobs: int = 2) -> None:
    t0 = time.time()
    frames = []
    for det, path in base.POOLS.items():
        feat = pd.read_parquet(path)
        for dom in ("essay", "reuter", "wp"):
            for gen in ("gpt", "claude"):
                sub = feat[(feat.domain == dom) & feat.source.isin(["human", gen])]
                cols = ["doc_id", "a", "score", "flag", "length"] + [f"y_{o}" for o in OUTCOMES]
                exp = Experiment(FeaturePool(sub[cols]), estimators(), outcomes=OUTCOMES)
                specs = [CellSpec(f"ghostbuster-{dom}", f"{det}/{gen}", e, 900, 0.05, d,
                                  tags={"domain": dom, "generator": gen, "det": det})
                         for e in base.EPS for d in DESIGNS]
                frames.append(exp.run(specs, n_rep, n_jobs))
    raw = pd.concat(frames, ignore_index=True)
    d = base.add_intervals(raw[raw.feasible.astype(bool) & (raw.estimator != "hajek")])
    key = ["det", "design", "domain", "generator", "eps", "rep", "outcome"]
    hj = raw[raw.estimator == "hajek"].set_index(key)
    rows = []
    for name in ("rc_ppi_pp_mc6", "rc_ppi_pp_mc4"):
        rc = raw[raw.estimator == name].set_index(key)
        itt = rc.point.where(rc.feasible.astype(bool), hj.point)
        e2 = ((itt - rc.truth) / rc.sd_h) ** 2
        g = pd.DataFrame({"e2": e2, "feas": rc.feasible.astype(float)}).groupby(
            ["det", "design", "eps"]).agg(itt_srmse=("e2", lambda v: float(np.sqrt(v.mean()))),
                                          feasible=("feas", "mean")).reset_index()
        cov = (d[d.estimator == name].groupby(["det", "design", "eps"])["cov_shrink_satt"]
               .mean().rename("coverage").reset_index())
        rows.append(g.merge(cov, on=["det", "design", "eps"], how="left").assign(estimator=name))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv("results/S4_tuning_dev_summary.csv", index=False)
    pd.set_option("display.width", 250)
    for col in ("itt_srmse", "feasible", "coverage"):
        print(col)
        print(out.pivot_table(index=["estimator", "det", "design"], columns="eps",
                              values=col).round(3))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
