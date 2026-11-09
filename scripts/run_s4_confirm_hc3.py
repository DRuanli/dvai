"""S4 confirmation on the second development corpus (HC3 x RoBERTa detector).

Checks whether the interim defaults chosen on Ghostbuster (adaptive design with
k_target = 16, feasibility 4/2, shrink + Satterthwaite intervals) hold on HC3, against the
SRS and top-decile designs. Writes ``results/raw/S4_confirm_hc3.parquet`` and
``results/S4_confirm_hc3_summary.csv``.
"""

from __future__ import annotations

import time

import pandas as pd

from d2vi.estimators import make_estimator
from d2vi.sampling.mixture import FeaturePool
from d2vi.sim.runner import CellSpec, Experiment

OUTCOMES = ["read", "sent", "trust", "subj"]
DESIGNS = ["srs", "top_decile", "adaptive"]
EPS = (0.05, 0.10, 0.20, 0.30)
EST = ["naive", "deletion", "hajek", "flag_ppi", "rc_ppi_pp"]


def main(n_rep: int = 300, n_jobs: int = 2) -> None:
    t0 = time.time()
    feat = pd.read_parquet("data/features/hc3_roberta.parquet")
    frames = []
    for dom in ("reddit_eli5", "finance", "open_qa"):
        sub = feat[feat.domain == dom]
        cols = ["doc_id", "a", "score", "flag", "length"] + [f"y_{o}" for o in OUTCOMES]
        exp = Experiment(FeaturePool(sub[cols]), [make_estimator(e) for e in EST],
                         outcomes=OUTCOMES)
        specs = [CellSpec(f"hc3-{dom}", "roberta_openai", e, 900, 0.05, d,
                          tags={"domain": dom}) for e in EPS for d in DESIGNS]
        frames.append(exp.run(specs, n_rep, n_jobs))
    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet("results/raw/S4_confirm_hc3.parquet", index=False)

    key = ["design", "domain", "eps", "rep", "outcome"]
    by = {e: raw[raw.estimator == e].set_index(key) for e in EST}
    rc, hj = by["rc_ppi_pp"], by["hajek"]
    itt = rc.point.where(rc.feasible.astype(bool), hj.point)
    rows = []
    for name, pts in [("rc_ppi_pp_itt", itt)] + [(e, by[e].point) for e in EST]:
        e2 = ((pts - rc.truth) / rc.sd_h) ** 2
        rows.append(e2.groupby(["design", "eps"]).mean().pow(0.5).rename(name))
    acc = pd.concat(rows, axis=1).reset_index()
    cov = []
    for e in ("hajek", "flag_ppi", "rc_ppi_pp"):
        d = by[e][by[e].feasible.astype(bool)]
        c = ((d.ci_low <= d.truth) & (d.truth <= d.ci_high)).groupby(["design", "eps"]).mean()
        cov.append(c.rename(f"cov_{e}"))
    feas = rc.feasible.astype(float).groupby(["design", "eps"]).mean().rename("feasible_rc")
    out = acc.merge(pd.concat(cov + [feas], axis=1).reset_index(), on=["design", "eps"])
    out.to_csv("results/S4_confirm_hc3_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
