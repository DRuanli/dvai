"""S4, second round (after the G4 review): audit design and cross-fitting folds.

The first round favoured the adaptive design, but that design weighted every record by
N_h / m_h although its second stage depends on the pilot labels, which biases the
estimators (finding F15). Designs now use stage-wise weights; this script compares them
again on the regime of the primary endpoints:

* both development corpora: Ghostbuster with TF-IDF and Fast-DetectGPT (3 domains x 2
  generators), HC3 with RoBERTa (3 sources) - 15 pools;
* N = 900, iid mixture, contamination 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.45, audit
  budget 2%, 5%, 10%;
* designs: SRS, proportional (10 strata), top-decile split, two-stage Neyman (5 strata),
  adaptive constrained Neyman with k_target 0, 8, 16, and k_target 16 with four
  second-stage records per stratum;
* estimators: naive, deletion, Hajek, flag-PPI, RC-PPI++ with 2 and with 5 folds.

Selection rule, fixed before this run: RC-PPI++ is scored intention-to-treat (flag-PPI on
the same audit when RC-PPI++ is infeasible). Among designs whose interval for that rule
has mean coverage >= 0.93 and >= 90% of cells >= 0.90, choose the lowest mean cell
standardised RMSE; the worst cell is reported. Folds: the same rule between K = 2 and 5.

Writes one raw file per pool to ``results/raw/S4b/`` (resumable) and
``results/S4b_design_dev_cells.csv`` and ``results/S4b_design_dev_summary.csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_s4_variance_dev as base

from d2vi.estimators import RCPPI, CrossFitPropensity, make_estimator
from d2vi.sampling.mixture import FeaturePool
from d2vi.sim.runner import CellSpec, Experiment

OUTCOMES = ["read", "sent", "trust", "subj"]
EPS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.45)
BUDGETS = (0.02, 0.05, 0.10)
DESIGNS = {"srs": "srs", "prop10": {"name": "proportional", "n_strata": 10},
           "top_decile": "top_decile", "neyman5": {"name": "neyman", "n_strata": 5},
           "adaptive_k0": {"name": "adaptive", "k_target": 0},
           "adaptive_k8": {"name": "adaptive", "k_target": 8},
           "adaptive_k16": {"name": "adaptive", "k_target": 16},
           "adaptive_k16_s4": {"name": "adaptive", "k_target": 16, "min_second": 4}}
CELL = ["det", "design_label", "budget", "eps", "domain", "generator", "outcome"]


def estimators():
    out = [make_estimator(e) for e in ("naive", "deletion", "hajek", "flag_ppi", "rc_ppi_pp")]
    k5 = RCPPI(tune=True, propensity=CrossFitPropensity(n_folds=5))
    k5.name = "rc_ppi_pp_k5"
    return out + [k5]


RAW = Path("results/raw/S4b")
KEEP = ["det", "design_label", "budget", "eps", "domain", "generator", "outcome", "rep",
        "estimator", "point", "ci_low", "ci_high", "feasible", "truth", "sd_h", "m_machine",
        "deletion_bias"]


def run(n_rep: int, n_jobs: int) -> None:
    """One parquet file per pool (resumable: finished pools are skipped)."""
    RAW.mkdir(parents=True, exist_ok=True)
    for det, path in base.POOLS.items():
        for corpus, label, tags, sub in base._pools(det, path):
            out = RAW / f"{det}__{corpus}__{label.replace('/', '-')}.parquet"
            if out.exists():
                continue
            exp = Experiment(FeaturePool(sub), estimators(), outcomes=OUTCOMES)
            specs = [CellSpec(corpus, label, e, 900, b, spec,
                              tags={**tags, "design_label": name})
                     for b in BUDGETS for e in EPS for name, spec in DESIGNS.items()]
            frames = [exp.run_cell(sp, n_rep, n_jobs)[KEEP] for sp in specs]
            pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
            print("done", out.name, flush=True)


def itt(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    """RC-PPI++ variant ``name`` with the flag-PPI fallback, one row per replication."""
    key = CELL + ["rep"]
    rc = raw[raw.estimator == name].set_index(key)
    fb = raw[raw.estimator == "flag_ppi"].set_index(key).reindex(rc.index)
    ok = rc["feasible"].astype(bool)
    out = pd.DataFrame({
        "point": rc["point"].where(ok, fb["point"]),
        "lo": rc["ci_low"].where(ok, fb["ci_low"]), "hi": rc["ci_high"].where(ok, fb["ci_high"]),
        "truth": rc["truth"], "sd_h": rc["sd_h"], "feasible": ok.astype(float),
        "m_machine": rc["m_machine"]})
    out["e2"] = ((out.point - out.truth) / out.sd_h) ** 2
    out["cover"] = ((out.lo <= out.truth) & (out.truth <= out.hi)).astype(float)
    return out


def cell_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Per cell and RC-PPI++ variant: ITT standardised RMSE, coverage, feasibility."""
    rows = []
    for name in ("rc_ppi_pp", "rc_ppi_pp_k5"):
        t = itt(raw, name)
        cell = t.groupby(CELL).agg(srmse=("e2", lambda v: float(np.sqrt(v.mean()))),
                                   cover=("cover", "mean"), feasible=("feasible", "mean"),
                                   m_ai=("m_machine", "mean")).reset_index()
        cell["estimator"] = name
        rows.append(cell)
    return pd.concat(rows, ignore_index=True)


def cells() -> pd.DataFrame:
    """Cell table over all pools, built file by file to bound memory."""
    return pd.concat([cell_table(pd.read_parquet(f)) for f in sorted(RAW.glob("*.parquet"))],
                     ignore_index=True)


def summarize(cell: pd.DataFrame) -> pd.DataFrame:
    g = cell.groupby(["estimator", "design_label", "budget"])
    return g.agg(mean_srmse=("srmse", "mean"), max_srmse=("srmse", "max"),
                 mean_cover=("cover", "mean"),
                 cells_ge90=("cover", lambda v: float(np.mean(v >= 0.90))),
                 min_cover=("cover", "min"), feasible=("feasible", "mean"),
                 m_ai=("m_ai", "mean")).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    run(args.reps, args.jobs)
    cell = cells()
    cell.to_csv("results/S4b_design_dev_cells.csv", index=False)
    s = summarize(cell)
    s.to_csv("results/S4b_design_dev_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(4).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
