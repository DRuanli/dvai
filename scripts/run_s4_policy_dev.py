"""S4 / E5-E6 on development corpora: which selection policy and certifier to freeze.

Grid: both development corpora (Ghostbuster x TF-IDF and Fast-DetectGPT, 3 domains x 2
generators; HC3 x RoBERTa, 3 sources), the frozen audit design (adaptive, 5% of N = 900),
contamination 0.01-0.30. Candidates:

* policies - squared bias ``plugin`` or ``debiased``, with or without requiring TOST
  certification before deletion may be chosen;
* certifiers - TOST with tau = 0.1, 0.2, 0.3 SD (Student quantile, flag-PPI fallback),
  TOST with the normal quantile and no fallback, and the v1 rule.

RC-PPI++ is scored intention-to-treat: flag-PPI on the same audit when it is infeasible.
The first run of this script (9 Nov 2026) used the Hajek fallback; it is kept as
``results/S4_policy_dev_regret_hajekfb.csv``.

Outputs ``results/raw/S4_policy_dev.parquet``, ``results/S4_policy_dev_regret.csv`` (per
policy: worst, P90 and mean regret against the best single estimator of each cell, as a
ratio of standardised RMSEs and as the excess standardised RMSE) and
``results/S4_policy_dev_certification.csv`` (false-certification rate when the realised
deletion bias is at least tau, and certification rate when it is below tau / 2).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import run_s4_variance_dev as base

from d2vi.decision.certify import LegacyCertifier, TOSTCertifier
from d2vi.decision.policy import MSEPolicy
from d2vi.estimators import make_estimator
from d2vi.sampling.mixture import FeaturePool
from d2vi.sim.runner import CellSpec, Experiment

OUTCOMES = ["read", "sent", "trust", "subj"]
SINGLE = ["naive", "deletion", "hajek", "flag_ppi", "rc_ppi_pp"]
EPS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
DESIGN = "adaptive"


def experiment(sub: pd.DataFrame) -> Experiment:
    policies = [MSEPolicy(b, c) for b in ("plugin", "debiased") for c in (False, True)]
    certifiers = [TOSTCertifier(tau_sd=t) for t in (0.1, 0.2, 0.3)]
    certifiers += [TOSTCertifier(tau_sd=0.2, quantile="z", fallback=False),
                   LegacyCertifier()]
    return Experiment(FeaturePool(sub), [make_estimator(e) for e in SINGLE],
                      policies=policies, certifiers=certifiers, outcomes=OUTCOMES)


def run(n_rep: int, n_jobs: int) -> pd.DataFrame:
    frames = []
    for det, path in base.POOLS.items():
        for corpus, label, tags, sub in base._pools(det, path):
            specs = [CellSpec(corpus, label, e, 900, 0.05, DESIGN, tags=tags) for e in EPS]
            frames.append(experiment(sub).run(specs, n_rep, n_jobs))
    return pd.concat(frames, ignore_index=True)


CELL = ["det", "domain", "generator", "eps", "outcome"]


def regret(raw: pd.DataFrame) -> pd.DataFrame:
    key = CELL + ["rep"]
    est = raw[~raw.estimator.str.startswith("cert_")].copy()
    fb = est[est.estimator == "flag_ppi"].set_index(key)["point"]
    rc = est[est.estimator == "rc_ppi_pp"].set_index(key)
    itt = rc["point"].where(rc["feasible"].astype(bool), fb)  # RC-PPI++ as the policy runs it
    est = est.set_index(key)
    est.loc[est.estimator == "rc_ppi_pp", "point"] = itt.reindex(
        est[est.estimator == "rc_ppi_pp"].index).to_numpy()
    est = est.reset_index()
    est["se2"] = ((est["point"] - est["truth"]) / est["sd_h"]) ** 2
    cell = est.groupby(CELL + ["estimator"])["se2"].mean().pow(0.5).unstack("estimator")
    best = cell[SINGLE].min(axis=1)
    pols = [c for c in cell.columns if c.startswith("policy_")]
    rows = []
    for name in pols + SINGLE:
        r = cell[name] / best              # ratio regret (unstable when best ~ 0)
        d = cell[name] - best              # additive regret, human-SD units
        rows.append({"rule": name, "max_regret": r.max(), "p90_regret": r.quantile(0.9),
                     "mean_regret": r.mean(), "max_excess_srmse": d.max(),
                     "p90_excess_srmse": d.quantile(0.9), "mean_excess_srmse": d.mean(),
                     "share_within_10pct": float((r <= 1.10).mean()),
                     "pooled_srmse": float(np.sqrt((cell[name] ** 2).mean()))})
    return pd.DataFrame(rows).sort_values("max_excess_srmse")


def _cluster_upper(flags: np.ndarray, clusters: np.ndarray, n_boot: int = 2000,
                   seed: int = 1) -> float:
    """One-sided 95% upper bound of a rate by a cluster bootstrap (clusters = replications,
    whose four outcomes share one audit)."""
    keys, inv = np.unique(clusters, return_inverse=True)
    hits = np.bincount(inv, weights=flags.astype(float))
    size = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, keys.shape[0], size=(n_boot, keys.shape[0]))
    rates = hits[draws].sum(1) / size[draws].sum(1)
    return float(np.quantile(rates, 0.95))


def certification(raw: pd.DataFrame) -> pd.DataFrame:
    """False certification (|b_del| >= tau), near the boundary (tau <= |b_del| < 1.1 tau),
    worst cell, and power (|b_del| < tau / 2). Upper bounds: cluster bootstrap by
    replication (decisive), and Clopper-Pearson for the near-boundary rate ignoring the
    clustering (a check)."""
    from scipy import stats

    c = raw[raw.estimator.str.startswith("cert_")].copy()
    c["abs_b"] = c["deletion_bias"].abs()
    cell_keys = [k for k in ("det", "budget", "domain", "generator", "eps") if k in c]
    c["cluster"] = c[cell_keys + ["rep"]].astype(str).agg("|".join, 1)
    rows = []
    for name, g in c.groupby("estimator"):
        cert = g["certified"].astype(bool)
        bad = g["abs_b"] >= g["tau"]
        near = bad & (g["abs_b"] < 1.1 * g["tau"])
        good = g["abs_b"] < g["tau"] / 2
        n_bad, n_near = int(bad.sum()), int(near.sum())
        k_near = int(cert[near].sum())
        per_cell = (g[bad].assign(cert=cert[bad])
                    .groupby(cell_keys + ["outcome"])["cert"]
                    .agg(["mean", "size"]))
        per_cell = per_cell[per_cell["size"] >= 30]
        rows.append({
            "certifier": name, "n_biased": n_bad,
            "false_cert_rate": float(cert[bad].mean()) if n_bad else np.nan,
            "false_cert_upper95_cluster": _cluster_upper(cert[bad].to_numpy(),
                                                         g.loc[bad, "cluster"].to_numpy())
            if n_bad else np.nan,
            "n_near_boundary": n_near,
            "near_boundary_rate": k_near / n_near if n_near else np.nan,
            "near_boundary_upper95_cluster": _cluster_upper(cert[near].to_numpy(),
                                                            g.loc[near, "cluster"].to_numpy())
            if n_near else np.nan,
            "near_boundary_upper95_cp": float(stats.beta.ppf(0.95, k_near + 1, n_near - k_near))
            if n_near else np.nan,
            "worst_cell_rate": float(per_cell["mean"].max()) if len(per_cell) else np.nan,
            "power_small_bias": float(cert[good].mean()) if good.any() else np.nan,
            "certified_overall": float(cert.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    raw = run(args.reps, args.jobs)
    raw.to_parquet("results/raw/S4_policy_dev.parquet", index=False)
    reg, cert = regret(raw), certification(raw)
    reg.to_csv("results/S4_policy_dev_regret.csv", index=False)
    cert.to_csv("results/S4_policy_dev_certification.csv", index=False)
    pd.set_option("display.width", 250)
    print(reg.round(3).to_string(index=False))
    print(cert.round(4).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
