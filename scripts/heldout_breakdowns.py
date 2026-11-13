"""Breakdowns of the held-out results quoted in Sections 5-6 of the paper.

    python scripts/heldout_breakdowns.py results/compact/S5 --out results/S5_breakdowns.json

Everything is computed from the compact tables (``results/compact/S5``) and the detector
operating points, with the same definitions as the endpoints: an error ratio is a ratio of
mean squared errors summed over cells (as PE1), coverage and feasibility are means over
cells (the pooled ratio of PE1). The intention-to-treat rule (ITT) is ``rc_ppi_pp`` with its flag-PPI fallback.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ATTACKS = ("paraphrase", "synonym", "homoglyph")
OPS = "results/S5_heldout_operating_points.csv"


def load(root: Path, grid: str, kind: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / grid / f"*.{kind}.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def parse_pool(pool: str) -> dict:
    """``corpus__label__detector`` with label ``[domain-]generator[-attack]``."""
    corpus, label, det = pool.split("__")
    out = {"detector": det, "domain": None, "generator": label, "attack": "none"}
    if corpus in ("raid", "m4", "hc3zh"):
        out["domain"], rest = label.split("-", 1)
        for a in ATTACKS:
            if rest.endswith("-" + a):
                rest, out["attack"] = rest[: -len(a) - 1], a
        out["generator"] = rest
    return out


def ratio(d: pd.DataFrame, a: str, b: str) -> float:
    return float(d[f"mse_{a}"].sum() / d[f"mse_{b}"].sum())


def summary(d: pd.DataFrame) -> dict:
    return {"cells": int(len(d)), "itt_vs_flag_ppi": ratio(d, "rc_ppi_pp", "flag_ppi"),
            "deletion_vs_itt": ratio(d, "deletion", "rc_ppi_pp"),
            "policy_vs_deletion": ratio(d, "policy_debiased", "deletion"),
            "coverage_mean": float(d.cover_itt.mean()), "coverage_min": float(d.cover_itt.min()),
            "feasible": float(d.feasible.mean())}


def by(d: pd.DataFrame, keys: list[str]) -> dict:
    return {"|".join(map(str, k if isinstance(k, tuple) else (k,))): summary(g)
            for k, g in d.groupby(keys)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="results/S5_breakdowns.json")
    args = ap.parse_args()
    root = Path(args.root)
    cells = load(root, "primary", "cells")
    cells = pd.concat([cells, pd.DataFrame([parse_pool(p) for p in cells.pool])], axis=1)
    cert = load(root, "primary", "certcells")
    cert = pd.concat([cert, pd.DataFrame([parse_pool(p) for p in cert.pool])], axis=1)
    ops = pd.read_csv(OPS)
    res: dict = {}

    # coverage below 0.90 (Section 5.3)
    low = cells[cells.cover_itt < 0.90]
    res["low_coverage_cells"] = {
        "n": int(len(low)), "of": int(len(cells)),
        "by_corpus": low.corpus.value_counts().to_dict(),
        "by_detector": low.detector.value_counts().to_dict(),
        "by_budget": {str(k): int(v) for k, v in low.budget.value_counts().items()},
        "at_eps_le_0.02": int((low.eps <= 0.02).sum()),
        "feasible_over_half": int((low.feasible > 0.5).sum()),
        "min": float(cells.cover_itt.min())}

    # PE1 pools with ratio above one (Section 5.2)
    per = cells.groupby(["corpus", "pool"]).agg(itt=("mse_rc_ppi_pp", "sum"), fl=("mse_flag_ppi", "sum"),
                                                pp_rc=("mse_pp_rc", "sum"), pp_fb=("mse_pp_fb", "sum"),
                                                feasible=("feasible", "mean"))
    per["r"] = per.itt / per.fl
    per["r_pp"] = per.pp_rc / per.pp_fb
    above = per[per.r > 1]
    res["pe1_pools_above_one"] = {
        "n": int(len(above)), "of": int(len(per)),
        "per_protocol_also_above": int((above.r_pp > 1).sum()),
        "feasible_mean_above": float(above.feasible.mean()), "feasible_mean_all": float(per.feasible.mean()),
        "max": float(per.r.max()), "n_above_1.3": int((per.r > 1.3).sum()),
        "pools": {p: round(float(v), 3) for (_, p), v in above.r.sort_values(ascending=False).items()}}

    # regime map: which rule has the lowest MSE (Section 5.3)
    rules = ["naive", "deletion", "flag_ppi", "rc_ppi_pp"]
    best = np.array(rules)[cells[[f"mse_{r}" for r in rules]].to_numpy().argmin(axis=1)]
    tab = pd.crosstab([cells.budget, cells.eps], best, normalize="index").round(3)
    res["regime_winner_share"] = {f"{b}|{e}": row.to_dict() for (b, e), row in tab.iterrows()}

    # feasibility by contamination x budget (Section 6.2)
    res["feasibility"] = {f"{b}|{e}": float(v) for (b, e), v in
                          cells.groupby(["budget", "eps"]).feasible.mean().items()}

    # RAID (Section 5.7)
    raid = cells[cells.corpus == "raid"]
    res["raid_by_attack_detector"] = by(raid, ["attack", "detector"])
    res["raid_by_attack_detector_budget"] = by(raid, ["attack", "detector", "budget"])
    res["raid_by_domain_detector"] = by(raid, ["domain", "detector"])
    rc = cert[cert.corpus == "raid"].groupby(["domain", "budget"])[["n_small", "k_small", "n_bad", "k_bad"]].sum()
    res["raid_certification_by_domain_budget"] = {
        f"{d}|{b}": {"certified_small_bias": float(r.k_small / r.n_small),
                     "false_certification": float(r.k_bad / max(r.n_bad, 1))}
        for (d, b), r in rc.iterrows()}

    # M4 and HC3-Chinese (Section 5.7)
    res["m4_by_language_detector"] = by(cells[cells.corpus == "m4"], ["domain", "detector"])
    res["hc3zh_by_source_detector"] = by(cells[cells.corpus == "hc3zh"], ["domain", "detector"])
    res["operating_points"] = {f"{r.corpus}|{r.domain}|{r.detector}|{r.group}": float(r.flagged)
                               for r in ops.itertuples()}

    # PE3 cells (Section 5.5)
    c = cert[cert.n_bad >= 30].assign(rate=lambda x: x.k_bad / x.n_bad)
    worst = c.nlargest(5, "rate")[["pool", "budget", "eps", "outcome", "n_bad", "k_bad", "rate"]]
    big = c[c.n_bad >= 300].nlargest(3, "rate")[["pool", "budget", "eps", "outcome", "n_bad", "k_bad", "rate"]]
    res["pe3_cells"] = {"cells_with_30_bad": int(len(c)), "above_0.075": int((c.rate > 0.075).sum()),
                        "worst": worst.to_dict("records"), "worst_with_300_bad": big.to_dict("records")}

    # large-N exclusions (Section 5.8)
    excl = {}
    for m in glob.glob(str(root / "large_n" / "*.manifest.json")):
        man = json.loads(Path(m).read_text())
        for name, p in man["pools"].items():
            if p.get("cells_skipped"):
                excl[name] = {"cells_run": p.get("cells_run"),
                              "skipped": [(s["eps"], s["reason"][:80]) for s in p["cells_skipped"]]}
    res["large_n_skips"] = {"pools_with_skips": len(excl),
                            "pools_without_cells": sorted(k for k, v in excl.items() if not v["cells_run"]),
                            "pools_partly_run": len([k for k, v in excl.items() if v["cells_run"]])}

    # coverage by detector and by outcome (Section 5.3)
    res["coverage_by_detector"] = {k: {"mean": float(g.cover_itt.mean()), "min": float(g.cover_itt.min()),
                                       "cells": int(len(g))} for k, g in cells.groupby("detector")}
    res["coverage_by_outcome"] = {k: {"mean": float(g.cover_itt.mean()), "min": float(g.cover_itt.min()),
                                      "cells": int(len(g))} for k, g in cells.groupby("outcome")}

    # lowest cells of the stratified audit designs (Section 5.4)
    des = pd.concat([pd.read_parquet(f) for f in glob.glob(str(root / "designs" / "*.sec.parquet"))])
    des["design"] = des["design"].astype(str)
    strat = des[des.design.str.startswith(("proportional", "neyman"))]
    res["designs_lowest_stratified"] = strat.nsmallest(6, "cover")[
        ["pool", "design", "eps", "outcome", "cover", "feasible"]].to_dict("records")

    # designs by contamination rate (Section 5.4): sRMSE of the ITT rule
    des["short"] = des.design.str.split("_").str[0]
    des["srmse"] = np.sqrt(des.itt_mse)
    tab = des.groupby(["eps", "short"]).agg(srmse=("srmse", "mean"), cover=("cover", "mean"))
    res["designs_by_contamination"] = {f"{e}|{d}": {"srmse": float(r.srmse), "cover": float(r.cover)}
                                       for (e, d), r in tab.iterrows()}

    Path(args.out).write_text(json.dumps(res, indent=1, default=float))
    print("written", args.out)


if __name__ == "__main__":
    main()
