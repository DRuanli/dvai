"""Study 3, descriptive: who the detectors flag among real respondents, and what deleting
them does to the human mean (the second term of Lemma 1, -C_0/(1 - f), in human SD units).

    python scripts/anes_descriptives.py

Reads ``data/private/anes_features.parquet`` (evaluation records); writes only aggregates
to ``results/S5_anes_descriptives.json``. Not an endpoint; computed after the feature table
was built and before the simulation results were inspected.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DETECTORS = ("tfidf", "roberta", "fdgpt")


def main() -> None:
    f = pd.read_parquet("data/private/anes_features.parquet")
    ys = [c for c in f.columns if c.startswith("y_")]
    human = f[f["a"] == 0]
    out: dict = {"human_records": int(len(human)),
                 "human_by_mode": human["mode"].value_counts().to_dict(), "detectors": {}}
    for det in DETECTORS:
        h = human[human[f"flag_{det}"].notna()]
        d = h[f"flag_{det}"].astype(int)
        res = {"scored_humans": int(len(h)), "false_positive_rate": float(d.mean()),
               "fpr_by_mode": h.groupby("mode")[f"flag_{det}"].mean().round(4).to_dict(),
               "selection_term_sd": {}, "fpr_by_outcome_tercile": {}}
        for y in ys:
            sd = h[y].std(ddof=1)
            res["selection_term_sd"][y] = float((h.loc[d == 0, y].mean() - h[y].mean()) / sd)
            terc = pd.qcut(h[y].rank(method="first"), 3, labels=["low", "mid", "high"])
            res["fpr_by_outcome_tercile"][y] = (
                d.groupby(terc, observed=True).mean().round(4).to_dict())
        machine = f[(f["a"] > 0) & f[f"flag_{det}"].notna()]
        by_gen = machine.groupby("source")[f"flag_{det}"].mean()
        res["sensitivity_by_generator"] = by_gen.round(4).to_dict()
        by_cond = machine.groupby("condition")[f"flag_{det}"].mean()
        res["sensitivity_by_condition"] = by_cond.round(4).to_dict()
        out["detectors"][det] = res
    gap = {}
    for y in ys:
        sd = human[y].std(ddof=1)
        gap[y] = {g: float((f.loc[f["source"].eq(g), y].mean() - human[y].mean()) / sd)
                  for g in sorted(f.loc[f["a"] > 0, "source"].unique())}
    out["machine_minus_human_mean_sd"] = gap
    Path("results/S5_anes_descriptives.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float)[:4000])


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
