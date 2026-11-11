"""Detector operating points on the evaluation records of every held-out corpus.

    python scripts/heldout_operating_points.py

For each corpus configuration in ``configs/s5`` and each of its detectors: the share of
flagged records per domain among humans (false-positive rate) and per domain x machine
group (sensitivity), with record counts. Writes ``results/S5_heldout_operating_points.csv``.
Descriptive; computed from the feature tables, not from the simulations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_s5  # noqa: E402

CORPORA = ("anes", "raid", "beemo", "m4", "hc3zh", "daigt")


def main() -> None:
    rows = []
    for c in CORPORA:
        cfg = yaml.safe_load(Path(f"configs/s5/{c}.yaml").read_text())
        feat = pd.read_parquet(cfg["features"])
        cfg = run_s5.resolve(cfg, feat)
        domains = cfg.get("domains") or [None]
        for det, cols in cfg["detectors"].items():
            for dom in domains:
                part = feat if dom is None else feat[feat["domain"].eq(dom)]
                part = part[part[cols["flag"]].notna()]
                human = part[run_s5._match(part, cfg["human"])]
                rows.append({"corpus": cfg["corpus"], "detector": det, "domain": dom or "all",
                             "group": "human", "records": len(human),
                             "flagged": float(human[cols["flag"]].mean())})
                for g, spec in cfg["machine_groups"].items():
                    m = part[run_s5._match(part, spec)]
                    if len(m):
                        rows.append({"corpus": cfg["corpus"], "detector": det,
                                     "domain": dom or "all", "group": g, "records": len(m),
                                     "flagged": float(m[cols["flag"]].mean())})
    out = pd.DataFrame(rows)
    out.to_csv("results/S5_heldout_operating_points.csv", index=False)
    h = out[out.group.eq("human")]
    m = out[~out.group.eq("human")]
    summary = pd.DataFrame({
        "fpr_min": h.groupby(["corpus", "detector"])["flagged"].min(),
        "fpr_max": h.groupby(["corpus", "detector"])["flagged"].max(),
        "sens_min": m.groupby(["corpus", "detector"])["flagged"].min(),
        "sens_max": m.groupby(["corpus", "detector"])["flagged"].max()})
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
