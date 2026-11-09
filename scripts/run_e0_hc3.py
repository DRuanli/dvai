"""E0 (stage S3) for Study 2: reproduce Table 3 of the first submission on HC3.

v1 settings: 3 sources (reddit_eli5, finance, open_qa), ChatGPT answers, N = 900,
contamination 0.05-0.30, 10 seeds, audit 5%, RoBERTa OpenAI detector at 0.5.
Writes ``results/raw/E0_hc3_legacy10.parquet`` and ``results/E0_table3_comparison.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from d2vi.estimators.legacy import legacy_replication

DOMAINS = ("reddit_eli5", "finance", "open_qa")
EPS = (0.05, 0.10, 0.20, 0.30)
OUTCOMES = ["sent", "read"]


def arrays(df):
    return {"score": df["score"].to_numpy(), "flag": df["flag"].to_numpy(),
            "length": df["length"].to_numpy(),
            **{f"y_{o}": df[f"y_{o}"].to_numpy() for o in OUTCOMES}}


def main(n_seeds: int = 10, n_jobs: int = 2) -> None:
    feat = pd.read_parquet("data/features/hc3_roberta.parquet")
    jobs = []
    for dom in DOMAINS:
        h = arrays(feat[(feat.domain == dom) & (feat.a == 0)])
        m = arrays(feat[(feat.domain == dom) & (feat.a == 1)])
        for eps in EPS:
            for seed in range(n_seeds):
                jobs.append((dom, h, m, eps, seed))

    def one(dom, h, m, eps, seed):
        rows = legacy_replication(h, m, 900, eps, seed, 0.05, OUTCOMES)
        return [dict(r, domain=dom) for r in rows]

    out = Parallel(n_jobs=n_jobs)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame([r for chunk in out for r in chunk])
    df.to_parquet(f"results/raw/E0_hc3_legacy{n_seeds}.parquet", index=False)
    rows = []
    for (o, eps), g in df.groupby(["outcome", "eps"]):
        row = {"outcome": o, "eps": eps}
        for c in ("naive", "deletion", "ppi", "rc_strat", "rc_strat_raw"):
            row[f"l{n_seeds}_{c}"] = float(np.sqrt(np.nanmean((g[c] - g["truth"]) ** 2)))
        row[f"l{n_seeds}_fallback_share"] = float(g["rc_strat_raw"].isna().mean())
        rows.append(row)
    ours = pd.DataFrame(rows)
    paper = pd.read_csv("docs/paper_v1_tables.csv")
    paper = paper[paper.table == "T3"].drop(columns=["table", "domain", "generator"])
    paper = paper.rename(columns={c: f"paper_{c}" for c in
                                  ("naive", "deletion", "flag_ppi", "strat_rc_ppi")})
    comp = paper.merge(ours, on=["outcome", "eps"])
    comp.to_csv(f"results/E0_table3_comparison_l{n_seeds}.csv", index=False)
    pd.set_option("display.width", 250)
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
