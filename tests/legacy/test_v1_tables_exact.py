"""Gate G3 as a regression test: the v1 code path reproduces Table 2 of the first
submission from the published Ghostbuster feature table."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ppi_py")

from d2vi.estimators.legacy import legacy_replication  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data/features/ghostbuster_tfidf.parquet"
PAPER = ROOT / "docs/paper_v1_tables.csv"


@pytest.mark.skipif(not FEATURES.exists(), reason="feature table not present")
def test_table2_is_reproduced_to_rounding():
    feat = pd.read_parquet(FEATURES)

    def arrays(df):
        return {c: df[c].to_numpy() for c in ["score", "flag", "length", "y_read", "y_sent"]}

    rows = []
    for dom in ("essay", "reuter", "wp"):
        h = arrays(feat[(feat.domain == dom) & (feat.source == "human")])
        for gen in ("gpt", "claude"):
            m = arrays(feat[(feat.domain == dom) & (feat.source == gen)])
            for eps in (0.05, 0.10, 0.20, 0.30):
                for seed in range(8):
                    rows += legacy_replication(h, m, 900, eps, seed, 0.05, ["read", "sent"])
    df = pd.DataFrame(rows)
    paper = pd.read_csv(PAPER)
    paper = paper[paper.table == "T2"]
    for _, p in paper.iterrows():
        g = df[(df.outcome == p.outcome) & np.isclose(df.eps, p.eps)]
        for ours, theirs in (("naive", "naive"), ("deletion", "deletion"), ("ppi", "flag_ppi"),
                             ("rc_strat", "strat_rc_ppi")):
            rmse = float(np.sqrt(np.nanmean((g[ours] - g["truth"]) ** 2)))
            assert rmse == pytest.approx(p[theirs], abs=6e-5), (p.outcome, p.eps, ours)


HC3 = ROOT / "data/features/hc3_roberta.parquet"


@pytest.mark.skipif(not HC3.exists(), reason="HC3 feature table not present")
def test_table3_is_reproduced_to_rounding():
    feat = pd.read_parquet(HC3)

    def arrays(df):
        return {c: df[c].to_numpy() for c in ["score", "flag", "length", "y_read", "y_sent"]}

    rows = []
    for dom in ("reddit_eli5", "finance", "open_qa"):
        h = arrays(feat[(feat.domain == dom) & (feat.a == 0)])
        m = arrays(feat[(feat.domain == dom) & (feat.a == 1)])
        for eps in (0.05, 0.10, 0.20, 0.30):
            for seed in range(10):
                rows += legacy_replication(h, m, 900, eps, seed, 0.05, ["sent", "read"])
    df = pd.DataFrame(rows)
    paper = pd.read_csv(PAPER)
    for _, p in paper[paper.table == "T3"].iterrows():
        g = df[(df.outcome == p.outcome) & np.isclose(df.eps, p.eps)]
        for ours, theirs in (("naive", "naive"), ("deletion", "deletion"), ("ppi", "flag_ppi"),
                             ("rc_strat", "strat_rc_ppi")):
            rmse = float(np.sqrt(np.nanmean((g[ours] - g["truth"]) ** 2)))
            assert rmse == pytest.approx(p[theirs], abs=6e-5), (p.outcome, p.eps, ours)
