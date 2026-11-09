"""Build the Ghostbuster x Fast-DetectGPT feature table from the Colab scores (stage S2).

Inputs: ``data/features/ghostbuster_fdgpt_scores.parquet`` (notebook F02: raw
Fast-DetectGPT scores for the full text and the first half) and the TF-IDF feature table,
which supplies the outcomes. Output: ``data/features/ghostbuster_fdgpt.parquet``.

Threshold (v1 rule): for each target domain, the 95th percentile of human raw scores in
the two other domains. Two versions are stored:

* ``flag`` uses all documents (the revision's setting);
* ``flag_v1`` uses only the documents the v1 notebook loaded (the first 600 files per
  domain and class, ``file_rank < 600``), for reproduction.

``score`` is the raw score (only ranks and standardised values are used downstream).
Documents too short to score (fewer than 20 tokens) are dropped and counted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from d2vi import __version__
from d2vi.features.pipeline import cross_domain_threshold

SCORES = Path("data/features/ghostbuster_fdgpt_scores.parquet")
TFIDF = Path("data/features/ghostbuster_tfidf.parquet")
OUT = Path("data/features/ghostbuster_fdgpt.parquet")


def main() -> None:
    s = pd.read_parquet(SCORES)
    t = pd.read_parquet(TFIDF)
    outcome_cols = [c for c in t.columns if c.startswith("y_")]
    df = s.merge(t[["doc_id", "text_sha256"] + outcome_cols], on=["doc_id", "text_sha256"],
                 how="inner", validate="one_to_one")
    assert len(df) == len(s) == len(t), "score and TF-IDF tables do not cover the same documents"
    n0 = len(df)
    df = df.dropna(subset=["score_raw"]).reset_index(drop=True)
    dropped = n0 - len(df)

    v1 = df["file_rank"] < 600
    df["thr"] = cross_domain_threshold(df, "score_raw")
    df["thr_v1"] = cross_domain_threshold(df, "score_raw", mask=v1)
    df["thr_h1"] = cross_domain_threshold(df, "score_raw_h1")
    df["score"] = df["score_raw"]
    df["flag"] = (df["score_raw"] > df["thr"]).astype(int)
    df["flag_v1"] = (df["score_raw"] > df["thr_v1"]).astype(int)
    df["score_h1"] = df["score_raw_h1"]
    df["flag_h1"] = (df["score_raw_h1"] > df["thr_h1"]).astype(int)
    df["in_v1_sample"] = v1.to_numpy()
    df.to_parquet(OUT, index=False)

    ops = (df.groupby(["domain", "source"])
           .agg(n=("flag", "size"), flag_rate=("flag", "mean"))
           .reset_index())
    ops_v1 = (df[df["in_v1_sample"]].groupby(["domain", "source"])
              .agg(n_v1=("flag_v1", "size"), flag_rate_v1=("flag_v1", "mean"))
              .reset_index())
    ops = ops.merge(ops_v1, on=["domain", "source"])
    manifest = {
        "inputs": {
            str(SCORES): hashlib.sha256(SCORES.read_bytes()).hexdigest(),
            str(TFIDF): hashlib.sha256(TFIDF.read_bytes()).hexdigest(),
        },
        "detector": "Fast-DetectGPT analytic criterion, EleutherAI/gpt-neo-2.7B, 400 tokens",
        "threshold_rule": ("per target domain: 95th percentile of human raw scores "
                           "in the other domains"),
        "thresholds": {d: round(float(v), 4)
                       for d, v in df.groupby("domain")["thr"].first().items()},
        "thresholds_v1_sample": {d: round(float(v), 4)
                                 for d, v in df.groupby("domain")["thr_v1"].first().items()},
        "dropped_unscorable": int(dropped),
        "operating_points": ops.round(4).to_dict(orient="records"),
        "rows": int(len(df)),
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "d2vi_version": __version__,
    }
    OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.set_option("display.width", 200)
    print(ops.round(3).to_string(index=False))
    print("dropped", dropped, "| thresholds", manifest["thresholds"],
          "| v1", manifest["thresholds_v1_sample"])
    sens = ops[ops.source != "human"].pivot(index="domain", columns="source", values="flag_rate_v1")
    print("sensitivity loss gpt -> claude (v1 sample, points):",
          ((sens["gpt"] - sens["claude"]) * 100).round(1).to_dict())
    np.testing.assert_array_equal(df["a"].isin([0.0, 1.0]), True)


if __name__ == "__main__":
    main()
