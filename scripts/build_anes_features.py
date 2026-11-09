"""Stage S2: feature table of the survey study (ANES 2024 + synthetic respondents).

Usage (from the repository root, after notebooks F03 and F04)::

    python scripts/build_anes_features.py <synthetic dir> <anes_detector_scores.parquet>

where ``<synthetic dir>`` holds the ``anes_synth_*.parquet`` files of F03.

Reads the human pool ``data/private/anes2024_human.parquet``
(``scripts/build_anes_human_pool.py``). Follows the frozen protocol (section 2):

* calibration split: 30% of every stratum (source x condition), seeded by ``doc_id``;
* Fast-DetectGPT: threshold = 95th percentile of human calibration scores;
* RoBERTa OpenAI detector: zero-shot, threshold 0.5;
* TF-IDF + logistic regression: trained on the calibration humans and the calibration
  records of one generator (Qwen2.5-7B-Instruct), threshold 0.5; the other generators are
  a generator shift.

Writes ``data/private/anes_features.parquet`` (evaluation records only; not committed,
since it holds ANES respondents' thermometers) and ``results/S2_anes_operating_points.json``.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from d2vi.data.splits import calibration_split, percentile_threshold
from d2vi.features.detectors import TfidfLRDetector
from d2vi.synthetic import ANES2024

TRAIN_GENERATOR = "Qwen/Qwen2.5-7B-Instruct"
KEYS = [t.key for t in ANES2024.thermometers]


def documents(synth_dir: str) -> pd.DataFrame:
    human = pd.read_parquet("data/private/anes2024_human.parquet")
    h = pd.DataFrame({"doc_id": human["doc_id"], "a": 0.0, "source": "human",
                      "condition": "human", "mode": human["mode"], "text": human["text"],
                      **{f"y_{k}": human[f"y_{k}"] for k in KEYS}})
    files = sorted(glob.glob(f"{synth_dir}/anes_synth_*.parquet"))
    s = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    s = s[s["parse_ok"].astype(bool) & s["text"].fillna("").str.len().gt(0)]
    m = pd.DataFrame({"doc_id": s["resp_id"], "a": 1.0, "source": s["generator"],
                      "condition": s["condition"], "mode": "synthetic", "text": s["text"],
                      **{f"y_{k}": s[k] / 100.0 for k in KEYS}})
    docs = pd.concat([h, m], ignore_index=True)
    docs["length"] = docs["text"].str.split().str.len()
    if not docs["doc_id"].is_unique:
        raise ValueError("duplicate doc ids")
    return docs


def main(synth_dir: str, scores_path: str) -> None:
    docs = documents(synth_dir)
    scores = pd.read_parquet(scores_path)[["doc_id", "score_roberta", "score_fdgpt"]]
    docs = docs.merge(scores, on="doc_id", how="left", validate="one_to_one")
    if docs["score_roberta"].isna().any():
        raise ValueError("documents without detector scores")
    docs["calibration"] = calibration_split(docs, ["source", "condition"])
    cal, ev = docs[docs.calibration], docs[~docs.calibration].copy()

    thr_fd = percentile_threshold(cal.loc[cal.a == 0, "score_fdgpt"].to_numpy())
    train = cal[(cal.a == 0) | (cal.source == TRAIN_GENERATOR)]
    tfidf = TfidfLRDetector().fit(train["text"].tolist(), train["a"].to_numpy())
    ev["score_tfidf"] = tfidf.predict(ev["text"].tolist())
    ev["flag_tfidf"] = (ev["score_tfidf"] > 0.5).astype(int)
    ev["flag_roberta"] = (ev["score_roberta"] > 0.5).astype(int)
    ev["flag_fdgpt"] = np.where(ev["score_fdgpt"].isna(), np.nan,
                                (ev["score_fdgpt"] > thr_fd).astype(float))
    ev = ev.drop(columns=["text", "calibration"])
    out = Path("data/private")
    out.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(out / "anes_features.parquet", index=False)

    ops = {}
    for det in ("tfidf", "roberta", "fdgpt"):
        f = ev[f"flag_{det}"]
        ops[det] = {
            "flag_rate_by_source": ev.groupby("source")[f"flag_{det}"].mean().round(4).to_dict(),
            "flag_rate_human_by_mode": ev[ev.a == 0].groupby("mode")[f"flag_{det}"].mean()
            .round(4).to_dict(),
            "unscorable_share": float(f.isna().mean()),
        }
    summary = {"calibration_records": int(len(cal)), "evaluation_records": int(len(ev)),
               "evaluation_by_source": ev["source"].value_counts().to_dict(),
               "threshold_fdgpt": thr_fd, "tfidf_training_generator": TRAIN_GENERATOR,
               "operating_points": ops}
    Path("results/S2_anes_operating_points.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
