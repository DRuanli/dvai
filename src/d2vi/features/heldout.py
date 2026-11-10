"""Feature tables of held-out corpora, following the frozen protocol (section 2).

:func:`score_documents` needs a GPU (run in a notebook); :func:`finalize_features` applies
the calibration rule and is deterministic:

* ``"cross_domain"`` (corpora with several domains, the rule of Study 1): for each target
  domain the Fast-DetectGPT threshold is the 95th percentile of human scores in the other
  domains, and the TF-IDF detector is trained on the human documents and the documents of
  one generator in the other domains. Every record is an evaluation record.
* ``"split"`` (single-domain corpora): 30% of the groups of every domain stratum form the
  calibration set (all versions of a group together); threshold and TF-IDF training use it
  and only the remaining records are returned.

The table returned holds no text.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.splits import calibration_split, percentile_threshold
from .detectors import TfidfLRDetector

__all__ = ["score_documents", "finalize_features", "operating_points"]

RADAR_ID = "TrustSafeAI/RADAR-Vicuna-7B"


def _radar(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """RADAR (Hu et al., 2023): probability of the machine class (index 0 of its head)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(RADAR_ID)
    model = AutoModelForSequenceClassification.from_pretrained(RADAR_ID).eval().cuda()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to("cuda")
            out.append(torch.softmax(model(**enc).logits, -1)[:, 0].float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out)


def score_documents(docs: pd.DataFrame, subj_dir: str | Path, cache_path: str | Path,
                    radar: bool = True, flush_every: int = 500) -> pd.DataFrame:
    """Outcomes and raw detector scores for ``docs`` (columns ``doc_id``, ``text``).

    Fast-DetectGPT scores are cached in ``cache_path`` (CSV) every ``flush_every``
    documents, so an interrupted run resumes. RADAR is a secondary detector: if it cannot
    be loaded its column is NaN and the run continues.
    """
    from .neural import FastDetectGPT, RobertaOpenAIDetector
    from .pipeline import outcome_table

    texts = docs["text"].tolist()
    feats = outcome_table(texts, subj_dir)
    feats.insert(0, "doc_id", docs["doc_id"].to_numpy())
    feats["score_roberta"] = RobertaOpenAIDetector().predict(texts)
    feats["score_radar"] = np.nan
    if radar:
        try:
            feats["score_radar"] = _radar(texts)
        except Exception as err:  # noqa: BLE001 - secondary detector must not stop the run
            print("RADAR skipped:", repr(err)[:300], flush=True)
    done = (pd.read_csv(cache_path, dtype={"doc_id": str}) if os.path.exists(cache_path)
            else pd.DataFrame(columns=["doc_id", "score_fdgpt"]))
    todo = docs[~docs["doc_id"].isin(done["doc_id"])]
    print(f"Fast-DetectGPT: cached {len(done)}, to score {len(todo)}", flush=True)
    if len(todo):
        fd = FastDetectGPT()
        for s in range(0, len(todo), flush_every):
            part = todo.iloc[s:s + flush_every]
            new = pd.DataFrame({"doc_id": part["doc_id"].to_numpy(),
                                "score_fdgpt": [fd.score_one(t) for t in part["text"]]})
            done = pd.concat([done, new], ignore_index=True)
            done.to_csv(cache_path, index=False)
            print(f"  {len(done)} scored", flush=True)
    return feats.merge(done.drop_duplicates("doc_id"), on="doc_id", how="left")


def _flags(part: pd.DataFrame, thr_fd: float, tfidf: TfidfLRDetector) -> pd.DataFrame:
    part = part.copy()
    part["thr_fdgpt"] = thr_fd
    part["flag_fdgpt"] = np.where(part["score_fdgpt"].isna(), np.nan,
                                  (part["score_fdgpt"] > thr_fd).astype(float))
    part["score_tfidf"] = tfidf.predict(part["text"].tolist())
    part["flag_tfidf"] = (part["score_tfidf"] > 0.5).astype(int)
    part["flag_roberta"] = (part["score_roberta"] > 0.5).astype(int)
    return part


def finalize_features(table: pd.DataFrame, calibration: str, train_mask: pd.Series,
                      share: float = 0.30, strata: tuple[str, ...] = ("domain",)
                      ) -> tuple[pd.DataFrame, dict]:
    """Thresholds, TF-IDF detector and flags; returns the evaluation table and a summary.

    ``table``: documents with ``doc_id``, ``domain``, ``a``, ``text``, outcomes and raw
    scores (and ``group`` for ``calibration="split"``). ``train_mask`` marks the machine
    documents of the generator the TF-IDF detector may be trained on. ``strata``: columns
    (constant within a group) by which the calibration split is stratified.
    """
    human = table["a"].eq(0)
    info: dict = {"calibration": calibration, "rows": int(len(table))}
    parts = []
    if calibration == "cross_domain":
        if table["domain"].nunique() < 2:
            raise ValueError("cross-domain calibration needs at least two domains")
        info["thresholds_fdgpt"], info["tfidf_training"] = {}, {}
        for dom in sorted(table["domain"].unique()):
            cal = table["domain"].ne(dom)
            thr = percentile_threshold(table.loc[cal & human, "score_fdgpt"].to_numpy())
            train = table[cal & (human | train_mask)]
            tfidf = TfidfLRDetector().fit(train["text"].tolist(), train["a"].to_numpy())
            parts.append(_flags(table[~cal], thr, tfidf))
            info["thresholds_fdgpt"][dom] = thr
            info["tfidf_training"][dom] = {"human": int((train.a == 0).sum()),
                                           "machine": int((train.a > 0).sum())}
    elif calibration == "split":
        groups = table.drop_duplicates("group")[["group", *strata]].rename(
            columns={"group": "doc_id"})
        cal_groups = set(groups.loc[calibration_split(groups, list(strata), share), "doc_id"])
        cal = table["group"].isin(cal_groups)
        thr = percentile_threshold(table.loc[cal & human, "score_fdgpt"].to_numpy())
        train = table[cal & (human | train_mask)]
        tfidf = TfidfLRDetector().fit(train["text"].tolist(), train["a"].to_numpy())
        parts.append(_flags(table[~cal], thr, tfidf))
        info |= {"threshold_fdgpt": thr, "calibration_groups": len(cal_groups),
                 "calibration_rows": int(cal.sum()),
                 "tfidf_training": {"human": int((train.a == 0).sum()),
                                    "machine": int((train.a > 0).sum())}}
    else:
        raise ValueError(f"unknown calibration rule {calibration!r}")
    out = pd.concat(parts).drop(columns=["text"]).sort_values("doc_id").reset_index(drop=True)
    info["evaluation_rows"] = int(len(out))
    info["unscorable_fdgpt"] = float(out["score_fdgpt"].isna().mean())
    return out, info


def operating_points(features: pd.DataFrame,
                     by: tuple[str, ...] = ("domain", "source", "version")) -> pd.DataFrame:
    """Flag rate of every detector per group (sensitivity for machine groups, false-positive
    rate for the human group)."""
    flags = [c for c in features.columns if c.startswith("flag_")]
    g = features.groupby(list(by))
    out = g[flags].mean()
    out.insert(0, "n", g.size())
    return out.reset_index()
