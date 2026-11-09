"""Shared feature computation for CPU scripts and Colab notebooks."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from .outcomes import FleschReadingEase, NRCTrust, PangLeeSubjectivity, VaderSentiment, split_halves

__all__ = ["outcome_table", "cross_domain_threshold", "save_feature_table", "package_versions"]

ID_COLUMNS = ["doc_id", "corpus", "domain", "source", "a", "generator", "text_sha256"]


def package_versions(*names: str) -> dict:
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = "missing"
    return out


def outcome_table(texts: list[str], subj_dir: str | Path) -> pd.DataFrame:
    """Length, the four outcome families, and second-half outcomes for split-text."""
    second = [split_halves(t)[1] for t in texts]
    scorers = [VaderSentiment(), FleschReadingEase(), NRCTrust(), PangLeeSubjectivity(subj_dir)]
    out = pd.DataFrame({"length": [len(t.split()) for t in texts]})
    for sc in scorers:
        out[f"y_{sc.name}"] = sc.score(texts)
    for sc in scorers[:2]:
        out[f"y_{sc.name}_h2"] = sc.score(second)
    return out


def cross_domain_threshold(df: pd.DataFrame, score_col: str = "score_raw", q: float = 0.95,
                           mask: pd.Series | None = None) -> pd.Series:
    """v1 rule for zero-shot scores: for each target domain, the ``q`` quantile of human
    scores in the other domains. Returns one threshold per row."""
    base = df if mask is None else df[mask]
    thr = {}
    for dom in df["domain"].unique():
        other = base[(base["domain"] != dom) & (base["a"] == 0)][score_col].dropna()
        thr[dom] = float(np.quantile(other, q))
    return df["domain"].map(thr)


def save_feature_table(docs: pd.DataFrame, feats: pd.DataFrame, out_path: str | Path,
                       manifest: dict) -> Path:
    keep = [c for c in ID_COLUMNS + ["file_rank"] if c in docs.columns]
    out = pd.concat([docs[keep].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    manifest = dict(manifest, rows=int(len(out)),
                    sha256=hashlib.sha256(out_path.read_bytes()).hexdigest())
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_path
