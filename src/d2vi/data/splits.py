"""Calibration / evaluation splits for held-out corpora (frozen protocol, section 1a).

Detectors are trained, and zero-shot thresholds calibrated, only on calibration records;
evaluation pools are drawn only from the remaining records. For corpora with a single
domain the calibration set is a seeded share of every stratum (class x generator x
condition), so all strata appear in both parts in the same proportions.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

__all__ = ["calibration_split", "percentile_threshold"]


def calibration_split(df: pd.DataFrame, by: list[str], share: float = 0.30,
                      seed: int = 20261109) -> pd.Series:
    """Boolean Series, True for calibration records; deterministic given ``doc_id``s."""
    if not 0 < share < 1:
        raise ValueError("share must be in (0, 1)")
    out = pd.Series(False, index=df.index)
    for key, g in df.groupby(by, sort=True, dropna=False):
        tag = zlib.crc32(repr(key).encode())
        rng = np.random.default_rng([seed, tag])
        ids = np.sort(g["doc_id"].to_numpy())
        k = int(round(share * len(ids)))
        pick = set(rng.choice(ids, size=k, replace=False)) if k else set()
        out.loc[g.index] = g["doc_id"].isin(pick).to_numpy()
    return out


def percentile_threshold(human_scores: np.ndarray, q: float = 95.0) -> float:
    """Threshold of a zero-shot detector: the ``q``-th percentile of human scores."""
    s = np.asarray(human_scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("no finite human scores to calibrate on")
    return float(np.percentile(s, q))
