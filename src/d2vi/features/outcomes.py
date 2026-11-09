"""Outcome scorers: the measurements a social researcher computes from a response.

* :class:`VaderSentiment` - VADER compound score in [-1, 1] (Hutto and Gilbert, 2014).
* :class:`FleschReadingEase` - Flesch reading ease divided by 100.
* :class:`NRCTrust` - share of tokens listed under "trust" in the NRC emotion lexicon
  (Mohammad and Turney, 2013), using the lexicon bundled with the ``nrclex`` package.
* :class:`PangLeeSubjectivity` - mean predicted probability of subjectivity over the
  sentences of a document, from a TF-IDF logistic classifier trained on the Pang-Lee
  subjectivity corpus (5,000 subjective and 5,000 objective sentences).

All scorers truncate a text to its first 5,000 characters, as in the first submission.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

__all__ = ["Outcome", "VaderSentiment", "FleschReadingEase", "NRCTrust", "PangLeeSubjectivity",
           "split_halves"]

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_SENT = re.compile(r"(?<=[.!?])\s+")
MAX_CHARS = 5000


def split_halves(text: str) -> tuple[str, str]:
    """Split a text into two halves by word count (for the split-text design)."""
    words = text.split()
    k = len(words) // 2
    return " ".join(words[:k]), " ".join(words[k:])


class Outcome(ABC):
    name: str = "outcome"

    @abstractmethod
    def score(self, texts: list[str]) -> np.ndarray:
        """Return one value per text."""


class VaderSentiment(Outcome):
    name = "sent"

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._sia = SentimentIntensityAnalyzer()

    def score(self, texts):
        return np.array([self._sia.polarity_scores(t[:MAX_CHARS])["compound"] for t in texts])


class FleschReadingEase(Outcome):
    name = "read"

    def score(self, texts):
        import textstat

        return np.array([textstat.flesch_reading_ease(t[:MAX_CHARS]) / 100.0 for t in texts])


class NRCTrust(Outcome):
    name = "trust"

    def __init__(self, lexicon_path: str | Path | None = None) -> None:
        if lexicon_path is None:
            import nrclex

            lexicon_path = Path(nrclex.__file__).parent / "data" / "nrc_en.json"
        lex = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
        self._trust = {w for w, emos in lex.items() if "trust" in emos}

    def score(self, texts):
        out = np.empty(len(texts))
        for i, t in enumerate(texts):
            toks = _TOKEN.findall(t[:MAX_CHARS].lower())
            out[i] = sum(tok in self._trust for tok in toks) / len(toks) if toks else 0.0
        return out


class PangLeeSubjectivity(Outcome):
    """Supervised, non-lexical subjectivity scorer."""

    name = "subj"

    def __init__(self, corpus_dir: str | Path, seed: int = 0) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        corpus_dir = Path(corpus_dir)
        subj = (corpus_dir / "quote.tok.gt9.5000").read_text(encoding="latin-1").splitlines()
        obj = (corpus_dir / "plot.tok.gt9.5000").read_text(encoding="latin-1").splitlines()
        x = subj + obj
        y = np.r_[np.ones(len(subj)), np.zeros(len(obj))]
        self._model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                                    sublinear_tf=True),
                                    LogisticRegression(max_iter=2000, random_state=seed))
        self._model.fit(x, y)

    def score(self, texts):
        out = np.empty(len(texts))
        for i, t in enumerate(texts):
            sents = [s for s in _SENT.split(t[:MAX_CHARS]) if s.strip()] or [t[:MAX_CHARS]]
            out[i] = float(self._model.predict_proba(sents)[:, 1].mean())
        return out
