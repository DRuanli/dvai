"""Detectors that run on CPU. Neural detectors live in the Colab notebooks (``colab/``).

:class:`TfidfLRDetector` is the Study 1 detector: TF-IDF unigrams and bigrams with a
logistic regression, trained on the non-target domains (human vs one generator) and
deployed on the held-out target domain, so that no in-domain labels are used.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TfidfLRDetector"]


class TfidfLRDetector:
    """TF-IDF + logistic regression; ``predict`` returns ``P(machine)``.

    Defaults reproduce the v1 code exactly: vocabulary capped at 30,000 terms, unigrams
    and bigrams, ``sublinear_tf = True``, ``min_df = 1``, L2 penalty at the default
    strength, ``max_iter = 2000``. With these settings the legacy pipeline reproduces all
    32 cells of the first submission's Table 2; the paper's text said ``min_df = 2``,
    which changes one flagged essay and is corrected in the revision (finding F9).
    ``analyzer="char_wb"`` with character n-grams is used for languages without
    whitespace word boundaries (protocol addendum A1).
    """

    def __init__(self, max_features: int = 30000, ngram_range=(1, 2), min_df: int = 1,
                 sublinear_tf: bool = True, c: float = 1.0, seed: int = 0,
                 analyzer: str = "word") -> None:
        self.params = dict(max_features=max_features, ngram_range=tuple(ngram_range),
                           min_df=min_df, sublinear_tf=sublinear_tf, c=c, seed=seed,
                           analyzer=analyzer)
        self._model = None

    def fit(self, texts: list[str], labels: np.ndarray) -> TfidfLRDetector:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        p = self.params
        self._model = make_pipeline(
            TfidfVectorizer(max_features=p["max_features"], ngram_range=p["ngram_range"],
                            min_df=p["min_df"], sublinear_tf=p["sublinear_tf"],
                            analyzer=p["analyzer"]),
            LogisticRegression(C=p["c"], max_iter=2000, random_state=p["seed"]),
        )
        self._model.fit(texts, labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("detector is not fitted")
        return self._model.predict_proba(texts)[:, 1]
