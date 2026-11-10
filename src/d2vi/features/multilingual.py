"""Outcomes and detectors for non-English held-out corpora (protocol addendum A1).

The frozen protocol fixed the detector for non-English text (XLM-R trained on English
calibration data, applied cross-lingually) but not the outcomes, because the outcomes of
the English corpora (VADER, Flesch, NRC, Pang-Lee) are English-specific. Addendum A1,
written before any non-English document was scored, fixes:

* ``y_sentx`` - sentiment from a multilingual classifier
  (``cardiffnlp/twitter-xlm-roberta-base-sentiment``): P(positive) - P(negative) on the
  first 256 subword tokens;
* ``y_ttr`` - lexical diversity: type-token ratio of the first 50 XLM-R subword tokens.
  Documents with fewer than 50 subword tokens are excluded and counted;
* ``score_xlmr`` - ``xlm-roberta-base`` fine-tuned on English human and ChatGPT documents of
  M4 (Wikipedia and Reddit; 2 epochs, learning rate 2e-5, 256 tokens), P(machine), flag
  above 0.5; no document of the target language is used;
* ``score_tfidfc`` (secondary) - character 2-4-gram TF-IDF with logistic regression, trained
  per language on the 30% calibration split (human and ChatGPT documents), flag above 0.5;
  where a language has a second generator, that generator is a generator shift.

``length`` is the number of subword tokens. GPU code imports ``torch`` lazily.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.heldout import SEED
from ..data.splits import calibration_split
from .detectors import TfidfLRDetector

__all__ = ["subword_ttr", "XLMRDetector", "XLMRSentiment", "score_multilingual",
           "finalize_multilingual", "MIN_TOKENS", "XLMR_ID", "SENTIMENT_ID"]

XLMR_ID = "xlm-roberta-base"
SENTIMENT_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MIN_TOKENS = 50
MAX_LEN = 256


def subword_ttr(ids: Sequence[int], k: int = MIN_TOKENS) -> float:
    """Type-token ratio of the first ``k`` subword tokens; ``nan`` if there are fewer."""
    ids = list(ids)
    if len(ids) < k:
        return float("nan")
    return len(set(ids[:k])) / k


def _batches(n: int, size: int):
    for i in range(0, n, size):
        yield slice(i, i + size)


class XLMRDetector:
    """XLM-R sequence classifier: ``fit`` on English documents, ``predict`` = P(machine)."""

    def __init__(self, model_id: str = XLMR_ID, max_len: int = MAX_LEN, epochs: int = 2,
                 lr: float = 2e-5, batch_size: int = 32, seed: int = SEED,
                 device: str | None = None) -> None:
        self.model_id, self.max_len, self.epochs = model_id, int(max_len), int(epochs)
        self.lr, self.batch_size, self.seed = float(lr), int(batch_size), int(seed)
        self.device = device
        self.tok = None
        self.model = None
        self.log: list[dict] = []

    def _setup(self, source: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2)
        self.model.to(self.device)

    def _encode(self, texts: list[str]):
        return self.tok(texts, padding=True, truncation=True, max_length=self.max_len,
                        return_tensors="pt").to(self.device)

    def fit(self, texts: list[str], labels: Sequence[float]) -> XLMRDetector:
        import torch

        torch.manual_seed(self.seed)
        self._setup(self.model_id)
        y = torch.tensor(np.asarray(labels).astype(int))
        n = len(texts)
        steps = self.epochs * int(np.ceil(n / self.batch_size))
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        warm = max(1, int(0.06 * steps))

        def schedule(s: int) -> float:          # linear warm-up, then linear decay
            return (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
        rng = np.random.default_rng(self.seed)
        self.model.train()
        step = 0
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for sl in _batches(n, self.batch_size):
                idx = order[sl]
                enc = self._encode([texts[i] for i in idx])
                loss = self.model(**enc, labels=y[idx].to(self.device)).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 50 == 0 or step == steps:
                    self.log.append({"step": step, "epoch": epoch, "loss": loss.item()})
                    print(f"  step {step}/{steps} loss {loss.item():.4f}", flush=True)
        self.model.eval()
        return self

    def predict(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("detector is not fitted or loaded")
        self.model.eval()
        out = []
        with torch.no_grad():
            for sl in _batches(len(texts), batch_size):
                logits = self.model(**self._encode(texts[sl])).logits.float()
                out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def save(self, path: str | Path) -> None:
        self.model.save_pretrained(str(path))

    def load(self, path: str | Path) -> XLMRDetector:
        self._setup(str(path))
        self.model.eval()
        return self


class XLMRSentiment:
    """P(positive) - P(negative) of a multilingual sentiment classifier. The classifier is an
    XLM-R base model, so the ``xlm-roberta-base`` tokenizer is used."""

    def __init__(self, model_id: str = SENTIMENT_ID, max_len: int = MAX_LEN,
                 device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(XLMR_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device).eval()
        self.max_len = int(max_len)
        labels = {int(i): str(v).lower() for i, v in self.model.config.id2label.items()}
        self.neg = next((i for i, v in labels.items() if v.startswith("neg")), 0)
        self.pos = next((i for i, v in labels.items() if v.startswith("pos")), 2)
        if self.model.config.num_labels != 3 or self.neg == self.pos:
            raise ValueError(f"unexpected sentiment labels {labels}")

    def predict(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for sl in _batches(len(texts), batch_size):
                enc = self.tok(texts[sl], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(self.device)
                p = torch.softmax(self.model(**enc).logits.float(), -1)
                out.append((p[:, self.pos] - p[:, self.neg]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def score_multilingual(docs: pd.DataFrame, detector, sentiment, tokenizer,
                       cache_path: str | Path, chunk: int = 2000) -> pd.DataFrame:
    """``length``, ``y_ttr``, ``y_sentx`` and ``score_xlmr`` for ``docs`` (``doc_id``,
    ``text``); cached in ``cache_path`` every ``chunk`` documents so that a run resumes."""
    cols = ["doc_id", "length", "y_ttr", "y_sentx", "score_xlmr"]
    done = (pd.read_csv(cache_path, dtype={"doc_id": str}) if os.path.exists(cache_path)
            else pd.DataFrame(columns=cols))
    todo = docs[~docs["doc_id"].isin(done["doc_id"])]
    print(f"cached {len(done)}, to score {len(todo)}", flush=True)
    for sl in _batches(len(todo), chunk):
        part = todo.iloc[sl]
        texts = part["text"].tolist()
        ids = tokenizer(texts, add_special_tokens=False, truncation=True,
                        max_length=4096)["input_ids"]
        new = pd.DataFrame({"doc_id": part["doc_id"].to_numpy(),
                            "length": [len(i) for i in ids],
                            "y_ttr": [subword_ttr(i) for i in ids],
                            "y_sentx": sentiment.predict(texts),
                            "score_xlmr": detector.predict(texts)})
        done = pd.concat([done, new], ignore_index=True)
        done.to_csv(cache_path, index=False)
        print(f"  {len(done)} scored", flush=True)
    return done.drop_duplicates("doc_id")[cols]


def finalize_multilingual(table: pd.DataFrame, share: float = 0.30,
                          min_tokens: int = MIN_TOKENS) -> tuple[pd.DataFrame, dict]:
    """Length filter, calibration split by ``group`` within language, in-language character
    TF-IDF detector, flags. Returns the evaluation table (no text) and a summary.

    ``table``: ``doc_id``, ``domain`` (language or source), ``source``, ``a``, ``group``,
    ``text``, ``length``, ``y_ttr``, ``y_sentx``, ``score_xlmr``.
    """
    short = table["length"] < min_tokens
    info: dict = {"rows": int(len(table)), "too_short": int(short.sum()),
                  "too_short_by_class": {str(k): int(v) for k, v in
                                         short.groupby(table["a"]).sum().items()},
                  "calibration": {}}
    table = table[~short]
    parts = []
    for dom in sorted(table["domain"].unique()):
        t = table[table["domain"].eq(dom)]
        groups = t.drop_duplicates("group")[["group"]].rename(columns={"group": "doc_id"})
        groups["domain"] = dom
        cal_groups = set(groups.loc[calibration_split(groups, ["domain"], share), "doc_id"])
        cal = t["group"].isin(cal_groups)
        train = t[cal & (t["a"].eq(0) | t["source"].eq("chatgpt"))]
        if train["a"].nunique() < 2:
            raise ValueError(f"{dom}: calibration split lacks a class for the TF-IDF detector")
        tfidf = TfidfLRDetector(analyzer="char_wb", ngram_range=(2, 4))
        tfidf.fit(train["text"].tolist(), train["a"].to_numpy())
        ev = t[~cal].copy()
        ev["score_tfidfc"] = tfidf.predict(ev["text"].tolist())
        ev["flag_tfidfc"] = (ev["score_tfidfc"] > 0.5).astype(int)
        ev["flag_xlmr"] = (ev["score_xlmr"] > 0.5).astype(int)
        parts.append(ev)
        info["calibration"][dom] = {"groups": len(cal_groups), "rows": int(cal.sum()),
                                    "tfidf_human": int((train.a == 0).sum()),
                                    "tfidf_machine": int((train.a > 0).sum())}
    out = pd.concat(parts).drop(columns=["text"]).sort_values("doc_id").reset_index(drop=True)
    info["evaluation_rows"] = int(len(out))
    return out, info
