"""Corpus loaders. Each returns a document table with one schema.

Columns: ``doc_id`` (stable string id), ``corpus``, ``domain``, ``source`` (``human`` or
the generator name), ``a`` (machine share), ``generator``, ``text``, ``path`` and
``text_sha256``. Texts never leave the machine that computes features; only features and
ids are published.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

__all__ = ["load_ghostbuster", "load_hc3", "GHOSTBUSTER_DOMAINS"]

GHOSTBUSTER_DOMAINS = ("essay", "reuter", "wp")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_ghostbuster(root: str | Path, domains=GHOSTBUSTER_DOMAINS,
                     sources=("human", "gpt", "claude"), min_chars: int = 60) -> pd.DataFrame:
    """Load the Ghostbuster corpus (Verma et al., 2024) from a clone of ghostbuster-data.

    Files are read in sorted path order; ``reuter`` keeps one sub-folder per author.
    Files under ``logprobs/`` are skipped. The same length filter (``> min_chars``
    characters after stripping) applies to every class, as in the original notebooks.
    """
    root = Path(root)
    rows = []
    for dom in domains:
        for src in sources:
            base = root / dom / src
            files = sorted(base.glob("*.txt")) or sorted(
                p for p in base.glob("*/*.txt") if "logprobs" not in p.parts)
            for rank, fp in enumerate(files):
                text = fp.read_text(encoding="utf-8", errors="ignore").strip()
                if len(text) <= min_chars:
                    continue
                rel = fp.relative_to(root).as_posix()
                rows.append({
                    "doc_id": f"ghostbuster:{rel}",
                    "corpus": "ghostbuster",
                    "domain": dom,
                    "source": src,
                    "a": 0.0 if src == "human" else 1.0,
                    "generator": "human" if src == "human" else src,
                    "text": text,
                    "path": rel,
                    "file_rank": rank,  # v1 Fast-DetectGPT pairing used file_rank < 600
                    "text_sha256": _sha(text),
                })
    return pd.DataFrame(rows)


def load_hc3(jsonl_path: str | Path, n_domains: int = 3, cap_per_class: int = 1500,
             min_chars: int = 60, sample_seed: int = 0) -> pd.DataFrame:
    """Load HC3 (Guo et al., 2023) from ``all.jsonl`` as in the v1 notebook.

    Human and ChatGPT answers longer than ``min_chars`` characters are kept; the
    ``n_domains`` largest sources are used, and each class is subsampled to
    ``cap_per_class`` documents with ``DataFrame.sample(random_state=sample_seed)``,
    which reproduces the v1 sample.
    """
    import json

    rows = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            d = json.loads(line)
            src = d.get("source", "unknown")
            for kind, a, key in (("human", 0.0, "human_answers"),
                                 ("chatgpt", 1.0, "chatgpt_answers")):
                for k, t in enumerate(d.get(key, []) or []):
                    if t and len(t) > min_chars:
                        text = t.strip()
                        rows.append({"doc_id": f"hc3:{line_no}:{kind}:{k}", "corpus": "hc3",
                                     "domain": src, "source": kind, "a": a,
                                     "generator": "human" if a == 0 else "chatgpt",
                                     "text": text, "text_sha256": _sha(text)})
    raw = pd.DataFrame(rows)
    domains = raw["domain"].value_counts().head(n_domains).index.tolist()
    parts = []
    for dom in domains:
        g = raw[raw["domain"] == dom]
        for a in (0.0, 1.0):
            cls = g[g["a"] == a]
            parts.append(cls.sample(min(cap_per_class, len(cls)), random_state=sample_seed))
    return pd.concat(parts, ignore_index=True)
