"""Held-out corpora of stage S5: seeded, order-independent subsets (frozen protocol, section 1).

The subsets below were fixed on 10 November 2026, after the protocol freeze and before any
held-out document was scored; they are sized to the free GPU budget (about 48,000 documents).

**RAID** (Dugan et al., ACL 2024; training split, which has labels). Five domains - news,
Reddit posts, IMDb reviews, poetry, book summaries (the last two for the creative-text
request) - with every human document (at most ``human_cap`` per domain), four generators
without attack (ChatGPT, GPT-4, Llama 2 70B chat, Mistral 7B chat) and three adversarial
attacks (paraphrase, synonym swap, homoglyph) on two generators (ChatGPT, Llama 2 chat);
``per_group`` documents per domain x generator x attack, all decoding settings together.

**Beemo** (Artemova et al., NAACL 2025). All 2,187 prompts. Versions: the human answer, the
raw output of one of ten open LLMs, the expert-edited output, and the outputs edited by
Llama 3.1 70B and by GPT-4o (one of the three editing prompts per item, chosen by hash).

Sampling is by a hash of the document id and the seed, so a subset does not depend on file
order or chunking and can be rebuilt exactly.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

__all__ = ["RAID", "hash_key", "sample_raid", "load_beemo", "MIN_CHARS"]

MIN_CHARS = 60          # same length filter as the development corpora
SEED = 20261109

RAID = {
    "domains": ("news", "reddit", "reviews", "poetry", "books"),
    "models": ("chatgpt", "gpt4", "llama-chat", "mistral-chat"),
    "attacks": ("paraphrase", "synonym", "homoglyph"),
    "attack_models": ("chatgpt", "llama-chat"),
    "columns": ("id", "model", "decoding", "attack", "domain", "generation"),
}


def hash_key(ids: Iterable, seed: int = SEED) -> np.ndarray:
    """Pseudo-random 64-bit key per id; smallest keys are sampled."""
    return np.array([int.from_bytes(hashlib.md5(f"{seed}|{i}".encode()).digest()[:8], "big")
                     for i in ids], dtype=np.uint64)


def _wanted(chunk: pd.DataFrame) -> pd.Series:
    dom = chunk["domain"].isin(RAID["domains"])
    none = chunk["attack"].eq("none")
    human = chunk["model"].eq("human") & none
    plain = chunk["model"].isin(RAID["models"]) & none
    attacked = chunk["model"].isin(RAID["attack_models"]) & chunk["attack"].isin(RAID["attacks"])
    return dom & (human | plain | attacked)


def sample_raid(chunks: Iterable[pd.DataFrame], per_group: int = 600, human_cap: int = 2000,
                seed: int = SEED) -> pd.DataFrame:
    """Subset of RAID from an iterator of chunks (``pd.read_csv(..., chunksize=...)`` over
    ``train_none.csv`` and ``train.csv``; a row present in both is kept once)."""
    keep: pd.DataFrame | None = None
    seen = {"model": set(), "domain": set(), "attack": set()}
    for chunk in chunks:
        missing = set(RAID["columns"]) - set(chunk.columns)
        if missing:
            raise ValueError(f"RAID file lacks columns {sorted(missing)}; "
                             f"found {list(chunk.columns)}")
        for col in seen:
            seen[col] |= set(chunk[col].dropna().unique())
        part = chunk.loc[_wanted(chunk), list(RAID["columns"])].copy()
        part = part[part["generation"].astype(str).str.len() >= MIN_CHARS]
        if part.empty:
            continue
        part["_key"] = hash_key(part["id"], seed)
        keep = part if keep is None else pd.concat([keep, part], ignore_index=True)
        keep = keep.drop_duplicates("id")
        cap = np.where(keep["model"].eq("human"), human_cap, per_group)
        rank = keep.groupby(["domain", "model", "attack"])["_key"].rank(method="first")
        keep = keep[rank <= cap].reset_index(drop=True)
    for col, want in (("domain", RAID["domains"]), ("attack", ("none", *RAID["attacks"])),
                      ("model", ("human", *RAID["models"]))):
        absent = set(want) - seen[col]
        if absent:
            raise ValueError(f"RAID: values {sorted(absent)} of '{col}' not found; "
                             f"the files contain {sorted(map(str, seen[col]))}")
    if keep is None:
        raise ValueError("RAID: no rows selected")
    out = pd.DataFrame({
        "doc_id": "raid-" + keep["id"].astype(str), "corpus": "raid", "domain": keep["domain"],
        "source": keep["model"], "version": keep["attack"],
        "a": (~keep["model"].eq("human")).astype(float), "decoding": keep["decoding"],
        "text": keep["generation"].astype(str)})
    return out.sort_values("doc_id").reset_index(drop=True)


def _edit(cell: str, item_id, seed: int) -> tuple[str, str]:
    """One of the editing prompts P1-P3 of an LLM-edited Beemo item, chosen by hash."""
    variants = {k: v for d in ast.literal_eval(cell) for k, v in d.items()}
    names = sorted(variants)
    pick = names[int(hash_key([item_id], seed)[0] % len(names))]
    return pick, str(variants[pick])


def load_beemo(path: str, seed: int = SEED) -> pd.DataFrame:
    """Beemo in long format: one row per prompt x version. ``group`` is the prompt, so that
    all versions of a prompt fall on the same side of the calibration split."""
    d = pd.read_parquet(path)
    need = {"id", "category", "model", "model_output", "human_output", "human_edits",
            "llama-3.1-70b_edits", "gpt-4o_edits"}
    if need - set(d.columns):
        raise ValueError(f"Beemo file lacks columns {sorted(need - set(d.columns))}")
    rows = []
    for rec in d.to_dict("records"):
        base = {"corpus": "beemo", "domain": rec["category"], "group": f"beemo-{rec['id']}"}
        versions = [("human", "human", 0.0, rec["human_output"], ""),
                    ("machine", rec["model"], 1.0, rec["model_output"], ""),
                    ("expert_edited", rec["model"], 1.0, rec["human_edits"], "")]
        for col, name in (("llama-3.1-70b_edits", "llama31_edited"), ("gpt-4o_edits",
                                                                      "gpt4o_edited")):
            prompt, text = _edit(rec[col], rec["id"], seed)
            versions.append((name, rec["model"], 1.0, text, prompt))
        for version, source, a, text, prompt in versions:
            rows.append({**base, "doc_id": f"beemo-{rec['id']}-{version}", "source": source,
                         "version": version, "a": a, "edit_prompt": prompt, "text": str(text)})
    out = pd.DataFrame(rows)
    out = out[out["text"].str.len() >= MIN_CHARS]
    return out.sort_values("doc_id").reset_index(drop=True)
