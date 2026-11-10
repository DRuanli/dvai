"""Held-out corpora of stage S5, second batch: non-English text and student essays.

Defined on 10 November 2026, after the protocol freeze and before any document of these
corpora was scored (protocol, section 8, addendum A1). Only file formats, lengths and
duplicate counts were inspected.

**M4** (Wang et al., EACL 2024), non-English subsets other than Chinese: Arabic (Wikipedia),
Bulgarian (news), German (Wikipedia), Indonesian (news), Russian (RuATD) and Urdu (news).
Each file holds 3,000 pairs of a human text and a machine text written from the same
prompt; Bulgarian and Russian have two generators (ChatGPT, davinci-003) over the same
human texts. Chinese is covered by HC3-Chinese, so the Chinese M4 subset is not used.

**HC3-Chinese** (Guo et al., 2023): the three sources with the most questions; human and
ChatGPT answers, at most ``cap`` per class and source.

**DAIGT v2** (Kaggle, with PERSUADE 2.0 as the human class): student essays and essays
written by several LLMs on the same prompts.

Sampling is by hash of the document id (``d2vi.data.heldout.hash_key``). ``group`` ties a
human text to the machine texts written from the same prompt, so that a calibration split
never separates them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .heldout import SEED, hash_key

__all__ = ["M4_URL", "M4_LANGUAGES", "M4_ENGLISH", "load_m4", "load_m4_english",
           "load_hc3_chinese", "load_daigt", "DAIGT"]

M4_URL = "https://raw.githubusercontent.com/mbzuai-nlp/M4/main/data"
M4_LANGUAGES = {
    "arabic": {"chatgpt": "arabic_chatGPT.jsonl"},
    "bulgarian": {"chatgpt": "bulgarian_true_and_fake_news_chatGPT.jsonl",
                  "davinci": "bulgarian_true_and_fake_news_davinci.jsonl"},
    "german": {"chatgpt": "germanwikipedia_chatgpt.jsonl"},
    "indonesian": {"chatgpt": "id-newspaper_chatGPT.jsonl"},
    "russian": {"chatgpt": "russian_chatGPT.jsonl", "davinci": "russian_davinci.jsonl"},
    "urdu": {"chatgpt": "urdu_chatGPT.jsonl"},
}
# English documents on which the cross-lingual detector is trained (never evaluated)
M4_ENGLISH = {"wikipedia": "wikipedia_chatgpt.jsonl", "reddit": "reddit_chatGPT.jsonl"}
DAIGT = {"columns": ("text", "label", "prompt_name", "source"), "human_source": "persuade_corpus"}


def _h(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _pairs(path: Path) -> list[tuple[str, str]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if "human_text" not in d or "machine_text" not in d:
                raise ValueError(f"{path.name}: expected keys human_text and machine_text, "
                                 f"found {sorted(d)}")
            out.append((str(d["human_text"] or "").strip(), str(d["machine_text"] or "").strip()))
    return out


def _m4_rows(pairs: list[tuple[str, str]], domain: str, generator: str, corpus: str) -> list:
    rows = []
    for human, machine in pairs:
        if not human or not machine:
            continue
        group = f"{corpus}-{domain}-{_h(human)}"
        rows.append({"doc_id": f"{group}-h", "corpus": corpus, "domain": domain,
                     "source": "human", "version": "none", "a": 0.0, "group": group,
                     "text": human})
        rows.append({"doc_id": f"{group}-{generator}-{_h(machine)}", "corpus": corpus,
                     "domain": domain, "source": generator, "version": "none", "a": 1.0,
                     "group": group, "text": machine})
    return rows


def load_m4(data_dir: str | Path, languages: dict | None = None) -> pd.DataFrame:
    """Non-English M4 subsets in long format; ``domain`` is the language. A human text that
    appears in the files of two generators is kept once."""
    rows = []
    for lang, files in (languages or M4_LANGUAGES).items():
        for generator, name in files.items():
            rows += _m4_rows(_pairs(Path(data_dir) / name), lang, generator, "m4")
    out = pd.DataFrame(rows).drop_duplicates("doc_id")
    out = out.sort_values("doc_id").drop_duplicates(["domain", "source", "text"])
    return out.reset_index(drop=True)


def load_m4_english(data_dir: str | Path, per_file: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Training documents of the cross-lingual detector: ``per_file`` pairs from each English
    M4 file of :data:`M4_ENGLISH`, chosen by hash."""
    parts = []
    for domain, name in M4_ENGLISH.items():
        d = pd.DataFrame(_m4_rows(_pairs(Path(data_dir) / name), domain, "chatgpt", "m4en"))
        groups = d.drop_duplicates("group")[["group"]]
        groups = groups.assign(key=hash_key(groups["group"], seed)).nsmallest(per_file, "key")
        parts.append(d[d["group"].isin(groups["group"])])
    return pd.concat(parts).drop_duplicates("doc_id").sort_values("doc_id").reset_index(drop=True)


def load_hc3_chinese(path: str | Path, n_sources: int = 3, cap: int = 2000,
                     seed: int = SEED) -> pd.DataFrame:
    """HC3-Chinese from ``all.jsonl``: the ``n_sources`` sources with the most questions,
    at most ``cap`` human and ``cap`` ChatGPT answers per source (by hash)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            if not line.strip():
                continue
            d = json.loads(line)
            if "human_answers" not in d or "chatgpt_answers" not in d:
                raise ValueError(f"HC3-Chinese: expected human_answers and chatgpt_answers, "
                                 f"found {sorted(d)}")
            src = str(d.get("source", "unknown"))
            group = f"hc3zh-{src}-{line_no}"
            for kind, a, key in (("human", 0.0, "human_answers"), ("chatgpt", 1.0,
                                                                   "chatgpt_answers")):
                for k, t in enumerate(d.get(key) or []):
                    t = str(t or "").strip()
                    if t:
                        rows.append({"doc_id": f"{group}-{kind}-{k}", "corpus": "hc3zh",
                                     "domain": src, "source": kind, "version": "none", "a": a,
                                     "group": group, "text": t})
    raw = pd.DataFrame(rows)
    top = raw.drop_duplicates("group")["domain"].value_counts().head(n_sources).index
    raw = raw[raw["domain"].isin(top)].copy()
    raw["_key"] = hash_key(raw["doc_id"], seed)
    keep = raw[raw.groupby(["domain", "source"])["_key"].rank(method="first") <= cap]
    return keep.drop(columns="_key").sort_values("doc_id").reset_index(drop=True)


def load_daigt(csv_path: str | Path, human_cap: int = 4000, per_source: int = 1000,
               min_source: int = 700, max_sources: int = 8, seed: int = SEED) -> pd.DataFrame:
    """DAIGT v2 training file: PERSUADE essays as the human class (at most ``human_cap``) and
    the ``max_sources`` largest machine sources with at least ``min_source`` essays
    (``per_source`` each), all chosen by hash of the text."""
    d = pd.read_csv(csv_path)
    missing = set(DAIGT["columns"]) - set(d.columns)
    if missing:
        raise ValueError(f"DAIGT file lacks columns {sorted(missing)}; found {list(d.columns)}")
    d = d.dropna(subset=["text"]).drop_duplicates("text")
    d["text"] = d["text"].astype(str).str.strip()
    human = d[d["label"].eq(0) & d["source"].eq(DAIGT["human_source"])]
    if human.empty:
        raise ValueError(f"DAIGT: no human essays with source {DAIGT['human_source']!r}; "
                         f"sources are {sorted(map(str, d['source'].unique()))}")
    machine = d[d["label"].eq(1)]
    sizes = machine["source"].value_counts()
    sizes = sizes[sizes >= min_source]
    chosen = sorted(sizes.sort_values(ascending=False, kind="stable").head(max_sources).index)
    if not chosen:
        raise ValueError(f"DAIGT: no machine source with {min_source} essays; "
                         f"sizes are {machine['source'].value_counts().to_dict()}")
    parts = []
    for src, cls, cap, a in [("human", human, human_cap, 0.0)] + [
            (s, machine[machine["source"].eq(s)], per_source, 1.0) for s in chosen]:
        ids = [f"daigt-{_h(t)}" for t in cls["text"]]
        part = pd.DataFrame({"doc_id": ids, "corpus": "daigt",
                             "domain": cls["prompt_name"].astype(str).to_numpy(),
                             "source": str(src).replace("/", "_"), "version": "none", "a": a,
                             "group": ids, "text": cls["text"].to_numpy()})
        part = part.assign(_key=hash_key(part["doc_id"], seed)).nsmallest(cap, "_key")
        parts.append(part.drop(columns="_key"))
    out = pd.concat(parts).drop_duplicates("doc_id")
    return out.sort_values("doc_id").reset_index(drop=True)
