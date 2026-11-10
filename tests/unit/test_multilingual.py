"""Second batch of held-out corpora: loaders and feature rules (constructed files only)."""

import json

import numpy as np
import pandas as pd
import pytest

from d2vi.data.multilingual import load_daigt, load_hc3_chinese, load_m4, load_m4_english
from d2vi.features.heldout import finalize_features
from d2vi.features.multilingual import finalize_multilingual, score_multilingual, subword_ttr


def _jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_m4_loader_pairs_texts_and_keeps_shared_human_texts_once(tmp_path):
    humans = [f"human text number {i}" for i in range(5)]
    _jsonl(tmp_path / "ru_a.jsonl", [{"human_text": h, "machine_text": f"gpt {i}", "model": "x",
                                      "promt": "typo key"} for i, h in enumerate(humans)])
    _jsonl(tmp_path / "ru_b.jsonl", [{"human_text": h, "machine_text": f"dav {i}", "source_id": 1}
                                     for i, h in enumerate(humans[:4])] +
           [{"human_text": "only here", "machine_text": ""}])          # empty machine: dropped
    d = load_m4(tmp_path, {"russian": {"chatgpt": "ru_a.jsonl", "davinci": "ru_b.jsonl"}})
    assert d.groupby("source").size().to_dict() == {"chatgpt": 5, "davinci": 4, "human": 5}
    assert d["doc_id"].is_unique and set(d["domain"]) == {"russian"}
    g = d[d["text"].eq("human text number 0")]["group"].iloc[0]
    assert set(d[d["group"].eq(g)]["source"]) == {"human", "chatgpt", "davinci"}
    with pytest.raises(ValueError, match="human_text"):
        _jsonl(tmp_path / "bad.jsonl", [{"text": "x"}])
        load_m4(tmp_path, {"x": {"chatgpt": "bad.jsonl"}})


def test_m4_english_training_sample_is_order_independent(tmp_path, monkeypatch):
    import d2vi.data.multilingual as ml

    rows = [{"human_text": f"h{i}", "machine_text": f"m{i}"} for i in range(30)]
    _jsonl(tmp_path / "a.jsonl", rows)
    _jsonl(tmp_path / "b.jsonl", rows[::-1])
    monkeypatch.setattr(ml, "M4_ENGLISH", {"wiki": "a.jsonl"})
    one = load_m4_english(tmp_path, per_file=10)
    monkeypatch.setattr(ml, "M4_ENGLISH", {"wiki": "b.jsonl"})
    two = load_m4_english(tmp_path, per_file=10)
    assert len(one) == 20 and one["doc_id"].tolist() == two["doc_id"].tolist()
    assert one.groupby("group").size().eq(2).all()


def test_hc3_chinese_takes_the_largest_sources_and_caps_by_hash(tmp_path):
    rows = []
    for src, n in (("baike", 6), ("law", 5), ("open_qa", 4), ("tiny", 1)):
        rows += [{"question": "q", "human_answers": [f"{src} h{i}", ""],
                  "chatgpt_answers": [f"{src} c{i}"], "source": src} for i in range(n)]
    _jsonl(tmp_path / "all.jsonl", rows)
    d = load_hc3_chinese(tmp_path / "all.jsonl", n_sources=3, cap=4)
    assert set(d["domain"]) == {"baike", "law", "open_qa"}
    assert d.groupby(["domain", "source"]).size().max() == 4
    assert d.groupby("group")["source"].nunique().max() == 2 and d["doc_id"].is_unique


def test_daigt_selects_persuade_humans_and_the_largest_machine_sources(tmp_path):
    rows = [{"text": f"human essay {i}", "label": 0, "prompt_name": "p1",
             "source": "persuade_corpus"} for i in range(12)]
    rows += [{"text": f"other human {i}", "label": 0, "prompt_name": "p1", "source": "train_essays"}
             for i in range(3)]
    for src, n in (("org/big", 9), ("mid", 7), ("small", 2)):
        rows += [{"text": f"{src} essay {i}", "label": 1, "prompt_name": "p2", "source": src}
                 for i in range(n)]
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    d = load_daigt(path, human_cap=10, per_source=6, min_source=5, max_sources=8)
    assert d.groupby("source").size().to_dict() == {"human": 10, "mid": 6, "org_big": 6}
    assert d["doc_id"].is_unique and d["group"].equals(d["doc_id"])
    again = load_daigt(path, human_cap=10, per_source=6, min_source=5, max_sources=8)
    assert d["doc_id"].tolist() == again["doc_id"].tolist()
    pd.DataFrame(rows).drop(columns="source").to_csv(path, index=False)
    with pytest.raises(ValueError, match="lacks columns"):
        load_daigt(path)


def test_subword_ttr():
    assert subword_ttr(list(range(50))) == 1.0
    assert subword_ttr([1, 2] * 40) == 2 / 50
    assert np.isnan(subword_ttr(list(range(49))))
    assert subword_ttr(list(range(50)) + [0] * 500) == 1.0          # only the first 50 count


class _Stub:
    def __init__(self, value):
        self.value, self.calls = value, 0

    def predict(self, texts):
        self.calls += len(texts)
        return np.full(len(texts), self.value)


def _tokenizer(texts, **kwargs):
    return {"input_ids": [[int(w[1:]) for w in t.split()] for t in texts]}


def test_score_multilingual_resumes_from_its_cache(tmp_path):
    docs = pd.DataFrame({"doc_id": [f"d{i}" for i in range(7)],
                         "text": [" ".join(f"w{j}" for j in range(40 + 5 * i)) for i in range(7)]})
    det, sent, cache = _Stub(0.9), _Stub(-0.2), tmp_path / "cache.csv"
    first = score_multilingual(docs.iloc[:4], det, sent, _tokenizer, cache, chunk=3)
    assert det.calls == 4 and first["length"].tolist() == [40, 45, 50, 55]
    assert np.isnan(first["y_ttr"].iloc[0]) and first["y_ttr"].iloc[2] == pytest.approx(1.0)
    full = score_multilingual(docs, det, sent, _tokenizer, cache, chunk=3)
    assert det.calls == 7 and len(full) == 7 and (full["score_xlmr"] == 0.9).all()


def _table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for lang, gens in (("xx", ("chatgpt", "davinci")), ("yy", ("chatgpt",))):
        for i in range(n):
            g = f"{lang}-{i}"
            rows.append((f"{g}-h", lang, "human", 0.0, g, f"plain human words {lang} {i} abc"))
            for gen in gens:
                rows.append((f"{g}-{gen}", lang, gen, 1.0, g, f"ZZZ machine QQQ {lang} {i} zzz"))
    t = pd.DataFrame(rows, columns=["doc_id", "domain", "source", "a", "group", "text"])
    t["length"] = 60
    t.loc[t.index[:10], "length"] = 20                       # too short: excluded
    t["y_ttr"], t["y_sentx"] = rng.random(len(t)), rng.normal(size=len(t))
    t["score_xlmr"] = np.where(t["a"] == 1, 0.8, 0.3)
    return t


def test_finalize_multilingual_splits_by_group_and_trains_per_language():
    t = _table()
    out, info = finalize_multilingual(t)
    assert info["too_short"] == 10 and "text" not in out
    assert len(out) + sum(v["rows"] for v in info["calibration"].values()) == len(t) - 10
    cal_groups = set(t[~t["length"].lt(50)]["group"]) - set(out["group"])
    assert not cal_groups & set(out["group"])                # a group is never split
    for lang in ("xx", "yy"):
        share = info["calibration"][lang]["groups"] / t[t.domain.eq(lang)]["group"].nunique()
        assert share == pytest.approx(0.30, abs=0.02)
    # trained on human vs ChatGPT only; davinci documents are a generator shift
    for lang in ("xx", "yy"):
        assert info["calibration"][lang]["tfidf_machine"] <= info["calibration"][lang]["groups"]
    assert out.loc[out.a.eq(1), "flag_tfidfc"].mean() > 0.95
    assert out.loc[out.a.eq(0), "flag_tfidfc"].mean() < 0.05
    assert out["flag_xlmr"].equals(out["a"].astype(int))


def test_finalize_features_split_can_stratify_by_generator():
    rng = np.random.default_rng(1)
    rows = []
    for src, a, n in (("human", 0.0, 200), ("g1", 1.0, 100), ("g2", 1.0, 50)):
        for i in range(n):
            rows.append((f"{src}-{i}", "essay", src, a, f"{src}-{i}",
                         f"{'human' if a == 0 else 'machine'} words {src} {i}"))
    t = pd.DataFrame(rows, columns=["doc_id", "domain", "source", "a", "group", "text"])
    t["score_fdgpt"], t["score_roberta"] = rng.normal(size=len(t)), rng.random(len(t))
    out, info = finalize_features(t, "split", t["source"].eq("g1"), strata=("source",))
    assert out.groupby("source").size().to_dict() == {"g1": 70, "g2": 35, "human": 140}
    assert info["tfidf_training"] == {"human": 60, "machine": 30}
