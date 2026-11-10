"""Held-out corpus loaders and the calibration rule on constructed files (tests only)."""

import numpy as np
import pandas as pd
import pytest

from d2vi.data.heldout import RAID, hash_key, load_beemo, sample_raid
from d2vi.features.heldout import finalize_features, operating_points

TEXT = "this is a sufficiently long document about something " * 3


def _raid_rows(n_per=30):
    rows, i = [], 0
    for dom in (*RAID["domains"], "wiki"):
        for model in ("human", *RAID["models"], "gpt2"):
            attacks = ("none", *RAID["attacks"], "whitespace") if model != "human" else ("none",)
            for att in attacks:
                for _ in range(n_per):
                    rows.append({"id": f"id{i}", "adv_source_id": "x", "model": model,
                                 "decoding": "greedy", "attack": att, "domain": dom,
                                 "title": "t", "prompt": "p", "generation": TEXT + str(i)})
                    i += 1
    return pd.DataFrame(rows)


def test_raid_subset_is_order_and_chunk_independent_and_respects_the_design():
    raw = _raid_rows()
    a = sample_raid([raw], per_group=10, human_cap=20)
    shuffled = raw.sample(frac=1, random_state=1)
    chunks = [shuffled.iloc[i::7] for i in range(7)] + [raw.head(500)]
    b = sample_raid(chunks, per_group=10, human_cap=20)
    assert a["doc_id"].tolist() == b["doc_id"].tolist()
    sizes = a.groupby(["domain", "source", "version"]).size()
    assert set(a["domain"]) == set(RAID["domains"]) and "gpt2" not in set(a["source"])
    assert sizes.xs("human", level="source").eq(20).all()
    assert sizes.drop("human", level="source").eq(10).all()
    attacked = a[a.version != "none"]
    assert set(attacked["source"]) == set(RAID["attack_models"])
    assert len(sizes) == 5 * (1 + 4 + 2 * 3)
    assert a.loc[a.source == "human", "a"].eq(0).all()
    assert a.loc[a.source != "human", "a"].eq(1).all()


def test_raid_schema_errors_are_explicit():
    raw = _raid_rows(2)
    with pytest.raises(ValueError, match="lacks columns"):
        sample_raid([raw.drop(columns=["attack"])])
    with pytest.raises(ValueError, match="poetry"):
        sample_raid([raw[raw.domain != "poetry"]])


def test_hash_key_is_deterministic():
    assert (hash_key(["a", "b"], 1) == hash_key(["a", "b"], 1)).all()
    assert hash_key(["a"], 1)[0] != hash_key(["a"], 2)[0]


@pytest.fixture
def beemo_file(tmp_path):
    edits = str([{"P1": TEXT + "one"}, {"P2": TEXT + "two"}, {"P3": TEXT + "three"}])
    d = pd.DataFrame({"id": range(40), "category": ["Generation", "Rewrite"] * 20,
                      "model": "m/x", "prompt_id": "p", "prompt": "q",
                      "model_output": TEXT + "machine", "human_output": [TEXT + "human"] * 39
                      + ["short"], "human_edits": TEXT + "edited",
                      "llama-3.1-70b_edits": edits, "gpt-4o_edits": edits})
    path = tmp_path / "beemo.parquet"
    d.to_parquet(path)
    return str(path)


def test_beemo_long_format(beemo_file):
    b = load_beemo(beemo_file)
    assert b.groupby("version").size().to_dict() == {
        "expert_edited": 40, "gpt4o_edited": 40, "human": 39, "llama31_edited": 40,
        "machine": 40}
    assert b.loc[b.version == "human", "a"].eq(0).all()
    assert set(b.loc[b.version == "gpt4o_edited", "edit_prompt"]) <= {"P1", "P2", "P3"}
    assert b["doc_id"].is_unique and b["group"].nunique() == 40


def _scored(docs, rng):
    docs = docs.copy()
    docs["score_fdgpt"] = rng.normal(docs["a"] * 2.0, 1.0)
    docs["score_roberta"] = rng.random(len(docs))
    docs["y_sent"] = rng.random(len(docs))
    docs["length"] = 30
    docs["text"] = np.where(docs["a"] > 0, "generated content model ", "people wrote this ") \
        + docs["text"]
    return docs


def test_split_calibration_keeps_groups_together_and_returns_evaluation_rows(beemo_file):
    table = _scored(load_beemo(beemo_file), np.random.default_rng(0))
    out, info = finalize_features(table, "split", table["version"].eq("machine"))
    assert "text" not in out and info["calibration_groups"] == 12
    assert out["group"].nunique() == 28
    assert info["evaluation_rows"] == len(out) == len(table) - info["calibration_rows"]
    ops = operating_points(out, by=("version",)).set_index("version")
    assert ops.loc["machine", "flag_tfidf"] > 0.9 and ops.loc["human", "flag_tfidf"] < 0.1


def test_cross_domain_calibration_uses_other_domains_only():
    rng = np.random.default_rng(1)
    table = _scored(sample_raid([_raid_rows()], per_group=12, human_cap=25), rng)
    shift = table["domain"].eq("news") & table["a"].eq(0)
    table.loc[shift, "score_fdgpt"] += 100.0       # must not affect the news threshold
    train = table["source"].eq("chatgpt") & table["version"].eq("none")
    out, info = finalize_features(table, "cross_domain", train)
    assert len(out) == len(table)
    assert info["thresholds_fdgpt"]["news"] < 10 < info["thresholds_fdgpt"]["poetry"]
    assert info["tfidf_training"]["news"] == {"human": 100, "machine": 48}
    assert out.loc[out.domain == "news", "flag_fdgpt"].mean() > 0.5
