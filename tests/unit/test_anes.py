"""ANES loader on a tiny synthetic file pair with the real layout (tests only)."""

import numpy as np
import pandas as pd
import pytest

from d2vi.data.anes import ANES2024, chatbot_hallmark, clean_open_end, has_redaction
from d2vi.synthetic import ANES2024 as INSTRUMENT
from d2vi.synthetic import sample_personas


@pytest.mark.parametrize("raw, clean", [
    ("Stance on abortion /AE/ Their pro-business stances /AE/ No", "Stance on abortion Their "
     "pro-business stances"),
    ("he is  very racist /ao/ n", "he is very racist"),
    ("feminist ./ao/ thats it", "feminist"),
    ("their fiscal policies/ae/no", "their fiscal policies"),
    ("line one[CHAR(10)][CHAR(13)]line two      ", "line one\nline two"),
    ("Nothing.", ""),
    ("and/or the economy", "and/or the economy"),
])
def test_clean_open_end(raw, clean):
    assert clean_open_end(raw) == clean


def test_clean_open_end_is_fast_on_space_padded_cells():
    import time
    t = time.time()
    clean_open_end("word " + " " * 20000 + "x")
    assert time.time() - t < 0.5


def test_redaction_and_hallmarks():
    assert has_redaction("my [REDACTED FAMILY MEMBER] said")
    assert not has_redaction("[oil] prices")
    assert chatbot_hallmark("Here are points:[CHAR(10)]1. **Political Experience**: she")
    assert chatbot_hallmark("As an AI, I cannot vote")
    assert not chatbot_hallmark("1. too much mudslinging 2. name calling")
    assert not chatbot_hallmark("He's an arrogant a**")


@pytest.fixture
def files(tmp_path):
    ids = [140001, 140002, 140003, 140004, 140005, 140006]
    main = pd.DataFrame({"V240001": ids, "V240002a": [2, 1, 3, 2, 2, 2], "V240003": 2,
                         "V241458x": [34, 80, 50, -2, 41, 29], "V241550": [1, 2, 2, 1, 2, 1],
                         "V241501x": [1, 2, 3, 1, 6, 4], "V241465x": [2, 4, 5, 3, 1, 4],
                         "V243007": [1, 2, 3, 4, 3, 2], "V241227x": [1, 7, 4, 2, 6, 3],
                         "V241177": [2, 6, 99, 3, 5, 4]})
    rng = np.random.default_rng(0)
    for t in INSTRUMENT.thermometers:
        main[t.variable] = rng.integers(0, 101, len(ids))
    main.loc[4, "V241166"] = -9                        # missing thermometer
    for i in INSTRUMENT.open_items:
        main[i.filter_variable], main[i.variable] = 1, -2
    csv = tmp_path / "anes.csv"
    main.to_csv(csv, index=False)
    texts = {140001: "Her stance on abortion", 140002: "the economy /AE/ no",
             140003: "paper respondent", 140004: "my [REDACTED STATE] job",
             140005: "lower taxes", 140006: "Nothing"}
    xlsx = tmp_path / "oe.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        for k, i in enumerate(INSTRUMENT.open_items):
            col = f"{i.variable} - PRE: text"
            d = pd.DataFrame({"V240001": ids, col: [texts[j] if k == 0 else "" for j in ids]})
            d.to_excel(w, sheet_name=i.variable, index=False)
    return csv, xlsx


def test_respondents_exclusions_and_pool(files):
    data = ANES2024(*files)
    r = data.respondents().set_index("V240001")
    assert r.loc[140003, "exclusion"] == "paper interview"
    assert r.loc[140004, "exclusion"] == "redaction tag"
    assert r.loc[140005, "exclusion"] == "missing thermometer"
    assert r.loc[140006, "exclusion"] == "no open-ended text"
    assert r.loc[140002, "text"] == "the economy"
    pool = data.human_pool()
    assert pool["doc_id"].tolist() == ["anes-140001", "anes-140002"]
    assert (pool["a"] == 0).all() and pool["y_therm_harris"].between(0, 1).all()
    ps = sample_personas(pool, 5, seed=0)
    assert {p.source_id for p in ps} <= {"140001", "140002"}
