import json

import pandas as pd
import pytest

from d2vi.synthetic import (
    ANES2024,
    ANES2024_CODEBOOK,
    Persona,
    RespondentPrompt,
    is_nothing,
    parse_answer,
    respondent_document,
    sample_personas,
)
from d2vi.synthetic.generator import SyntheticSurvey


def _rows(n=5):
    return pd.DataFrame({"V240001": range(100, 100 + n), "V241458x": [34, 80, -2, 51, 19][:n],
                         "V241550": [1, 2, 2, -9, 1][:n], "V241501x": [1, 2, 3, 4, 5][:n],
                         "V241465x": [2, 4, 5, 1, 3][:n], "V243007": [1, 2, 3, 4, 3][:n],
                         "V241227x": [1, 7, 4, 3, -8][:n], "V241177": [2, 6, 99, 4, -9][:n]})


def test_persona_levels_and_missing_codes():
    p = ANES2024_CODEBOOK.persona(_rows().iloc[2])
    assert p.age is None and p.sex == "woman"
    demo, pol = p.describe("demographic"), p.describe("political")
    assert "independent" not in demo and "an independent." in pol
    assert "haven't thought much" in pol


def test_codebook_rejects_unknown_codes():
    df = _rows()
    df.loc[0, "V241465x"] = 9
    with pytest.raises(ValueError, match="V241465x"):
        ANES2024_CODEBOOK.validate(df)


def test_sample_personas_is_seeded():
    a = sample_personas(_rows(), 20, seed=3)
    b = sample_personas(_rows(), 20, seed=3)
    assert [p.source_id for p in a] == [p.source_id for p in b]


def _reply(**over):
    obj = {i.key: 50 for i in ANES2024.thermometers}
    obj |= {i.key: "" for i in ANES2024.open_items}
    obj |= {"like_harris": "Her stance on abortion rights", "dislike_trump": "Nothing."}
    obj |= over
    return "Sure!\n```json\n" + json.dumps(obj) + "\n```"


def test_prompt_lists_every_item_and_parser_reads_fenced_json():
    msgs = RespondentPrompt(ANES2024, "political", "humanlike").messages(
        Persona("1", 40, "woman", "White", "a bachelor's degree", "South", "a strong Democrat",
                "liberal"))
    body = msgs[1]["content"]
    assert all(f'"{i.key}"' in body for i in ANES2024.items)
    p = parse_answer(_reply(), ANES2024)
    assert p.ok and p.thermometers["therm_trump"] == 50
    assert p.open["dislike_trump"] == ""           # "Nothing." means no answer
    assert respondent_document(p.open, ANES2024) == "Her stance on abortion rights"


@pytest.mark.parametrize("bad", [{"therm_harris": 101}, {"therm_harris": "warm"},
                                 {"therm_harris": 55.5}])
def test_parser_rejects_invalid_thermometers(bad):
    assert not parse_answer(_reply(**bad), ANES2024).ok


def test_is_nothing():
    assert is_nothing("  None. ") and is_nothing("") and is_nothing("I don't know")
    assert not is_nothing("No more wars")


class FakeGenerator:
    model_id = "fake/model"

    def __init__(self, fail_first=()):
        self.calls, self.fail_first = 0, set(fail_first)

    def generate(self, conversations, seed):
        self.calls += 1
        out = []
        for c in conversations:
            age = c[1]["content"].split("-year-old")[0].split()[-1]
            bad = age in self.fail_first and seed < 1_000_003
            out.append("I cannot answer that." if bad else _reply(therm_harris=int(age)))
        return out


def test_survey_retries_unparsed_replies_and_resumes(tmp_path):
    personas = [Persona(str(i), 20 + i, "man", None, None, None) for i in range(5)]
    survey = SyntheticSurvey(ANES2024, RespondentPrompt(ANES2024), str(tmp_path / "c.csv"))
    gen = FakeGenerator(fail_first={"22"})
    t = survey.run(gen, personas, prefix="fake", chunk=2)
    assert t.parse_ok.all() and t.loc[t.source_id == "2", "attempt"].item() == 1
    assert t.therm_harris.tolist() == [20, 21, 22, 23, 24]
    calls = gen.calls
    t2 = survey.run(gen, personas, prefix="fake", chunk=2)   # everything cached
    assert gen.calls == calls and t2.equals(t)


def test_generator_repeats_a_batch_whose_sampling_distribution_is_invalid():
    torch = pytest.importorskip("torch")
    from d2vi.synthetic.generator import GENERATORS, HFChatGenerator

    assert all(GENERATORS.values())                       # all three models load in 4 bit

    class Enc(dict):
        def to(self, device):
            return self

    class Tok:
        def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=True):
            return conv[-1]["content"]

        def __call__(self, texts, return_tensors="pt", padding=True):
            return Enc(input_ids=torch.zeros((len(texts), 3), dtype=torch.long))

        def batch_decode(self, ids, skip_special_tokens=True):
            return [f"reply {int(row[0])}" for row in ids]

    class Model:
        device, calls = "cpu", []

        def generate(self, input_ids, **kw):
            self.calls.append(kw)
            if len(self.calls) == 2 and "remove_invalid_values" not in kw:
                raise RuntimeError("probability tensor contains either `inf`, `nan` or element < 0")
            new = torch.full((input_ids.shape[0], 2), len(self.calls))
            return torch.cat([input_ids, new], dim=1)

    gen = object.__new__(HFChatGenerator)
    gen.tok, gen.model, gen.batch_size = Tok(), Model(), 2
    gen.gen_kw = {"max_new_tokens": 4}
    gen.settings = {"nan_repaired_batches": 0}
    out = gen.generate([[{"role": "user", "content": f"q{i}"}] for i in range(4)], seed=1)
    assert out == ["reply 1", "reply 1", "reply 3", "reply 3"]
    assert gen.settings["nan_repaired_batches"] == 1
    assert gen.model.calls[2].get("remove_invalid_values") is True
    gen.model.generate = lambda **kw: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))
    with pytest.raises(RuntimeError, match="out of memory"):
        gen.generate([[{"role": "user", "content": "q"}]], seed=1)
    gen.close()
    assert gen.model is None
