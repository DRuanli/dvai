"""Prompts and answer parsing for synthetic survey respondents.

A synthetic respondent answers the whole instrument in one call and returns JSON: one
integer per thermometer and one string per open-ended question (empty when the
respondent has nothing in particular to say, which corresponds to "no" on the ANES
filter question).

Two answer styles are separate generator conditions:

* ``"plain"``     - no guidance on writing style (the model's own register);
* ``"humanlike"`` - asked to write as a person typing into an online form would, which
  is what a survey bot or a respondent pasting from a chatbot is trying to achieve.

Human and synthetic records are turned into the detector's input document by the same
function, :func:`respondent_document`, so that no formatting difference can separate them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .instrument import SurveyInstrument
from .persona import Persona

__all__ = ["RespondentPrompt", "ParsedAnswer", "parse_answer", "respondent_document",
           "is_nothing"]

_NOTHING = re.compile(r"^\s*(no|nope|none|nothing|nothing really|not really|n/?a|na|-+|"
                      r"no comment|nothing in particular|not that i know of|i don'?t know|"
                      r"dk|idk)\s*[.!]*\s*$", re.IGNORECASE)

STYLE = {
    "plain": "",
    "humanlike": (
        "Write each open-ended answer the way this person would type it into an online "
        "survey form: usually a few words to two sentences, in their own everyday words, "
        "without lists or headings. "
    ),
}


def is_nothing(text: str | None) -> bool:
    """True for empty answers and for answers that only say there is nothing."""
    return text is None or not str(text).strip() or bool(_NOTHING.match(str(text)))


@dataclass(frozen=True)
class RespondentPrompt:
    instrument: SurveyInstrument
    level: str = "political"
    style: str = "plain"

    def __post_init__(self) -> None:
        if self.style not in STYLE:
            raise ValueError(f"unknown style {self.style!r}")

    @property
    def condition(self) -> str:
        return f"{self.level}-{self.style}"

    def messages(self, persona: Persona) -> list[dict]:
        ins = self.instrument
        keys_t = [i.key for i in ins.thermometers]
        keys_o = [i.key for i in ins.open_items]
        therm = "\n".join(f'- "{i.key}": {i.wording}' for i in ins.thermometers)
        open_ = "\n".join(f'- "{i.key}": {i.wording}' for i in ins.open_items)
        schema = "{" + ", ".join([f'"{k}": <integer 0-100>' for k in keys_t]
                                 + [f'"{k}": "<answer or empty>"' for k in keys_o]) + "}"
        system = ("You are taking part in a public opinion survey of American adults in the "
                  "weeks before the November 2024 presidential election between Kamala Harris "
                  "and Donald Trump. Answer every question as the person described below "
                  "would, in the first person.")
        user = (f"About you: {persona.describe(self.level)}\n\n"
                f"Feeling thermometer. {ins.thermometer_intro}\nRate:\n{therm}\n\n"
                f"Open-ended questions. If there is nothing in particular, leave the answer "
                f"empty.\n{open_}\n\n"
                f"{STYLE[self.style]}"
                f"Reply with a single JSON object and nothing else, in this form:\n{schema}")
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class ParsedAnswer:
    ok: bool
    thermometers: dict
    open: dict
    error: str = ""


def _json_block(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:j + 1])
                    except json.JSONDecodeError:
                        break
                    return obj if isinstance(obj, dict) else None
        start = text.find("{", start + 1)
    return None


def parse_answer(text: str, instrument: SurveyInstrument) -> ParsedAnswer:
    """Parse a model reply; ``ok`` only if every thermometer is an integer in [0, 100]."""
    obj = _json_block(text or "")
    if obj is None:
        return ParsedAnswer(False, {}, {}, "no JSON object")
    therm, opn, bad = {}, {}, []
    for item in instrument.thermometers:
        v = obj.get(item.key)
        try:
            f = float(v)
        except (TypeError, ValueError):
            bad.append(item.key)
            continue
        if not (0 <= f <= 100) or f != round(f):
            bad.append(item.key)
            continue
        therm[item.key] = int(f)
    for item in instrument.open_items:
        v = obj.get(item.key, "")
        v = "" if v is None else str(v).strip()
        opn[item.key] = "" if is_nothing(v) else v
    if bad:
        return ParsedAnswer(False, therm, opn, f"invalid thermometers: {bad}")
    return ParsedAnswer(True, therm, opn)


def respondent_document(answers: dict, instrument: SurveyInstrument) -> str:
    """Detector input: the non-empty open-ended answers in instrument order, one per line."""
    lines = [str(answers.get(i.key, "")).strip() for i in instrument.open_items]
    return "\n".join(t for t in lines if t and not is_nothing(t))
