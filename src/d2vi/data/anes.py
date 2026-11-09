"""ANES 2024 Time Series: the human records of the survey study (Study 3).

:class:`ANES2024` joins the main data file (CSV, full release of 19 May 2026) with the
"Redacted Open-Ends" Excel file (23 September 2025): one sheet per open-ended variable,
columns ``V240001`` (case id) and the text.

Text cleaning (:func:`clean_open_end`) is applied identically to human and synthetic
answers:

* ``[CHAR(10)]`` and ``[CHAR(13)]`` in the Excel file encode line breaks;
* in interviewer-administered modes the interviewer types the probe "anything else?" as
  ``/AE/`` or ``/AO/`` (questionnaire, CANDLIK and PTYLIK probes) and the respondent's
  "no" after it; markers are removed and segments that only say there is nothing more are
  dropped, so the detector does not see the interview mode.

Respondents are excluded from the human pool when (in this order) they were interviewed on
paper (no open-ended questions), lack a valid value on any of the four thermometers, have
no open-ended text, have an answer with an ANES redaction tag (``[REDACTED ...]``; a
synthetic answer never has one, so the tag would reveal the class), or have an answer with
an unmistakable chatbot hallmark (Markdown bold headings or headings, "as an AI", "language
model"). The last group is evidence of machine text already present in the 2024 web
survey; it is reported, not used as human data.

The ANES files are public but are not redistributed: the human pool stays on the machine
that builds it (``data/private/``, ignored by git).
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..synthetic.instrument import ANES2024 as INSTRUMENT
from ..synthetic.instrument import SurveyInstrument
from ..synthetic.persona import ANES2024_CODEBOOK, PersonaCodebook
from ..synthetic.prompt import is_nothing

__all__ = ["clean_open_end", "has_redaction", "chatbot_hallmark", "ANES2024", "MODES"]

MODES = {1: "face-to-face", 2: "web", 3: "paper", 4: "phone"}
_PROBE = re.compile(r"/?\b(?:ae|ao|oa|tm|ms|wm)\b/", re.IGNORECASE)
_REDACTED = re.compile(r"\[REDACTED[^\]]*\]")
_HALLMARK = re.compile(r"\*\*[A-Z][^*\n]{2,80}\*\*|(?m:^\s*#{2,}\s)|\bAs an AI\b|"
                       r"\blanguage model\b", re.IGNORECASE)


def clean_open_end(text) -> str:
    """Normalise one open-ended answer (see the module docstring)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = str(text).replace("[CHAR(13)]", "").replace("[CHAR(10)]", "\n")
    t = re.sub(r"[ \t]+", " ", t).strip()  # first: the Excel cells are space-padded
    parts = [p.strip(" \t\n.,;") for p in _PROBE.split(t)]
    parts = [p for p in parts if p and not is_nothing(p)]
    t = " ".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def has_redaction(text: str) -> bool:
    return bool(_REDACTED.search(text or ""))


def chatbot_hallmark(text: str) -> bool:
    return bool(_HALLMARK.search(text or ""))


@dataclass
class ANES2024:
    """Respondent table of the ANES 2024 pre-election wave with cleaned open-ended text.

    The table keeps the persona background variables, so synthetic respondents can be
    drawn from the eligible human records (``sample_personas(pool, ...)``).
    """

    csv_path: str | Path
    openends_path: str | Path
    instrument: SurveyInstrument = INSTRUMENT
    codebook: PersonaCodebook = ANES2024_CODEBOOK

    def _main(self) -> pd.DataFrame:
        path = Path(self.csv_path)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
                return pd.read_csv(z.open(name), low_memory=False)
        return pd.read_csv(path, low_memory=False)

    def _openends(self) -> pd.DataFrame:
        book = pd.ExcelFile(self.openends_path)
        out = None
        for item in self.instrument.open_items:
            if item.variable not in book.sheet_names:
                raise KeyError(f"sheet {item.variable} not in {self.openends_path}")
            d = book.parse(item.variable)
            if d.columns[0] != "V240001" or d.shape[1] != 2:
                raise ValueError(f"unexpected layout of sheet {item.variable}")
            if d["V240001"].duplicated().any():
                raise ValueError(f"duplicate case ids in sheet {item.variable}")
            d = d.rename(columns={d.columns[1]: f"raw_{item.key}"})
            out = d if out is None else out.merge(d, on="V240001", how="outer")
        return out

    def respondents(self) -> pd.DataFrame:
        """All pre-election respondents with outcomes, cleaned texts and exclusion flags."""
        m = self._main()
        ins = self.instrument
        df = pd.DataFrame({"V240001": m["V240001"].astype(int),
                           "mode": m["V240002a"].map(MODES), "sample_type": m["V240003"]})
        for t in ins.thermometers:
            v = pd.to_numeric(m[t.variable], errors="coerce")
            df[f"y_{t.key}"] = v.where((v >= 0) & (v <= 100)) / 100.0
        for item in ins.open_items:
            df[f"filter_{item.key}"] = m[item.filter_variable]
        cb = self.codebook
        for var in (cb.age_var, cb.sex_var, cb.race_var, cb.education_var, cb.region_var,
                    cb.party_var, cb.ideology_var):
            df[var] = m[var].to_numpy()
        oe = self._openends()
        df = df.merge(oe, on="V240001", how="left")
        raw_cols = [f"raw_{i.key}" for i in ins.open_items]
        for item in ins.open_items:
            df[f"text_{item.key}"] = df[f"raw_{item.key}"].map(clean_open_end)
        texts = df[[f"text_{i.key}" for i in ins.open_items]]
        df["text"] = ["\n".join(t for t in row if t) for row in texts.itertuples(index=False)]
        raw = df[raw_cols].fillna("").astype(str)
        df["redacted"] = raw.apply(lambda r: any(has_redaction(x) for x in r), axis=1)
        df["hallmark"] = raw.apply(lambda r: any(chatbot_hallmark(x) for x in r), axis=1)
        df = df.drop(columns=raw_cols)
        df["length"] = df["text"].str.split().str.len()
        df["exclusion"] = self._exclusion(df)
        return df

    def _exclusion(self, df: pd.DataFrame) -> pd.Series:
        ys = [f"y_{t.key}" for t in self.instrument.thermometers]
        reason = pd.Series("", index=df.index, dtype=object)
        rules = [("paper interview", df["mode"].eq("paper")),
                 ("missing thermometer", df[ys].isna().any(axis=1)),
                 ("no open-ended text", df["text"].eq("")),
                 ("redaction tag", df["redacted"]),
                 ("chatbot hallmark", df["hallmark"])]
        for name, mask in rules:
            reason = reason.mask(reason.eq("") & mask, name)
        return reason

    def human_pool(self, respondents: pd.DataFrame | None = None) -> pd.DataFrame:
        """Eligible respondents as records of the response set (``a = 0``)."""
        r = self.respondents() if respondents is None else respondents
        pool = r[r["exclusion"].eq("")].copy()
        pool.insert(0, "doc_id", "anes-" + pool["V240001"].astype(str))
        pool["a"] = 0.0
        pool["source"] = "human"
        pool["generator"] = "human"
        return pool.reset_index(drop=True)
