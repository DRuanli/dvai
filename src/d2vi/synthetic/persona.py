"""Personas for synthetic survey respondents ("silicon sampling", Argyle et al., 2023).

A persona is built from the background variables of a real ANES respondent, so that the
synthetic respondents have the demographic make-up of the survey they contaminate. Two
conditioning levels are used as separate generator conditions:

* ``"demographic"`` - age, sex, race and ethnicity, education, census region;
* ``"political"``  - the same plus party identification and ideology.

Codes follow the ANES 2024 codebook. :meth:`PersonaCodebook.validate` refuses a data file
containing a code that the codebook does not map (other than the ANES missing codes),
so that a codebook change cannot silently produce wrong personas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["Persona", "PersonaCodebook", "ANES2024_CODEBOOK", "sample_personas"]

MISSING = {-9, -8, -7, -6, -5, -4, -3, -2, -1}
NO_IDEOLOGY = "not thought about"


@dataclass(frozen=True)
class Persona:
    source_id: str
    age: int | None
    sex: str | None
    race: str | None
    education: str | None
    region: str | None
    party_id: str | None = None
    ideology: str | None = None

    def describe(self, level: str = "demographic") -> str:
        """First-person background, as in the prompts of silicon-sampling studies."""
        if level not in ("demographic", "political"):
            raise ValueError(f"unknown conditioning level {level!r}")
        parts = []
        if self.age is not None and self.sex is not None:
            parts.append(f"I am a {self.age}-year-old {self.sex}.")
        elif self.age is not None:
            parts.append(f"I am {self.age} years old.")
        if self.race:
            parts.append(f"Racially or ethnically, I identify as {self.race}.")
        if self.education:
            parts.append(f"My highest level of education is {self.education}.")
        if self.region:
            parts.append(f"I live in the {self.region} of the United States.")
        if level == "political":
            if self.party_id:
                parts.append(f"Politically, I think of myself as {self.party_id}.")
            if self.ideology == NO_IDEOLOGY:
                parts.append("I haven't thought much about whether I am liberal or "
                             "conservative.")
            elif self.ideology:
                parts.append(f"I would describe my political views as {self.ideology}.")
        return " ".join(parts)


@dataclass(frozen=True)
class PersonaCodebook:
    """ANES variable names and value labels used to build personas."""

    id_var: str
    age_var: str
    sex_var: str
    race_var: str
    education_var: str
    region_var: str
    party_var: str
    ideology_var: str
    sex: dict = field(default_factory=dict)
    race: dict = field(default_factory=dict)
    education: dict = field(default_factory=dict)
    region: dict = field(default_factory=dict)
    party: dict = field(default_factory=dict)
    ideology: dict = field(default_factory=dict)

    def _maps(self) -> dict:
        return {self.sex_var: self.sex, self.race_var: self.race,
                self.education_var: self.education, self.region_var: self.region,
                self.party_var: self.party, self.ideology_var: self.ideology}

    def validate(self, df: pd.DataFrame) -> None:
        missing_cols = [v for v in (self.id_var, self.age_var, *self._maps()) if v not in df]
        if missing_cols:
            raise KeyError(f"columns not in data: {missing_cols}")
        for var, labels in self._maps().items():
            codes = set(pd.to_numeric(df[var], errors="coerce").dropna().astype(int))
            unknown = sorted(codes - set(labels) - MISSING)
            if unknown:
                raise ValueError(f"{var}: codes {unknown} are not in the persona codebook")

    @staticmethod
    def _label(labels: dict, value) -> str | None:
        v = pd.to_numeric(value, errors="coerce")
        if pd.isna(v) or int(v) in MISSING:
            return None
        return labels.get(int(v))

    def persona(self, row: pd.Series) -> Persona:
        age = pd.to_numeric(row[self.age_var], errors="coerce")
        age = None if pd.isna(age) or int(age) in MISSING else int(age)
        return Persona(source_id=str(row[self.id_var]), age=age,
                       sex=self._label(self.sex, row[self.sex_var]),
                       race=self._label(self.race, row[self.race_var]),
                       education=self._label(self.education, row[self.education_var]),
                       region=self._label(self.region, row[self.region_var]),
                       party_id=self._label(self.party, row[self.party_var]),
                       ideology=self._label(self.ideology, row[self.ideology_var]))


ANES2024_CODEBOOK = PersonaCodebook(
    id_var="V240001", age_var="V241458x", sex_var="V241550", race_var="V241501x",
    education_var="V241465x", region_var="V243007", party_var="V241227x",
    ideology_var="V241177",
    sex={1: "man", 2: "woman"},
    race={1: "White", 2: "Black or African American", 3: "Hispanic or Latino",
          4: "Asian American, Native Hawaiian or Pacific Islander",
          5: "Native American or another race", 6: "multiracial"},
    education={1: "less than a high school diploma", 2: "a high school diploma",
               3: "some college or a two-year degree", 4: "a bachelor's degree",
               5: "a graduate degree"},
    region={1: "Northeast", 2: "Midwest", 3: "South", 4: "West"},
    party={1: "a strong Democrat", 2: "a Democrat, but not a very strong one",
           3: "an independent who leans toward the Democratic Party", 4: "an independent",
           5: "an independent who leans toward the Republican Party",
           6: "a Republican, but not a very strong one", 7: "a strong Republican"},
    ideology={1: "extremely liberal", 2: "liberal", 3: "slightly liberal",
              4: "moderate, middle of the road", 5: "slightly conservative",
              6: "conservative", 7: "extremely conservative",
              99: NO_IDEOLOGY},
)


def sample_personas(df: pd.DataFrame, n: int, codebook: PersonaCodebook = ANES2024_CODEBOOK,
                    seed: int = 0) -> list[Persona]:
    """Draw ``n`` personas from the respondents in ``df`` (with replacement, seeded)."""
    codebook.validate(df)
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, len(df), size=n)
    return [codebook.persona(df.iloc[i]) for i in rows]
