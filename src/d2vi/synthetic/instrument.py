"""Survey instrument for synthetic respondents (Study 3, ANES 2024).

The items are those of the ANES 2024 Time Series pre-election interview that pair a
closed-ended social measurement with open-ended text: four feeling thermometers (the
outcomes) and eight like/dislike questions about the two presidential candidates and the
two parties (the text a detector sees). Each open-ended question is asked after a yes/no
filter ("Is there anything ...?"); a respondent who answers "no" has no text for it.

Variable names follow the ANES 2024 codebook (full release, 19 May 2026). Wording follows
the web version of the 2024 pre-election questionnaire (8 August 2024): CANDLIK_* (pp.
42-47), THERMPRE_* (pp. 59-66), PTYLIK_* (pp. 66-70); the follow-up "(What is that?)"
appears without parentheses on the web. Checked on 9 November 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SurveyItem", "SurveyInstrument", "ANES2024"]

# Web version of THERMPRE_THINTRO (questionnaire of 8 Aug 2024, p. 60), without the
# graphic and the "don't recognize" clause (synthetic respondents rate every item)
THERMOMETER_INTRO = (
    "We would like to get your feelings toward some of our political leaders and other people "
    "who are in the news these days. We will show the name of a person and we'd like you to "
    "rate that person using something we call the feeling thermometer. Ratings between 50 "
    "degrees and 100 degrees mean that you feel favorable and warm toward the person. Ratings "
    "between 0 degrees and 50 degrees mean that you don't feel favorable toward the person and "
    "that you don't care too much for that person. You would rate the person at the 50 degree "
    "mark if you don't feel particularly warm or cold toward the person. The two parties are "
    "rated on the same thermometer."
)
# web versions of CANDLIK and PTYLIK intros (pp. 42, 66)
OPEN_INTRO = ("We would like to ask you about the good and bad points of the major candidates "
              "for President, and about the good and bad points of the two national parties.")


@dataclass(frozen=True)
class SurveyItem:
    """One question. ``kind`` is ``"thermometer"`` (0-100) or ``"open"`` (filtered text)."""

    key: str
    variable: str
    kind: str
    wording: str
    filter_variable: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("thermometer", "open"):
            raise ValueError(f"unknown item kind {self.kind!r}")
        if self.kind == "open" and self.filter_variable is None:
            raise ValueError(f"open item {self.key} needs its yes/no filter variable")


@dataclass(frozen=True)
class SurveyInstrument:
    name: str
    items: tuple[SurveyItem, ...]
    thermometer_intro: str = THERMOMETER_INTRO
    open_intro: str = OPEN_INTRO
    wording_checked: bool = False
    notes: dict = field(default_factory=dict)

    @property
    def thermometers(self) -> tuple[SurveyItem, ...]:
        return tuple(i for i in self.items if i.kind == "thermometer")

    @property
    def open_items(self) -> tuple[SurveyItem, ...]:
        return tuple(i for i in self.items if i.kind == "open")

    def item(self, key: str) -> SurveyItem:
        for i in self.items:
            if i.key == key:
                return i
        raise KeyError(key)


def _open(key, var, filt, wording):
    return SurveyItem(key, var, "open", wording, filter_variable=filt)


ANES2024 = SurveyInstrument(
    name="ANES 2024 Time Series, pre-election (likes/dislikes and thermometers)",
    wording_checked=True,
    items=(
        SurveyItem("therm_harris", "V241156", "thermometer", "Kamala Harris"),
        SurveyItem("therm_trump", "V241157", "thermometer", "Donald Trump"),
        SurveyItem("therm_dem_party", "V241166", "thermometer", "the Democratic Party"),
        SurveyItem("therm_rep_party", "V241167", "thermometer", "the Republican Party"),
        _open("like_harris", "V241110", "V241109",
              "Is there anything in particular about Kamala Harris that might make you want "
              "to vote for her? What is that?"),
        _open("dislike_harris", "V241112", "V241111",
              "Is there anything in particular about Kamala Harris that might make you want "
              "to vote against her? What is that?"),
        _open("like_trump", "V241114", "V241113",
              "Is there anything in particular about Donald Trump that might make you want to "
              "vote for him? What is that?"),
        _open("dislike_trump", "V241116", "V241115",
              "Is there anything in particular about Donald Trump that might make you want to "
              "vote against him? What is that?"),
        _open("like_dem_party", "V241170", "V241169",
              "Is there anything in particular that you like about the Democratic party? "
              "What is that?"),
        _open("dislike_dem_party", "V241172", "V241171",
              "Is there anything in particular that you don't like about the Democratic "
              "party? What is that?"),
        _open("like_rep_party", "V241174", "V241173",
              "Is there anything in particular that you like about the Republican party? "
              "What is that?"),
        _open("dislike_rep_party", "V241176", "V241175",
              "Is there anything in particular that you don't like about the Republican "
              "party? What is that?"),
    ),
)
