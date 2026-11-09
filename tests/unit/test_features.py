import pandas as pd

from d2vi.features.outcomes import split_halves
from d2vi.features.pipeline import cross_domain_threshold


def test_split_halves_partitions_words():
    a, b = split_halves("one two three four five")
    assert a.split() + b.split() == ["one", "two", "three", "four", "five"]
    assert len(a.split()) == 2


def test_cross_domain_threshold_ignores_target_domain_and_machines():
    df = pd.DataFrame({"domain": ["x", "x", "y", "y", "z"], "a": [0, 1, 0, 1, 0],
                       "score_raw": [100.0, 5.0, 1.0, 50.0, 3.0]})
    thr = cross_domain_threshold(df, q=0.5)
    # threshold for x uses the humans of y and z only: median(1, 3) = 2
    assert thr.iloc[0] == 2.0
