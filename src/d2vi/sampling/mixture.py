"""Build response sets by mixing genuine human and genuine machine documents.

A :class:`FeaturePool` holds the precomputed features of one corpus and one detector:
origin, detector score, flag, length and outcome columns. Mixture samplers draw a
response set of size ``N`` with a given contamination rate and return a
:class:`~d2vi.data.schema.ResponseSet`. All text is real; only the mixing is experimental.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..data.schema import ObservedData, ResponseSet

__all__ = [
    "FeaturePool",
    "MixtureSampler",
    "IIDMixture",
    "AdversarialMixture",
    "ClusteredMixture",
    "MultiGeneratorMixture",
    "make_mixture",
]

OUTCOME_PREFIX = "y_"


class FeaturePool:
    """Documents with precomputed features.

    Required columns: ``a`` (machine share), ``score`` and at least one outcome column
    named ``y_<name>``. Optional: ``flag`` (otherwise ``score > threshold``), ``length``
    (word count, used for the ``loglen`` covariate), ``generator``, ``domain``, ``doc_id``
    and any column listed in ``extra_covariates``.
    """

    def __init__(self, df: pd.DataFrame, threshold: float = 0.5,
                 extra_covariates: tuple[str, ...] = ()) -> None:
        missing = {"a", "score"} - set(df.columns)
        if missing:
            raise ValueError(f"feature pool lacks columns {sorted(missing)}")
        self.df = df.reset_index(drop=True).copy()
        if "flag" not in self.df:
            self.df["flag"] = (self.df["score"] > threshold).astype(int)
        self.outcomes = [c[len(OUTCOME_PREFIX):] for c in self.df.columns
                         if c.startswith(OUTCOME_PREFIX)]
        if not self.outcomes:
            raise ValueError("feature pool has no outcome columns (prefix 'y_')")
        self.extra_covariates = tuple(extra_covariates)
        self._human = np.flatnonzero(self.df["a"].to_numpy() == 0)
        self._machine = np.flatnonzero(self.df["a"].to_numpy() > 0)

    @property
    def human_index(self) -> np.ndarray:
        return self._human

    def machine_index(self, column: str | None = None, value=None) -> np.ndarray:
        if column is None:
            return self._machine
        mask = self.df[column].to_numpy()[self._machine] == value
        return self._machine[mask]

    def human_sd(self, outcome: str) -> float:
        """Standard deviation of an outcome among human documents (for standardised RMSE)."""
        return float(self.df[OUTCOME_PREFIX + outcome].to_numpy()[self._human].std(ddof=1))

    def to_response_set(self, rows: np.ndarray, rng: np.random.Generator) -> ResponseSet:
        rows = rng.permutation(np.asarray(rows, dtype=int))
        sub = self.df.iloc[rows]
        y = {o: sub[OUTCOME_PREFIX + o].to_numpy(dtype=float) for o in self.outcomes}
        cov = {}
        if "length" in sub:
            cov["loglen"] = np.log1p(sub["length"].to_numpy(dtype=float))
        for c in self.extra_covariates:
            cov[c] = sub[c].to_numpy(dtype=float)
        obs = ObservedData(y=y, score=sub["score"].to_numpy(dtype=float),
                           flag=sub["flag"].to_numpy(dtype=int), covariates=cov)
        meta_cols = [c for c in ("doc_id", "generator", "domain", "attack", "language")
                     if c in sub]
        meta = sub[meta_cols].reset_index(drop=True) if meta_cols else None
        return ResponseSet(observed=obs, a=sub["a"].to_numpy(dtype=float), meta=meta)


class MixtureSampler(ABC):
    """Draws a response set of ``n`` records with contamination rate ``eps``."""

    name = "mixture"

    def sample(self, pool: FeaturePool, n: int, eps: float,
               rng: np.random.Generator) -> ResponseSet:
        n_machine = int(round(n * eps))
        n_human = n - n_machine
        humans = self._draw(pool.human_index, n_human, rng, "human")
        machines = self._machine_rows(pool, n_machine, rng)
        return pool.to_response_set(np.concatenate([humans, machines]), rng)

    @staticmethod
    def _draw(index: np.ndarray, k: int, rng: np.random.Generator, what: str) -> np.ndarray:
        if k > index.shape[0]:
            raise ValueError(f"pool has {index.shape[0]} {what} documents, {k} requested")
        return rng.choice(index, size=k, replace=False) if k else np.empty(0, dtype=int)

    @abstractmethod
    def _machine_rows(self, pool: FeaturePool, k: int, rng: np.random.Generator) -> np.ndarray:
        """Choose the machine documents."""


class IIDMixture(MixtureSampler):
    """Machine documents drawn uniformly from the pool."""

    name = "iid"

    def _machine_rows(self, pool, k, rng):
        return self._draw(pool.machine_index(), k, rng, "machine")


class AdversarialMixture(MixtureSampler):
    """Only the hardest-to-detect machine documents (lowest-score ``fraction``)."""

    def __init__(self, fraction: float = 0.5) -> None:
        self.fraction = float(fraction)
        self.name = f"adversarial{int(round(100 * self.fraction))}"

    def _machine_rows(self, pool, k, rng):
        idx = pool.machine_index()
        scores = pool.df["score"].to_numpy()[idx]
        keep = idx[scores <= np.quantile(scores, self.fraction)]
        return self._draw(keep, k, rng, "low-score machine")


class ClusteredMixture(MixtureSampler):
    """Bot-farm arrivals: all machine documents come from one cluster.

    The cluster is a value of ``column`` (for example the generator or the prompt
    template), chosen at random among values with at least ``k`` documents.
    """

    def __init__(self, column: str = "generator") -> None:
        self.column = column
        self.name = f"clustered_{column}"

    def _machine_rows(self, pool, k, rng):
        idx = pool.machine_index()
        values = pool.df[self.column].to_numpy()[idx]
        uniq, counts = np.unique(values, return_counts=True)
        eligible = uniq[counts >= k]
        if eligible.shape[0] == 0:
            raise ValueError(f"no {self.column} value has {k} machine documents")
        chosen = eligible[rng.integers(eligible.shape[0])]
        return self._draw(idx[values == chosen], k, rng, "clustered machine")


class MultiGeneratorMixture(MixtureSampler):
    """Several generators at once, with Dirichlet-distributed shares."""

    def __init__(self, column: str = "generator", concentration: float = 1.0) -> None:
        self.column, self.concentration = column, float(concentration)
        self.name = f"multigen_{column}"

    def _machine_rows(self, pool, k, rng):
        idx = pool.machine_index()
        values = pool.df[self.column].to_numpy()[idx]
        uniq = np.unique(values)
        shares = rng.dirichlet(np.full(uniq.shape[0], self.concentration))
        counts = rng.multinomial(k, shares)
        rows = [self._draw(idx[values == v], c, rng, f"{v} machine")
                for v, c in zip(uniq, counts, strict=True)]
        return np.concatenate(rows) if rows else np.empty(0, dtype=int)


def make_mixture(spec: str | dict) -> MixtureSampler:
    if isinstance(spec, str):
        spec = {"name": spec}
    params = dict(spec)
    name = params.pop("name")
    registry = {
        "iid": IIDMixture,
        "adversarial": AdversarialMixture,
        "clustered": ClusteredMixture,
        "multigen": MultiGeneratorMixture,
    }
    if name not in registry:
        raise KeyError(f"unknown mixture {name!r}; choose from {sorted(registry)}")
    return registry[name](**params)
