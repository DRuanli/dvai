"""Containers that separate what the analyst sees from what only the simulator knows.

* :class:`ResponseSet` is one collected response set, the finite population of ``N``
  records. It holds the true origin ``a`` of every record, coded as the machine share in
  ``[0, 1]`` (``0`` = human-written, ``1`` = machine-written, values in between = graded
  or hybrid authorship).
* :class:`ObservedData` is the analyst's view of the same records: outcomes, detector
  score, detector flag and covariates, but no origin. Estimators only ever receive an
  :class:`ObservedData` plus an audit (:class:`d2vi.sampling.audit.AuditSample`), so
  they cannot read the origin of unaudited records by construction.

The estimand used throughout is the finite-population human mean

    mu_H = sum_i (1 - a_i) y_i / sum_i (1 - a_i),

which equals the mean over human records when ``a`` is binary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["ObservedData", "ResponseSet"]


@dataclass(frozen=True)
class ObservedData:
    """Analyst-visible data for ``N`` records.

    Attributes
    ----------
    y:
        Mapping from outcome name to an array of length ``N``.
    score:
        Continuous detector score (higher = more machine-like).
    flag:
        Binary detector flag ``D`` (1 = flagged as machine-written).
    covariates:
        Additional observable covariates, e.g. ``"loglen"`` or a style stratum ``"G"``.
    """

    y: dict[str, np.ndarray]
    score: np.ndarray
    flag: np.ndarray
    covariates: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.score.shape[0]
        if self.flag.shape[0] != n:
            raise ValueError("flag and score must have the same length")
        for name, values in self.y.items():
            if values.shape[0] != n:
                raise ValueError(f"outcome {name!r} has length {values.shape[0]}, expected {n}")
        for name, values in self.covariates.items():
            if values.shape[0] != n:
                raise ValueError(f"covariate {name!r} has length {values.shape[0]}, expected {n}")

    @property
    def n(self) -> int:
        return int(self.score.shape[0])

    @property
    def outcomes(self) -> list[str]:
        return list(self.y)

    def outcome(self, name: str) -> np.ndarray:
        try:
            return self.y[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"unknown outcome {name!r}; available: {self.outcomes}") from exc

    def feature(self, name: str) -> np.ndarray:
        """Return an observable feature by name (``score``, ``flag`` or a covariate)."""
        if name == "score":
            return self.score.astype(float)
        if name == "flag":
            return self.flag.astype(float)
        if name in self.covariates:
            return np.asarray(self.covariates[name], dtype=float)
        raise KeyError(f"unknown feature {name!r}")

    def design_matrix(self, names: tuple[str, ...], standardize: bool = True) -> np.ndarray:
        """Stack observable features into an ``N x p`` matrix.

        Standardisation uses all ``N`` records, which is allowed because the features are
        observed for every record (phase-one data).
        """
        cols = [self.feature(n) for n in names]
        x = np.column_stack(cols) if cols else np.empty((self.n, 0))
        if standardize and x.size:
            sd = x.std(axis=0)
            sd[sd == 0] = 1.0
            x = (x - x.mean(axis=0)) / sd
        return x


@dataclass(frozen=True)
class ResponseSet:
    """A response set with ground truth, produced by a mixture sampler.

    Attributes
    ----------
    observed:
        The analyst's view.
    a:
        True machine share of each record (``0`` human, ``1`` machine).
    meta:
        Optional per-record metadata (document id, generator, domain, ...).
    """

    observed: ObservedData
    a: np.ndarray
    meta: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.a.shape[0] != self.observed.n:
            raise ValueError("origin vector and observed data differ in length")
        if np.any((self.a < 0) | (self.a > 1)):
            raise ValueError("origin shares must lie in [0, 1]")

    @property
    def n(self) -> int:
        return self.observed.n

    def contamination(self) -> float:
        """Finite-population contamination rate, the mean machine share."""
        return float(np.mean(self.a))

    def human_mean(self, outcome: str) -> float:
        """Finite-population human mean, the estimand ``mu_H``."""
        w = 1.0 - self.a
        return float(np.sum(w * self.observed.outcome(outcome)) / np.sum(w))

    def machine_mean(self, outcome: str) -> float:
        w = self.a
        if np.sum(w) == 0:
            return float("nan")
        return float(np.sum(w * self.observed.outcome(outcome)) / np.sum(w))

    def operating_point(self) -> tuple[float, float]:
        """Return ``(s, f)``: sensitivity and false-positive rate of the flag."""
        d = self.observed.flag.astype(float)
        s = float(np.sum(self.a * d) / np.sum(self.a)) if np.sum(self.a) > 0 else float("nan")
        f = float(np.sum((1 - self.a) * d) / np.sum(1 - self.a))
        return s, f
