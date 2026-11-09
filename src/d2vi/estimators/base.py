"""Common interface for estimators of the human mean."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.schema import ObservedData
from ..sampling.audit import AuditSample

__all__ = ["EstimateResult", "EstimationContext", "Estimator"]


@dataclass
class EstimateResult:
    """Output of one estimator on one response set and audit.

    An infeasible estimator returns ``feasible=False`` with ``point = nan`` and a
    ``reason``. Estimators never fall back to another estimator silently; only
    :class:`d2vi.decision.policy.SelectionPolicy` may substitute one estimate for another,
    and it records that it did.
    """

    estimator: str
    outcome: str
    point: float
    se: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    feasible: bool = True
    reason: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def infeasible(cls, estimator: str, outcome: str, reason: str) -> EstimateResult:
        return cls(estimator=estimator, outcome=outcome, point=float("nan"), feasible=False,
                   reason=reason)

    def as_row(self) -> dict[str, Any]:
        row = {
            "estimator": self.estimator,
            "outcome": self.outcome,
            "point": self.point,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "feasible": self.feasible,
            "reason": self.reason,
        }
        for k, v in self.extras.items():
            if np.isscalar(v):
                row[f"x_{k}"] = v
        return row


class EstimationContext:
    """One response set (analyst view) and one audit, with a cache for shared fits.

    The propensity model does not depend on the outcome, so it is fitted once per
    context and reused by every outcome and every estimator that asks for the same spec.
    """

    def __init__(self, obs: ObservedData, audit: AuditSample | None,
                 rng: np.random.Generator | None = None) -> None:
        self.obs = obs
        self.audit = audit
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._cache: dict[Any, Any] = {}

    def cached(self, key: Any, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def with_audit(self, audit: AuditSample, rng: np.random.Generator) -> EstimationContext:
        return EstimationContext(self.obs, audit, rng)


class Estimator(ABC):
    """Estimator of the human mean of one outcome."""

    name: str = "estimator"
    needs_audit: bool = True

    @abstractmethod
    def estimate(self, ctx: EstimationContext, outcome: str) -> EstimateResult:
        """Estimate ``mu_H`` for ``outcome``."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
