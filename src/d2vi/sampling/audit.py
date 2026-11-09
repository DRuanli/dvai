"""Audit designs: which records are sent for verification of their origin.

An audit design selects records using observable information only (detector scores and
covariates). A :class:`Verifier` then reveals the origin of the selected records. Keeping
the two steps apart is what makes the audit a probability sample with known inclusion
probabilities, the design condition (D1) on which RC-PPI's guarantee rests.

Every design returns an :class:`AuditSample` with exactly ``m`` records, so comparisons
between designs are budget-matched by construction.

Two sampling kinds are represented:

* ``"stratified"``: simple random sampling without replacement within strata, with
  inclusion probability ``m_h / N_h`` in stratum ``h``;
* ``"pps"``: fixed-size unequal-probability (systematic PPS) sampling.

The variance estimator in :mod:`d2vi.inference.variance` handles both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np

from ..data.schema import ObservedData

__all__ = [
    "AuditSample",
    "Verifier",
    "PerfectVerifier",
    "NoisyVerifier",
    "AuditDesign",
    "SimpleRandom",
    "ProportionalStratified",
    "TopDecileSplit",
    "TwoStageNeyman",
    "AdaptiveNeyman",
    "ScoreRankPPS",
    "make_design",
]


@dataclass(frozen=True)
class AuditSample:
    """A drawn audit.

    Attributes
    ----------
    idx:
        Positions of audited records in the response set (sorted, unique).
    labels:
        Verified machine share of each audited record.
    rho:
        First-order inclusion probability of each audited record.
    strata:
        Design stratum of each audited record.
    stratum_sizes:
        Population size ``N_h`` of every design stratum.
    kind:
        ``"stratified"`` or ``"pps"``.
    design:
        Name of the design that produced the sample.
    weights:
        Optional weight override (used by bootstrap replicates); defaults to ``1 / rho``.
    """

    idx: np.ndarray
    labels: np.ndarray
    rho: np.ndarray
    strata: np.ndarray
    stratum_sizes: dict
    kind: str
    design: str
    weights: np.ndarray | None = None

    @property
    def m(self) -> int:
        return int(self.idx.shape[0])

    @property
    def w(self) -> np.ndarray:
        return self.weights if self.weights is not None else 1.0 / self.rho

    @property
    def n_machine(self) -> float:
        """Number of audited machine records (sum of shares for graded labels)."""
        return float(np.sum(self.labels))

    @property
    def n_human(self) -> float:
        return float(np.sum(1.0 - self.labels))

    def stratum_counts(self) -> dict:
        keys, counts = np.unique(self.strata, return_counts=True)
        return dict(zip(keys.tolist(), counts.tolist(), strict=True))

    def subset(self, mask: np.ndarray, weights: np.ndarray | None = None) -> AuditSample:
        """Return the audit restricted to ``mask`` (optionally with new weights)."""
        return replace(
            self,
            idx=self.idx[mask],
            labels=self.labels[mask],
            rho=self.rho[mask],
            strata=self.strata[mask],
            weights=None if weights is None else weights[mask],
        )


class Verifier(ABC):
    """Reveals the origin of audited records."""

    @abstractmethod
    def verify(self, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return the verified machine share for records ``idx``."""


class PerfectVerifier(Verifier):
    """Returns the true origin (assumption D2)."""

    def __init__(self, a: np.ndarray) -> None:
        self._a = np.asarray(a, dtype=float)

    def verify(self, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self._a[idx].copy()


class NoisyVerifier(Verifier):
    """Random verification error, independent of the outcome given the origin.

    ``phi0`` is the probability that a human record is labelled machine; ``phi1`` the
    probability that a machine record is labelled human (Proposition 6). Graded labels are
    first rounded at 0.5.
    """

    def __init__(self, a: np.ndarray, phi0: float = 0.0, phi1: float = 0.0) -> None:
        self._a = (np.asarray(a, dtype=float) >= 0.5).astype(float)
        self.phi0, self.phi1 = float(phi0), float(phi1)

    def verify(self, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        true = self._a[idx]
        u = rng.random(idx.shape[0])
        flip = np.where(true == 1, u < self.phi1, u < self.phi0)
        return np.where(flip, 1.0 - true, true)


@dataclass(frozen=True)
class _Selection:
    idx: np.ndarray
    rho: np.ndarray
    strata: np.ndarray
    stratum_sizes: dict
    kind: str


def _score_quantile_strata(score: np.ndarray, n_strata: int) -> np.ndarray:
    """Assign records to ``n_strata`` strata of (nearly) equal size by score rank."""
    order = np.argsort(score, kind="stable")
    strata = np.empty(score.shape[0], dtype=int)
    strata[order] = np.minimum((np.arange(score.shape[0]) * n_strata) // score.shape[0],
                               n_strata - 1)
    return strata


def _largest_remainder(shares: np.ndarray, m: int, caps: np.ndarray,
                       floor: np.ndarray | None = None) -> np.ndarray:
    """Integer allocation proportional to ``shares`` summing to ``m``, respecting caps."""
    shares = np.asarray(shares, dtype=float)
    caps = np.asarray(caps, dtype=int)
    alloc = np.zeros_like(caps) if floor is None else np.minimum(floor, caps).astype(int)
    remaining = m - int(alloc.sum())
    if remaining < 0:
        raise ValueError("floor allocation exceeds the budget")
    while remaining > 0:
        open_ = alloc < caps
        if not open_.any():
            raise ValueError("budget exceeds population size")
        s = np.where(open_, shares, 0.0)
        if s.sum() == 0:
            s = open_.astype(float)
        target = alloc + remaining * s / s.sum()
        add = np.minimum(np.floor(target).astype(int) - alloc, caps - alloc)
        add = np.maximum(add, 0)
        if add.sum() == 0:
            frac = np.where(open_, target - np.floor(target), -1.0)
            add = np.zeros_like(alloc)
            add[int(np.argmax(frac))] = 1
        add = np.minimum(add, remaining)
        alloc = alloc + add
        remaining = m - int(alloc.sum())
    return alloc


def _stratified_srswor(strata: np.ndarray, alloc: dict, rng: np.random.Generator,
                       preselected: dict | None = None) -> _Selection:
    """Draw ``alloc[h]`` records without replacement in every stratum ``h``."""
    chosen = []
    sizes = {}
    for h in np.unique(strata):
        members = np.flatnonzero(strata == h)
        sizes[int(h)] = int(members.shape[0])
        m_h = int(alloc.get(int(h), 0))
        already = np.asarray(preselected.get(int(h), []), dtype=int) if preselected else \
            np.empty(0, dtype=int)
        need = m_h - already.shape[0]
        if need < 0:
            raise ValueError("preselected records exceed the stratum allocation")
        pool = np.setdiff1d(members, already, assume_unique=False)
        extra = rng.choice(pool, size=need, replace=False) if need > 0 else np.empty(0, int)
        chosen.append(np.concatenate([already, extra]))
    idx = np.sort(np.concatenate(chosen)).astype(int)
    h_of = strata[idx]
    rho = np.array([alloc[int(h)] / sizes[int(h)] for h in h_of], dtype=float)
    return _Selection(idx=idx, rho=rho, strata=h_of, stratum_sizes=sizes, kind="stratified")


class AuditDesign(ABC):
    """Base class: select records from observables, then verify them."""

    name: str = "design"

    def draw(self, obs: ObservedData, m: int, rng: np.random.Generator,
             verifier: Verifier) -> AuditSample:
        if not 0 < m <= obs.n:
            raise ValueError(f"audit size m={m} must be in (0, N={obs.n}]")
        sel = self._select(obs, m, rng, verifier)
        if sel.idx.shape[0] != m:
            raise RuntimeError(f"{self.name} returned {sel.idx.shape[0]} records, expected {m}")
        if sel.kind == "stratified":
            counts = {h: int(np.sum(sel.strata == h)) for h in sel.stratum_sizes}
            empty = [h for h, c in counts.items() if c == 0]
            if empty:  # positivity (D1): every record needs a positive inclusion probability
                raise RuntimeError(f"{self.name}: strata {empty} received no audit records")
        labels = verifier.verify(sel.idx, rng)
        return AuditSample(idx=sel.idx, labels=labels, rho=sel.rho, strata=sel.strata,
                           stratum_sizes=sel.stratum_sizes, kind=sel.kind, design=self.name)

    @abstractmethod
    def _select(self, obs: ObservedData, m: int, rng: np.random.Generator,
                verifier: Verifier) -> _Selection:
        """Choose the records. Only adaptive designs may call ``verifier``."""


class SimpleRandom(AuditDesign):
    """Simple random sampling without replacement."""

    name = "srs"

    def _select(self, obs, m, rng, verifier):
        strata = np.zeros(obs.n, dtype=int)
        return _stratified_srswor(strata, {0: m}, rng)


class ProportionalStratified(AuditDesign):
    """Stratify by score quantiles, allocate proportionally to stratum size."""

    def __init__(self, n_strata: int = 10) -> None:
        self.n_strata = int(n_strata)
        self.name = f"prop{self.n_strata}"

    def _select(self, obs, m, rng, verifier):
        # at least two audited records per stratum, so that every record has a positive
        # inclusion probability and every stratum contributes to the variance estimate
        h_eff = max(1, min(self.n_strata, m // 2))
        strata = _score_quantile_strata(obs.score, h_eff)
        keys, sizes = np.unique(strata, return_counts=True)
        floor = np.minimum(np.full_like(sizes, 2 if m >= 2 * h_eff else 1), sizes)
        alloc = _largest_remainder(sizes.astype(float), m, sizes, floor=floor)
        return _stratified_srswor(strata, dict(zip(keys.tolist(), alloc.tolist(),
                                                   strict=True)), rng)


class TopDecileSplit(AuditDesign):
    """The original design: a share of the budget from the top score fraction ``q``.

    With the defaults, half of the audit comes from the top decile of detector scores and
    half from the remaining 90%.
    """

    def __init__(self, q: float = 0.10, top_share: float = 0.5) -> None:
        self.q, self.top_share = float(q), float(top_share)
        self.name = f"top{int(round(100 * self.q))}split"

    def _select(self, obs, m, rng, verifier):
        n = obs.n
        n_top = max(1, int(round(self.q * n)))
        order = np.argsort(-obs.score, kind="stable")
        strata = np.zeros(n, dtype=int)
        strata[order[:n_top]] = 1
        m_top = min(int(round(m * self.top_share)), n_top)
        m_rest = m - m_top
        if m_rest > n - n_top:
            m_rest = n - n_top
            m_top = m - m_rest
        return _stratified_srswor(strata, {1: m_top, 0: m_rest}, rng)


class TwoStageNeyman(AuditDesign):
    """Pilot audit, then Neyman allocation of the remaining budget across score strata.

    The pilot (``pilot_frac`` of the budget, proportional allocation) estimates the
    machine rate ``p_h`` in each stratum. The rest of the budget is allocated with
    ``m_h`` proportional to ``N_h * sqrt(p_h (1 - p_h))`` (Proposition 4 with the
    outcome-agnostic proxy ``S_rh ~ sqrt(p_h (1 - p_h))``), with at least
    ``min_per_stratum`` records per stratum. Within a stratum the pilot and second-stage
    records together form a simple random sample.
    """

    def __init__(self, n_strata: int = 5, pilot_frac: float = 0.3,
                 min_per_stratum: int = 2) -> None:
        self.n_strata = int(n_strata)
        self.pilot_frac = float(pilot_frac)
        self.min_per_stratum = int(min_per_stratum)
        self.name = f"neyman{self.n_strata}"

    def _select(self, obs, m, rng, verifier):
        h_eff = max(1, min(self.n_strata, m // max(self.min_per_stratum, 1)))
        strata = _score_quantile_strata(obs.score, h_eff)
        keys, sizes = np.unique(strata, return_counts=True)
        floor = np.minimum(np.full_like(sizes, self.min_per_stratum), sizes)
        m_pilot = max(int(round(self.pilot_frac * m)), int(floor.sum()))
        m_pilot = min(m_pilot, m)
        pilot_alloc = _largest_remainder(sizes.astype(float), m_pilot, sizes, floor=floor)
        pilot = _stratified_srswor(strata, dict(zip(keys.tolist(), pilot_alloc.tolist(),
                                                    strict=True)), rng)
        labels = verifier.verify(pilot.idx, rng)
        p_hat = np.empty(keys.shape[0])
        for j, h in enumerate(keys):
            in_h = pilot.strata == h
            k, n_h = float(labels[in_h].sum()), float(in_h.sum())
            p_hat[j] = (k + 0.5) / (n_h + 1.0)  # Jeffreys-type smoothing
        shares = sizes * np.sqrt(p_hat * (1 - p_hat))
        alloc = _largest_remainder(shares, m, sizes, floor=pilot_alloc)
        pre = {int(h): pilot.idx[pilot.strata == h] for h in keys}
        return _stratified_srswor(strata, dict(zip(keys.tolist(), alloc.tolist(), strict=True)),
                                  rng, preselected=pre)


class AdaptiveNeyman(AuditDesign):
    """Two-stage Neyman allocation with a feasibility constraint (Propositions 3-4).

    Strata are fixed by score quantile cut points (default: bottom 50%, 50-80%, 80-90%,
    top 10%). A proportional pilot (``pilot_frac`` of the budget, at least
    ``min_per_stratum`` per stratum) estimates the machine rate ``p_h`` of each stratum
    with Jeffreys smoothing. The remaining budget follows the Neyman shares
    ``N_h sqrt(p_h (1 - p_h))``, subject to an expected audited machine count of at least
    ``k_target``: while the constraint fails, one record moves from the stratum with the
    lowest ``p_h`` (above its pilot floor) to the stratum with the highest ``p_h`` (ties go
    to the higher-score stratum). At low contamination the design therefore behaves like
    the top-decile split, and at high contamination like Neyman allocation. Pilot and
    second-stage records form a simple random sample within each stratum.
    """

    def __init__(self, cuts: tuple[float, ...] = (0.5, 0.8, 0.9), pilot_frac: float = 0.3,
                 min_per_stratum: int = 2, k_target: float = 12.0) -> None:
        self.cuts = tuple(float(c) for c in cuts)
        self.pilot_frac = float(pilot_frac)
        self.min_per_stratum = int(min_per_stratum)
        self.k_target = float(k_target)
        self.name = "adaptive"

    def _strata(self, score: np.ndarray, m: int) -> np.ndarray:
        n = score.shape[0]
        ranks = np.empty(n)
        ranks[np.argsort(score, kind="stable")] = (np.arange(n) + 0.5) / n
        cuts = list(self.cuts)
        while cuts and (len(cuts) + 1) * self.min_per_stratum > m:
            cuts.pop(0)  # merge the lowest strata when the budget is very small
        return np.searchsorted(np.asarray(cuts), ranks, side="right")

    def _select(self, obs, m, rng, verifier):
        strata = self._strata(obs.score, m)
        keys, sizes = np.unique(strata, return_counts=True)
        floor = np.minimum(np.full_like(sizes, self.min_per_stratum), sizes)
        m_pilot = min(max(int(round(self.pilot_frac * m)), int(floor.sum())), m)
        pilot_alloc = _largest_remainder(sizes.astype(float), m_pilot, sizes, floor=floor)
        pilot = _stratified_srswor(strata, dict(zip(keys.tolist(), pilot_alloc.tolist(),
                                                    strict=True)), rng)
        labels = verifier.verify(pilot.idx, rng)
        p_hat = np.empty(keys.shape[0])
        for j, h in enumerate(keys):
            in_h = pilot.strata == h
            p_hat[j] = (float(labels[in_h].sum()) + 0.5) / (float(in_h.sum()) + 1.0)
        alloc = _largest_remainder(sizes * np.sqrt(p_hat * (1 - p_hat)), m, sizes,
                                   floor=pilot_alloc)
        order_lo = np.lexsort((keys, p_hat))            # lowest p first, then lower score
        order_hi = np.lexsort((-keys, -p_hat))          # highest p first, then higher score
        for _ in range(m):
            if float(np.dot(alloc, p_hat)) >= self.k_target:
                break
            donor = next((j for j in order_lo if alloc[j] > pilot_alloc[j]), None)
            receiver = next((j for j in order_hi if alloc[j] < sizes[j]), None)
            if donor is None or receiver is None or p_hat[receiver] <= p_hat[donor]:
                break
            alloc[donor] -= 1
            alloc[receiver] += 1
        pre = {int(h): pilot.idx[pilot.strata == h] for h in keys}
        return _stratified_srswor(strata, dict(zip(keys.tolist(), alloc.tolist(), strict=True)),
                                  rng, preselected=pre)


class ScoreRankPPS(AuditDesign):
    """Fixed-size systematic PPS with size measure ``floor + rank(score)/N``.

    Inclusion probabilities rise smoothly with the detector score, a label-free stand-in
    for uncertainty-driven (active) sampling.
    """

    def __init__(self, floor: float = 0.25) -> None:
        self.floor = float(floor)
        self.name = f"pps{int(round(100 * self.floor))}"

    def _select(self, obs, m, rng, verifier):
        n = obs.n
        ranks = np.empty(n)
        ranks[np.argsort(obs.score, kind="stable")] = (np.arange(n) + 0.5) / n
        x = self.floor + ranks
        rho = _capped_inclusion(x, m)
        order = rng.permutation(n)
        cum = np.concatenate([[0.0], np.cumsum(rho[order])])
        points = rng.random() + np.arange(m)
        pos = np.searchsorted(cum, points, side="right") - 1
        idx = np.sort(order[pos])
        if np.unique(idx).shape[0] != m:  # pragma: no cover - guarded by capping
            raise RuntimeError("systematic PPS selected a record twice")
        return _Selection(idx=idx, rho=rho[idx], strata=np.zeros(m, dtype=int),
                          stratum_sizes={0: n}, kind="pps")


def _capped_inclusion(x: np.ndarray, m: int) -> np.ndarray:
    """Inclusion probabilities proportional to ``x`` with sum ``m`` and maximum 1."""
    x = np.asarray(x, dtype=float)
    rho = np.zeros_like(x)
    capped = np.zeros(x.shape[0], dtype=bool)
    for _ in range(x.shape[0]):
        free = ~capped
        budget = m - capped.sum()
        rho[free] = budget * x[free] / x[free].sum()
        rho[capped] = 1.0
        newly = free & (rho >= 1.0)
        if not newly.any():
            break
        capped |= newly
    return np.clip(rho, 0.0, 1.0)


def make_design(spec: str | dict) -> AuditDesign:
    """Build a design from a short name or a ``{"name": ..., **params}`` mapping."""
    if isinstance(spec, str):
        spec = {"name": spec}
    params = dict(spec)
    name = params.pop("name")
    registry = {
        "srs": SimpleRandom,
        "proportional": ProportionalStratified,
        "top_decile": TopDecileSplit,
        "neyman": TwoStageNeyman,
        "adaptive": AdaptiveNeyman,
        "pps": ScoreRankPPS,
    }
    if name not in registry:
        raise KeyError(f"unknown design {name!r}; choose from {sorted(registry)}")
    return registry[name](**params)
