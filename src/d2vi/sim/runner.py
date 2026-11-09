"""Monte Carlo runner: mixtures x audits x estimators, with reproducible seeding.

Seeding. Every replication ``r`` of a cell gets a ``SeedSequence`` built from the root
seed, a CRC32 hash of the cell's *data key* (corpus, detector, contamination rate, size,
mixture) and ``r``. The mixture is drawn from that stream, so every audit design and
every budget evaluated on the same data key sees the same response sets (common random
numbers). The audit and the cross-fitting folds use a child stream that also depends on
the design and the budget. All estimators and outcomes in a replication share the same
response set and the same audit, so estimator comparisons are paired.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..decision.certify import TOSTCertifier
from ..decision.policy import MSEPolicy
from ..estimators.base import EstimationContext, Estimator
from ..sampling.audit import PerfectVerifier, make_design
from ..sampling.mixture import FeaturePool, make_mixture

__all__ = ["CellSpec", "Experiment"]


def _crc(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def _spec_str(spec: Any) -> str:
    if isinstance(spec, dict):
        return spec["name"] + "".join(f"_{k}{v}" for k, v in sorted(spec.items()) if k != "name")
    return str(spec)


@dataclass(frozen=True)
class CellSpec:
    corpus: str
    detector: str
    eps: float
    n: int
    budget: float
    design: Any = "srs"
    mixture: Any = "iid"
    tags: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def data_key(self) -> str:
        return f"{self.corpus}|{self.detector}|{self.eps:g}|{self.n}|{_spec_str(self.mixture)}"

    @property
    def audit_key(self) -> str:
        return f"{self.budget:g}|{_spec_str(self.design)}"

    @property
    def m(self) -> int:
        return max(int(round(self.budget * self.n)), 2)

    def describe(self) -> dict:
        d = asdict(self)
        d["design"] = _spec_str(self.design)
        d["mixture"] = _spec_str(self.mixture)
        d.pop("tags")
        d.update(self.tags)
        d["m"] = self.m
        return d


class Experiment:
    """Run cells of a simulation on one feature pool."""

    def __init__(self, pool: FeaturePool, estimators: list[Estimator],
                 policies: list[MSEPolicy] | None = None,
                 certifiers: list[TOSTCertifier] | None = None,
                 outcomes: list[str] | None = None, root_seed: int = 20261109) -> None:
        self.pool = pool
        self.estimators = list(estimators)
        self.policies = list(policies or [])
        self.certifiers = list(certifiers or [])
        self.outcomes = list(outcomes or pool.outcomes)
        self.root_seed = int(root_seed)
        self._sd = {o: pool.human_sd(o) for o in self.outcomes}

    def _seeds(self, spec: CellSpec, rep: int):
        data_ss = np.random.SeedSequence([self.root_seed, _crc(spec.data_key), rep])
        audit_ss = np.random.SeedSequence([self.root_seed, _crc(spec.data_key), rep,
                                           _crc(spec.audit_key)])
        return np.random.default_rng(data_ss), np.random.default_rng(audit_ss)

    def run_replication(self, spec: CellSpec, rep: int) -> list[dict]:
        rng_data, rng_audit = self._seeds(spec, rep)
        rs = make_mixture(spec.mixture).sample(self.pool, spec.n, spec.eps, rng_data)
        audit = make_design(spec.design).draw(rs.observed, spec.m, rng_audit,
                                              PerfectVerifier(rs.a))
        ctx = EstimationContext(rs.observed, audit, rng_audit)
        base = spec.describe()
        base.update(rep=rep, eps_real=rs.contamination(), m_machine=audit.n_machine,
                    m_human=audit.n_human)
        s, f = rs.operating_point()
        base.update(s=s, f=f)
        rows = []
        for o in self.outcomes:
            truth = rs.human_mean(o)
            deletion_bias = float(np.mean(rs.observed.outcome(o)[rs.observed.flag == 0])) - truth
            common = dict(base, outcome=o, truth=truth, sd_h=self._sd[o],
                          deletion_bias=deletion_bias)
            for est in self.estimators:
                res = est.estimate(ctx, o)
                rows.append({**common, **res.as_row(), "outcome": o})
            for pol in self.policies:
                pr = pol.choose(ctx, o)
                row = {**common, **pr.estimate.as_row(), "outcome": o}
                row["choice"] = pr.choice
                rows.append(row)
            for cert in self.certifiers:
                cr = cert.certify(ctx, o)
                rows.append({**common, "estimator": f"cert_tost{cert.tau_sd:g}", "outcome": o,
                             "point": cr.b_hat, "se": cr.se, "feasible": np.isfinite(cr.b_hat),
                             "certified": cr.certified, "tau": cr.tau, "reason": cr.reason})
        return rows

    def run_cell(self, spec: CellSpec, n_rep: int, n_jobs: int = 1) -> pd.DataFrame:
        if n_jobs == 1:
            rows = [r for rep in range(n_rep) for r in self.run_replication(spec, rep)]
        else:
            chunks = Parallel(n_jobs=n_jobs)(delayed(self.run_replication)(spec, rep)
                                             for rep in range(n_rep))
            rows = [r for chunk in chunks for r in chunk]
        return pd.DataFrame(rows)

    def run(self, specs: list[CellSpec], n_rep: int, n_jobs: int = 1) -> pd.DataFrame:
        return pd.concat([self.run_cell(s, n_rep, n_jobs) for s in specs], ignore_index=True)
