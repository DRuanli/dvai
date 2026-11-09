"""Cross-fitted model of the machine-origin propensity ``eta(x) = P(A = 1 | x)``.

The model is fitted on the audit only, with design weights, and cross-fitted so that the
prediction used in an audited record's residual never comes from a model that saw that
record (Zrnic and Candes, 2024). Records outside the audit receive the average of the
fold models, which makes the estimator the average of ``K`` fold-wise difference
estimators (see Section 3.5 of the revised paper and ``tests/theory``).

Graded labels ``a in [0, 1]`` are handled by the standard trick of duplicating each record
as a machine row with weight ``w a`` and a human row with weight ``w (1 - a)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..data.schema import ObservedData
from ..sampling.audit import AuditSample

__all__ = ["PropensityFit", "CrossFitPropensity"]


@dataclass(frozen=True)
class PropensityFit:
    feasible: bool
    reason: str = ""
    eta_all: np.ndarray | None = None  # fold-averaged prediction for every record
    eta_oof: np.ndarray | None = None  # out-of-fold prediction for each audited record
    folds: np.ndarray | None = None


class CrossFitPropensity:
    """Weighted logistic propensity, cross-fitted within the audit.

    Parameters
    ----------
    features:
        Observable features used as covariates.
    n_folds:
        Number of cross-fitting folds (2 in the original paper).
    c:
        Inverse L2 regularisation strength of the logistic regression.
    min_class:
        Feasibility rule: at least this many audited machine and human records
        (sums of shares for graded labels). The default 4 was chosen on development data
        in stage S4 (the first submission used 6).
    min_class_fold:
        Feasibility rule: at least this many of each class in every training fold.
    """

    def __init__(self, features: tuple[str, ...] = ("score", "loglen"), n_folds: int = 2,
                 c: float = 1.0, min_class: float = 4.0, min_class_fold: float = 2.0) -> None:
        self.features = tuple(features)
        self.n_folds = int(n_folds)
        self.c = float(c)
        self.min_class = float(min_class)
        self.min_class_fold = float(min_class_fold)

    @property
    def key(self) -> tuple:
        return ("propensity", self.features, self.n_folds, self.c, self.min_class,
                self.min_class_fold)

    def _features(self, obs: ObservedData) -> np.ndarray:
        names = tuple(f for f in self.features if f == "score" or f == "flag"
                      or f in obs.covariates)
        return obs.design_matrix(names, standardize=True)

    def _fold_ids(self, audit: AuditSample, rng: np.random.Generator) -> np.ndarray:
        """Balanced random folds within each (design stratum, label) cell."""
        folds = np.empty(audit.m, dtype=int)
        cell = audit.strata.astype(np.int64) * 3 + np.round(2 * audit.labels).astype(np.int64)
        start = int(rng.integers(self.n_folds))
        for c in np.unique(cell):
            members = np.flatnonzero(cell == c)
            members = rng.permutation(members)
            folds[members] = (start + np.arange(members.shape[0])) % self.n_folds
            start = (start + members.shape[0]) % self.n_folds
        return folds

    def _fit_predict(self, x_train, a_train, w_train, x_pred) -> np.ndarray:
        x2 = np.vstack([x_train, x_train])
        y2 = np.concatenate([np.ones(a_train.shape[0]), np.zeros(a_train.shape[0])])
        w2 = np.concatenate([w_train * a_train, w_train * (1.0 - a_train)])
        keep = w2 > 0
        model = LogisticRegression(C=self.c, max_iter=1000)
        model.fit(x2[keep], y2[keep], sample_weight=w2[keep] / w2[keep].mean())
        return model.predict_proba(x_pred)[:, 1]

    def fit(self, obs: ObservedData, audit: AuditSample,
            rng: np.random.Generator) -> PropensityFit:
        a = audit.labels
        if a.sum() < self.min_class or (1 - a).sum() < self.min_class:
            return PropensityFit(False, f"audit has {a.sum():.1f} machine and "
                                        f"{(1 - a).sum():.1f} human records "
                                        f"(< {self.min_class:g})")
        x = self._features(obs)
        xa = x[audit.idx]
        w = audit.w
        folds = self._fold_ids(audit, rng)
        eta_oof = np.empty(audit.m)
        eta_sum = np.zeros(obs.n)
        for k in range(self.n_folds):
            train = folds != k
            a_tr = a[train]
            if a_tr.sum() < self.min_class_fold or (1 - a_tr).sum() < self.min_class_fold:
                return PropensityFit(False, f"training fold {k} has too few records of a class")
            pred_all = self._fit_predict(xa[train], a_tr, w[train], x)
            eta_sum += pred_all
            eta_oof[folds == k] = pred_all[audit.idx[folds == k]]
        return PropensityFit(True, "", eta_sum / self.n_folds, eta_oof, folds)
