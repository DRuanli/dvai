"""Estimators of the human-population mean."""

from .base import EstimateResult, EstimationContext, Estimator
from .design_based import RCPPI, AuditHajek, DifferenceRatioEstimator, FlagPPI, GREGRatio
from .propensity import CrossFitPropensity, PropensityFit
from .simple import Deletion, MomentInversion, Naive

__all__ = [
    "EstimateResult",
    "EstimationContext",
    "Estimator",
    "Naive",
    "Deletion",
    "MomentInversion",
    "AuditHajek",
    "FlagPPI",
    "RCPPI",
    "GREGRatio",
    "DifferenceRatioEstimator",
    "CrossFitPropensity",
    "PropensityFit",
    "make_estimator",
]


def make_estimator(spec: str | dict) -> Estimator:
    """Build an estimator from a name or a ``{"name": ..., **params}`` mapping."""
    if isinstance(spec, str):
        spec = {"name": spec}
    params = dict(spec)
    name = params.pop("name")
    if name == "naive":
        return Naive()
    if name == "deletion":
        return Deletion()
    if name == "hajek":
        return AuditHajek()
    if name == "flag_ppi":
        return FlagPPI(tune=False)
    if name == "flag_ppi_pp":
        return FlagPPI(tune=True)
    if name in ("rc_ppi", "rc_ppi_pp"):
        prop = CrossFitPropensity(**params) if params else None
        return RCPPI(tune=name == "rc_ppi_pp", propensity=prop)
    if name == "greg":
        return GREGRatio(**params)
    if name == "moment":
        return MomentInversion(**params)
    raise KeyError(f"unknown estimator {name!r}")
