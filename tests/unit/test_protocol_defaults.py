"""The code defaults must equal the protocol configuration (G4 review, finding M1).

Reads ``configs/protocol/frozen.yaml`` when it exists, otherwise ``draft.yaml``.
"""

import inspect
from pathlib import Path

import pytest
import yaml

from d2vi.decision.certify import TOSTCertifier
from d2vi.decision.policy import MSEPolicy
from d2vi.estimators.design_based import DifferenceRatioEstimator
from d2vi.estimators.propensity import CrossFitPropensity
from d2vi.inference.variance import DEFAULT_TARGETS, variance_report
from d2vi.sampling.audit import make_design

ROOT = Path(__file__).resolve().parents[2] / "configs" / "protocol"
CFG = yaml.safe_load(((ROOT / "frozen.yaml") if (ROOT / "frozen.yaml").exists()
                      else (ROOT / "draft.yaml")).read_text())


def test_default_design_parameters_are_the_class_defaults():
    spec = dict(CFG["audit"]["default_design"])
    design = make_design(spec["name"])
    for key, value in spec.items():
        if key == "name":
            continue
        got = getattr(design, key)
        assert (tuple(got) if isinstance(got, tuple) else got) == \
            (tuple(value) if isinstance(value, list) else value), key


def test_propensity_defaults():
    p, cfg = CrossFitPropensity(), CFG["propensity"]
    assert tuple(p.features) == tuple(cfg["features"])
    for key in ("n_folds", "c", "min_class", "min_class_fold"):
        assert getattr(p, key) == pytest.approx(cfg[key]), key


def test_interval_rule_defaults():
    cfg = CFG["variance"]
    assert DifferenceRatioEstimator.variance_rule == cfg["rule"]
    assert DEFAULT_TARGETS[cfg["rule"]]["prior"] == cfg["target_prior"]
    kappa = inspect.signature(variance_report).parameters["kappa"].default
    assert kappa == pytest.approx(cfg["kappa"])


def test_policy_fallback_and_certifier_defaults():
    pol, cert = MSEPolicy(), TOSTCertifier()
    assert pol.bias_sq == CFG["policy"]["bias_sq"]
    assert pol.require_certification == CFG["policy"]["require_certification"]
    assert pol.fallback.name == CFG["fallback_when_infeasible"]
    c = CFG["certification"]
    assert cert.tau_sd == pytest.approx(c["tau_sd"]) and cert.alpha == pytest.approx(c["alpha"])
    assert cert.quantile == c["quantile"]
    assert cert.fallback == (c["fallback"] == CFG["fallback_when_infeasible"])
