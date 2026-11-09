"""Shared fixtures. All data here are SYNTHETIC and used only to test code and theory.

Paper results come exclusively from real corpora (see docs/EXPERIMENT_PROTOCOL.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from d2vi.sampling.mixture import FeaturePool


def synthetic_pool(n_h: int = 3000, n_a: int = 1500, seed: int = 0,
                   differential: bool = False) -> FeaturePool:
    """Human and machine documents with a score, a flag, a length and two outcomes.

    With ``differential=True`` the human false-positive rate rises with the outcome and the
    machine sensitivity falls with it, violating non-differentiality (M1).
    """
    rng = np.random.default_rng(seed)
    y_h = rng.normal(0.1, 0.5, n_h)
    y_a = rng.normal(0.6, 0.3, n_a)
    z_h = rng.normal(-1.0, 1.0, n_h) + (1.5 * y_h if differential else 0.0)
    z_a = rng.normal(1.2, 1.0, n_a) - (1.5 * y_a if differential else 0.0)
    score = 1 / (1 + np.exp(-np.concatenate([z_h, z_a])))
    df = pd.DataFrame({
        "doc_id": np.arange(n_h + n_a),
        "a": np.concatenate([np.zeros(n_h), np.ones(n_a)]),
        "score": score,
        "length": rng.integers(80, 600, n_h + n_a),
        "y_sent": np.concatenate([y_h, y_a]),
        "y_read": np.concatenate([rng.normal(0.5, 0.2, n_h), rng.normal(0.4, 0.1, n_a)]),
        "generator": np.concatenate([np.full(n_h, "human"),
                                     rng.choice(["g1", "g2", "g3"], n_a)]),
    })
    return FeaturePool(df)


@pytest.fixture(scope="session")
def pool() -> FeaturePool:
    return synthetic_pool()


@pytest.fixture(scope="session")
def diff_pool() -> FeaturePool:
    return synthetic_pool(differential=True, seed=1)
