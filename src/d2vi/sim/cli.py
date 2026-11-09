"""``d2vi-run CONFIG.yaml``: run one simulation experiment from a YAML file.

Example configuration::

    name: E2_dev_ghostbuster_essay
    features: data/features/ghostbuster_tfidf.parquet
    filter: {domain: essay, generator: [human, gpt]}
    corpus: ghostbuster-essay
    detector: tfidf_lr
    outcomes: [sent, read]
    n: 900
    eps: [0.05, 0.10, 0.20, 0.30]
    budgets: [0.05]
    designs: [srs, top_decile]
    mixture: iid
    estimators: [naive, deletion, hajek, flag_ppi, rc_ppi, rc_ppi_pp]
    policies: [{bias_sq: plugin}]
    certifiers: [{tau_sd: 0.2}]
    n_rep: 1000
    n_jobs: 4
    seed: 20261109
    out: results/raw/E2_dev_ghostbuster_essay.parquet

A manifest with the configuration hash, package version and timing is written next to
the output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import pandas as pd
import yaml

from .. import __version__
from ..decision.certify import TOSTCertifier
from ..decision.policy import MSEPolicy
from ..estimators import make_estimator
from ..sampling.mixture import FeaturePool
from .runner import CellSpec, Experiment


def _filter(df: pd.DataFrame, spec: dict | None) -> pd.DataFrame:
    for col, val in (spec or {}).items():
        df = df[df[col].isin(val)] if isinstance(val, list) else df[df[col] == val]
    return df


def run_config(path: str | Path) -> Path:
    path = Path(path)
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw)
    df = _filter(pd.read_parquet(cfg["features"]), cfg.get("filter"))
    pool = FeaturePool(df, threshold=cfg.get("threshold", 0.5),
                       extra_covariates=tuple(cfg.get("extra_covariates", ())))
    exp = Experiment(
        pool,
        [make_estimator(e) for e in cfg["estimators"]],
        policies=[MSEPolicy(**p) for p in cfg.get("policies", [])],
        certifiers=[TOSTCertifier(**c) for c in cfg.get("certifiers", [])],
        outcomes=cfg.get("outcomes"),
        root_seed=cfg.get("seed", 20261109),
    )
    specs = [CellSpec(cfg["corpus"], cfg["detector"], float(e), int(cfg["n"]), float(b), d,
                      cfg.get("mixture", "iid"))
             for e in cfg["eps"] for b in cfg["budgets"] for d in cfg["designs"]]
    t0 = time.time()
    out = exp.run(specs, int(cfg["n_rep"]), int(cfg.get("n_jobs", 1)))
    out_path = Path(cfg["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    manifest = {
        "config": str(path),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "features_sha256": hashlib.sha256(Path(cfg["features"]).read_bytes()).hexdigest(),
        "d2vi_version": __version__,
        "python": platform.python_version(),
        "machine": platform.platform(),
        "seconds": round(time.time() - t0, 1),
        "rows": int(len(out)),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    args = parser.parse_args()
    print(run_config(args.config))


if __name__ == "__main__":  # pragma: no cover
    main()
