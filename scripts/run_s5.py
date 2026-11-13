"""Stage S5: run a held-out corpus on a grid of the frozen protocol.

    python scripts/run_s5.py configs/s5/anes.yaml --grid primary

Everything that could be tuned comes from ``configs/protocol/frozen.yaml`` and the package
defaults (which a test keeps equal to it): design, estimators, interval rule, policy,
certifier, grid, replications, seed. The corpus configuration only says where the feature
table is and how pools are formed::

    corpus: anes2024
    features: data/private/anes_features.parquet
    outcomes: [therm_harris, therm_trump]
    human: {source: human}                       # rows of the human class
    machine_groups:                              # one pool per group x detector
      qwen: {source: Qwen/Qwen2.5-7B-Instruct}
    detectors:
      tfidf: {score: score_tfidf, flag: flag_tfidf}
    domains: [news, reddit]                      # optional: pools are formed within a domain
                                                 # ("all": every domain of the feature table)

``machine_groups: {by: source}`` makes one group per value of that column among the machine
records (for corpora whose generators are chosen by a size rule when the subset is built).


Grids: ``primary`` (all primary endpoints) and the secondary grids ``small_budget``,
``large_n``, ``designs``, ``mixtures``, ``verification_error``.

Held-out data are run once: a pool whose output file exists is skipped and there is no
overwrite option. A pool without records of one class is recorded as empty.
``--done-elsewhere`` skips the pools recorded in the manifests of runs on other machines (the
same seeds give the same results, so a pool is never run twice to choose between draws).
``--shard I/K`` runs every K-th pool of the plan, starting at I, so that K machines can share a
grid. ``--max-minutes`` stops before starting a new pool, so that a run can be split over sessions
(seeds depend on the cell, not on the order of the pools, so the split does not change any
result). ``--reps`` is for rehearsals only and requires ``--out-root`` outside
``results/raw/S5``. Output: ``<out-root>/<grid>/<corpus>__<group>__<detector>.parquet`` and
``<out-root>/<grid>/<corpus>.manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

import d2vi
from d2vi.decision.certify import LegacyCertifier, TOSTCertifier
from d2vi.decision.policy import MSEPolicy
from d2vi.estimators import make_estimator
from d2vi.sampling.mixture import FeaturePool
from d2vi.sim.runner import CellSpec, Experiment

FROZEN = Path("configs/protocol/frozen.yaml")
DEFAULT_ROOT = Path("results/raw/S5")
ESTIMATORS = ["naive", "deletion", "hajek", "flag_ppi", "rc_ppi_pp",      # endpoints
              "moment", "flag_ppi_pp", "rc_ppi", "greg"]                  # secondary
KEEP = ["corpus", "pool", "detector", "budget", "eps", "n", "design", "mixture", "phi0", "phi1",
        "outcome", "rep", "estimator", "point", "se", "ci_low", "ci_high", "feasible", "truth",
        "sd_h", "deletion_bias", "certified", "tau", "choice", "m_machine", "s", "f"]


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _match(df: pd.DataFrame, spec: dict) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in spec.items():
        mask &= df[col].isin(val) if isinstance(val, list) else df[col].eq(val)
    return mask


def build_pool(feat: pd.DataFrame, cfg: dict, group: str, det: str) -> tuple[pd.DataFrame, dict]:
    """Rows of one pool with the detector's columns renamed to ``score`` and ``flag``."""
    d = cfg["detectors"][det]
    rows = feat[_match(feat, cfg["human"]) | _match(feat, cfg["machine_groups"][group])]
    scored = rows[rows[d["score"]].notna() & rows[d["flag"]].notna()]
    ys = [f"y_{o}" for o in cfg["outcomes"]]
    keep = ["doc_id", "a", "length", *ys, *[c for c in ("source", "condition") if c in scored]]
    pool = scored[keep].copy()
    pool["score"] = scored[d["score"]].astype(float)
    pool["flag"] = scored[d["flag"]].astype(int)
    if "source" in pool:
        pool = pool.rename(columns={"source": "generator"})
    info = {"human": int((pool.a == 0).sum()), "machine": int((pool.a > 0).sum()),
            "unscorable_dropped": int(len(rows) - len(scored)),
            "flag_rate_human": float(pool.loc[pool.a == 0, "flag"].mean()),
            "flag_rate_machine": float(pool.loc[pool.a > 0, "flag"].mean())}
    return pool, info


def pool_plan(cfg: dict) -> list[tuple[str, str | None, str, str]]:
    """``(pool name, domain or None, machine group, detector)`` for every pool of a corpus:
    one pool per (domain x) group x detector (frozen protocol, section 2)."""
    plan = []
    for dom in cfg.get("domains") or [None]:
        for group, det in itertools.product(cfg["machine_groups"], cfg["detectors"]):
            label = group if dom is None else f"{dom}-{group}"
            plan.append((f"{cfg['corpus']}__{label}__{det}", dom, group, det))
    return plan


def resolve(cfg: dict, feat: pd.DataFrame) -> dict:
    """Expand the two data-dependent shortcuts of a corpus configuration."""
    cfg = dict(cfg)
    if cfg.get("domains") == "all":
        cfg["domains"] = sorted(map(str, feat["domain"].unique()))
    groups = cfg["machine_groups"]
    if set(groups) == {"by"}:
        col = groups["by"]
        values = sorted(map(str, feat.loc[feat["a"] > 0, col].unique()))
        cfg["machine_groups"] = {v: {col: v} for v in values}
    return cfg


def cell_specs(grid: str, frozen: dict, corpus: str, det: str, pool_name: str) -> tuple[list, int]:
    g = frozen["grid"]
    prim = g["primary"]
    default = frozen["audit"]["default_design"]
    tags = {"pool": pool_name}

    def spec(eps, budget, n=prim["N"], design=default, mixture="iid", phi0=0.0, phi1=0.0):
        return CellSpec(corpus, det, eps, n, budget, design, mixture, tags=tags, phi0=phi0,
                        phi1=phi1)

    if grid == "primary":
        return [spec(e, b) for b in prim["budgets"] for e in prim["eps"]], prim["replications"]
    sec = g["secondary"][grid]
    reps = sec["replications"]
    if grid == "small_budget":
        return [spec(e, b) for b in sec["budgets"] for e in sec["eps"]], reps
    if grid == "large_n":
        return [spec(e, b, n=sec["N"]) for b in sec["budgets"] for e in sec["eps"]], reps
    if grid == "designs":
        return [spec(e, b, design=d) for d in frozen["audit"]["compared"]
                for b in sec["budgets"] for e in sec["eps"]], reps
    if grid == "mixtures":
        return [spec(e, b, mixture=m) for m in sec["mixture"]
                for b in sec["budgets"] for e in sec["eps"]], reps
    if grid == "verification_error":
        return [spec(e, b, phi0=p0, phi1=p1) for p0, p1 in itertools.product(sec["phi"], repeat=2)
                for b in sec["budgets"] for e in sec["eps"]], reps
    raise SystemExit(f"unknown grid {grid!r}")


def experiment(pool: pd.DataFrame, outcomes: list[str], seed: int) -> Experiment:
    certifiers = [TOSTCertifier(tau_sd=t) for t in (0.1, 0.2, 0.3)] + [LegacyCertifier()]
    return Experiment(FeaturePool(pool), [make_estimator(e) for e in ESTIMATORS],
                      policies=[MSEPolicy()], certifiers=certifiers, outcomes=outcomes,
                      root_seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--grid", default="primary")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--reps", type=int, help="rehearsal only; needs --out-root")
    ap.add_argument("--out-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--max-minutes", type=float,
                    help="start no new pool after this many minutes (the run can be resumed)")
    ap.add_argument("--shard", default="0/1",
                    help="I/K: run only the pools whose position in the plan is I modulo K")
    ap.add_argument("--done-elsewhere", nargs="*", default=[],
                    help="manifests of runs on other machines: pools listed there are skipped")
    args = ap.parse_args()
    started = time.time()
    root = Path(args.out_root)
    if args.reps is not None and root.resolve() == DEFAULT_ROOT.resolve():
        raise SystemExit("--reps is for rehearsals: give --out-root outside results/raw/S5")
    frozen = yaml.safe_load(FROZEN.read_text())
    cfg = yaml.safe_load(Path(args.config).read_text())
    feat = pd.read_parquet(cfg["features"])
    cfg = resolve(cfg, feat)
    out_dir = root / args.grid
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / f"{cfg['corpus']}.manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"pools": {}}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    manifest |= {"corpus": cfg["corpus"], "grid": args.grid, "config_sha256": sha(args.config),
                 "features_sha256": sha(cfg["features"]), "frozen_sha256": sha(FROZEN),
                 "d2vi": d2vi.__version__, "git_commit": commit.strip(),
                 "rehearsal": args.reps is not None, "estimators": ESTIMATORS}
    man_path.write_text(json.dumps(manifest, indent=2))
    elsewhere = set()
    for m in args.done_elsewhere:
        elsewhere |= set(json.loads(Path(m).read_text())["pools"])
    shard, n_shards = map(int, args.shard.split("/"))
    for pos, (name, dom, group, det) in enumerate(pool_plan(cfg)):
        if name in elsewhere or pos % n_shards != shard:
            continue
        out = out_dir / f"{name}.parquet"
        if out.exists():
            print("exists, skipped:", out.name, flush=True)
            continue
        if name in manifest["pools"] and manifest["pools"][name].get("empty"):
            continue
        if args.max_minutes is not None and (time.time() - started) / 60 > args.max_minutes:
            print(f"time budget of {args.max_minutes} minutes reached; resume later", flush=True)
            break
        rows = feat if dom is None else feat[feat["domain"].eq(dom)]
        pool, info = build_pool(rows, cfg, group, det)
        if info["human"] == 0 or info["machine"] == 0:     # e.g. a generator absent in a language
            manifest["pools"][name] = {**info, "empty": True, "cells_run": 0}
            man_path.write_text(json.dumps(manifest, indent=2))
            print(f"empty pool, recorded: {name}", flush=True)
            continue
        specs, reps = cell_specs(args.grid, frozen, cfg["corpus"], det, name)
        reps = args.reps or reps
        exp = experiment(pool, cfg["outcomes"], frozen["seed"])
        t0, frames, skipped = time.time(), [], []
        for sp in specs:
            try:
                df = exp.run_cell(sp, reps, args.jobs)
            except ValueError as err:        # pool too small for this cell, or mixture n/a
                skipped.append({"eps": sp.eps, "budget": sp.budget, "n": sp.n,
                                "design": str(sp.design), "mixture": str(sp.mixture),
                                "reason": str(err)})
                continue
            df["phi0"], df["phi1"] = sp.phi0, sp.phi1
            frames.append(df.reindex(columns=KEEP))
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        manifest["pools"][name] = {**info, "cells_run": len(frames), "cells_skipped": skipped,
                                   "replications": reps, "seconds": round(time.time() - t0),
                                   "git_commit": commit.strip(), "platform": platform.node()}
        man_path.write_text(json.dumps(manifest, indent=2))
        print(f"done {name}: {len(frames)} cells, {len(skipped)} skipped, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
