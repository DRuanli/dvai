"""Reduce raw S5 replication files to the per-pool tables the reports need.

    python scripts/compact_s5.py RAW_GRID_DIR OUT_DIR

For every ``<pool>.parquet`` in ``RAW_GRID_DIR`` writes, in ``OUT_DIR``:
``<pool>.cells.parquet`` and ``<pool>.certcells.parquet`` (endpoint cells and certification
counts per cell, ``d2vi.analysis.endpoints``), ``<pool>.clusters.parquet`` (certification
counts per replication cluster) and ``<pool>.sec.parquet`` (the cell table of
``summarise_secondary.py``). The reports computed from these files equal those computed
from the raw files (``tests/unit/test_s5_driver.py``); the raw files remain the archive.
Used on Kaggle, where raw files are too large to send back. Manifests are copied as they are.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from d2vi.analysis.endpoints import DEFAULT, cell_stats, cert_tables

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarise_secondary import COLS, cell_table  # noqa: E402

SUFFIXES = (".cells.parquet", ".certcells.parquet", ".clusters.parquet", ".sec.parquet")


def is_single_design(raw: pd.DataFrame) -> bool:
    """True when every (budget, eps) cell has one design, mixture, N and verification error,
    as in the primary, small-budget and large-N grids. The endpoint tables key cells by
    (pool, budget, eps, outcome) and are not defined for the other secondary grids, which
    only need the ``.sec`` table."""
    keys = [c for c in ("design", "mixture", "n", "phi0", "phi1") if c in raw]
    return bool((raw.groupby(["budget", "eps"])[keys].nunique() <= 1).all().all())


def compact_file(path: Path, out_dir: Path) -> None:
    raw = pd.read_parquet(path)
    stem = path.name[: -len(".parquet")]
    if is_single_design(raw):          # endpoint tables are defined for one design per cell
        cell_stats(raw, DEFAULT).to_parquet(out_dir / f"{stem}.cells.parquet", index=False)
        clusters, cert_cells = cert_tables(raw, DEFAULT)
        clusters.to_parquet(out_dir / f"{stem}.clusters.parquet", index=False)
        cert_cells.to_parquet(out_dir / f"{stem}.certcells.parquet", index=False)
    cell_table(raw[[c for c in COLS if c in raw]]).to_parquet(out_dir / f"{stem}.sec.parquet",
                                                             index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(raw_dir.glob("*.parquet")):
        if f.name.endswith(SUFFIXES):
            continue
        if (out_dir / f"{f.name[:-len('.parquet')]}.sec.parquet").exists():  # done earlier
            continue
        compact_file(f, out_dir)
        print("compacted", f.name, flush=True)
    for m in raw_dir.glob("*.manifest.json"):
        shutil.copy(m, out_dir / m.name)


if __name__ == "__main__":
    main()
