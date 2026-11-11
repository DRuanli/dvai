"""Primary endpoints PE1-PE4 from a directory of raw replication files.

    python scripts/report_endpoints.py results/raw/S5/primary --out results/S5_endpoints.json

Each parquet file (one pool) is reduced on its own (`d2vi.analysis.endpoints`), so memory
stays bounded. Defaults follow the frozen protocol: cells are corpus x pool x budget x
contamination x outcome and the unit of inference is the corpus.

Dress rehearsal on development data (not an endpoint result; the units are pools because
there are only two development corpora)::

    python scripts/report_endpoints.py results/raw/S4c \\
        --cell det,domain,generator,budget,eps,outcome --unit det,domain,generator \\
        --out results/S5_rehearsal_dev_endpoints.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from d2vi.analysis.endpoints import EndpointSpec, cell_stats, cert_tables, evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", nargs="+",
                    help="directories of raw files or of compact files (compact_s5.py)")
    ap.add_argument("--cell", default="corpus,pool,budget,eps,outcome")
    ap.add_argument("--unit", default="corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()
    spec = EndpointSpec(cell=tuple(args.cell.split(",")), unit=tuple(args.unit.split(",")))
    cells, clusters, cert_cells, files = [], [], [], []
    for d in map(Path, args.raw_dir):
        compact = sorted(d.glob("*.cells.parquet"))
        raw_files = [f for f in sorted(d.glob("*.parquet")) if f.name.count(".") == 1]
        for f in compact:
            stem = f.name[: -len(".cells.parquet")]
            cells.append(pd.read_parquet(f))
            clusters.append(pd.read_parquet(d / f"{stem}.clusters.parquet"))
            cert_cells.append(pd.read_parquet(d / f"{stem}.certcells.parquet"))
            files.append(f"{d.name}/{stem} (compact)")
        done = {f.name[: -len(".cells.parquet")] for f in compact}
        for f in raw_files:
            if f.stem in done:
                continue
            raw = pd.read_parquet(f)
            cells.append(cell_stats(raw, spec))
            cl, cc = cert_tables(raw, spec)
            clusters.append(cl)
            cert_cells.append(cc)
            files.append(f"{d.name}/{f.stem}")
    if not files:
        raise SystemExit(f"no parquet files in {args.raw_dir}")
    if spec != EndpointSpec() and any("(compact)" in f for f in files):
        raise SystemExit("compact files were reduced with the default cells and units")
    cells = pd.concat(cells, ignore_index=True)
    res = evaluate(cells, pd.concat(clusters, ignore_index=True),
                   pd.concat(cert_cells, ignore_index=True), spec, n_boot=args.boot)
    res["source"] = {"raw_dir": args.raw_dir, "files": files,
                     "cell": list(spec.cell), "unit": list(spec.unit)}
    out = Path(args.out)
    out.write_text(json.dumps(res, indent=2))
    cells.to_csv(out.with_name(out.stem + "_cells.csv"), index=False)
    for pe in ("PE1", "PE2", "PE3"):
        print(pe, "passes:", res[pe]["passes"])
    print("PE1 ITT ratio", round(res["PE1"]["itt"]["pooled_ratio"], 3),
          "| per protocol", round(res["PE1"]["per_protocol"].get("pooled_ratio", float("nan")), 3))
    print("PE2 mean coverage", round(res["PE2"]["mean_coverage"], 3),
          "| cells >= 0.90:", round(res["PE2"]["share_cells_ge_090"], 3))
    print("PE3 false certification", round(res["PE3"]["bad"]["rate"], 4), "upper",
          round(res["PE3"]["bad"]["upper95_cluster_bootstrap"], 4))
    print("PE4 lowest worst-case regret:", res["PE4"]["pooled"]["lowest_worst_case"])


if __name__ == "__main__":
    main()
