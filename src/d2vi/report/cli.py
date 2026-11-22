"""``d2vi-report``: the D2VI protocol on your own response set, in two commands.

Step 1 - choose which responses to verify (needs only the detector scores)::

    d2vi-report plan responses.csv --score score --budget 0.05 --out audit.csv

``audit.csv`` lists the records to verify, with their design stratum and inclusion
probability. Verify the origin of every listed record and fill the ``label`` column
(0 = human, 1 = machine).

Step 2 - estimate and report::

    d2vi-report estimate responses.csv audit.csv --outcome rating trust \
        --score score --flag flagged --length words --out report.md --json report.json

The report gives the estimated contamination rate, the detector's operating point, the
naive, deletion and audit-based estimates of the human mean with a 95% interval, whether
deletion is certified as safe, the estimate the selection policy reports, and a disclosure
paragraph. Data files are CSV or Parquet, one row per response.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .core import SINGLE_STAGE_DESIGNS, analyse, plan_audit, render_markdown


def _read(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith((".parquet", ".pq")) else pd.read_csv(path)


def _column(df: pd.DataFrame, name: str, what: str):
    if name not in df.columns:
        raise SystemExit(f"{what} column {name!r} is not in the data file "
                         f"(columns: {', '.join(map(str, df.columns))})")
    return df[name].to_numpy()


def _plan(args: argparse.Namespace) -> None:
    df = _read(args.data)
    score = _column(df, args.score, "score")
    if (args.budget is None) == (args.m is None):
        raise SystemExit("give exactly one of --budget (a share of the records) and --m")
    m = args.m if args.m is not None else max(int(round(args.budget * len(df))), 2)
    ids = _column(df, args.id, "id") if args.id else None
    try:
        plan = plan_audit(score, m, args.design, args.seed, ids)
    except ValueError as err:
        raise SystemExit(str(err)) from err
    plan.to_csv(args.out, index=False)
    print(f"{len(plan)} of {len(df)} records to verify, written to {args.out}; fill the "
          "'label' column (0 = human, 1 = machine) and run 'd2vi-report estimate'.")


def _estimate(args: argparse.Namespace) -> None:
    df = _read(args.data)
    score = _column(df, args.score, "score")
    if (args.flag is None) == (args.threshold is None):
        raise SystemExit("give exactly one of --flag (a 0/1 column) and --threshold (on the score)")
    flag = _column(df, args.flag, "flag") if args.flag else (score > args.threshold).astype(int)
    length = _column(df, args.length, "length") if args.length else None
    y = {o: _column(df, o, "outcome") for o in args.outcome}
    try:
        res = analyse(y, score, flag, pd.read_csv(args.audit), length, args.tau_sd, args.alpha,
                      args.seed)
    except ValueError as err:
        raise SystemExit(str(err)) from err
    text = render_markdown(res)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, default=float))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="d2vi-report", description=__doc__.split("\n\n")[0],
                                 epilog="Worked example: python -m pydoc d2vi.report.cli")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="draw the verification audit from the detector scores")
    p.add_argument("data")
    p.add_argument("--score", required=True, help="column with the detector score")
    p.add_argument("--budget", type=float, help="audit size as a share of the records, e.g. 0.05")
    p.add_argument("--m", type=int, help="audit size as a number of records")
    p.add_argument("--design", default="proportional", choices=SINGLE_STAGE_DESIGNS)
    p.add_argument("--id", help="optional identifier column to copy into the plan")
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--out", required=True)
    p.set_defaults(run=_plan)
    e = sub.add_parser("estimate", help="estimate the human mean and write the report")
    e.add_argument("data")
    e.add_argument("audit", help="the plan written by 'plan', with the label column filled in")
    e.add_argument("--outcome", nargs="+", required=True, help="outcome column(s)")
    e.add_argument("--score", required=True)
    e.add_argument("--flag", help="column with the detector flag (0/1)")
    e.add_argument("--threshold", type=float, help="flag = score above this value")
    e.add_argument("--length", help="optional column with the response length (words)")
    e.add_argument("--tau-sd", type=float, default=0.2,
                   help="certification tolerance in SD of the outcome (declare it in advance)")
    e.add_argument("--alpha", type=float, default=0.05)
    e.add_argument("--seed", type=int, default=20261109, help="seed of the cross-fitting folds")
    e.add_argument("--out", help="Markdown report (default: print)")
    e.add_argument("--json", help="also write the numbers as JSON")
    e.set_defaults(run=_estimate)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
