"""Stage S2: build the human pool of the survey study from the ANES 2024 files.

Usage (from the repository root)::

    python scripts/build_anes_human_pool.py <main CSV or zip> <redacted open-ends xlsx>

Writes ``data/private/anes2024_human.parquet`` (not committed: ANES data are not
redistributed) and ``results/S2_anes_pool_summary.json`` (counts only, committed).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from d2vi.data.anes import ANES2024


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(csv: str, xlsx: str) -> None:
    data = ANES2024(csv, xlsx)
    r = data.respondents()
    pool = data.human_pool(r)
    out = Path("data/private")
    out.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(out / "anes2024_human.parquet", index=False)
    ys = [c for c in pool.columns if c.startswith("y_")]
    summary = {
        "files": {Path(csv).name: sha(csv), Path(xlsx).name: sha(xlsx)},
        "respondents": int(len(r)),
        "exclusions": r["exclusion"].replace("", "eligible").value_counts().to_dict(),
        "eligible_by_mode": pool["mode"].value_counts().to_dict(),
        "words_median_by_mode": pool.groupby("mode")["length"].median().to_dict(),
        "outcome_means": pool[ys].mean().round(4).to_dict(),
        "outcome_sds": pool[ys].std().round(4).to_dict(),
        "pool_sha256": sha(out / "anes2024_human.parquet"),
    }
    summary["chatbot_hallmark_respondents"] = {str(k): int(v) for k, v in
                                               r.loc[r["hallmark"], "mode"].value_counts().items()}
    Path("results/S2_anes_pool_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
