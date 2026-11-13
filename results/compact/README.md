# Compact S5 results

Per-pool reductions of the held-out replication files (`scripts/compact_s5.py`), enough to
reproduce every held-out number in the paper without the raw replication files. The raw
files can be regenerated with `scripts/run_s5.py` (seeds depend on the cell only, so a rerun
gives the same numbers):

    python scripts/report_endpoints.py results/compact/S5/primary --out results/S5_endpoints.json
    python scripts/summarise_secondary.py results/compact/S5 --tag final

`<grid>/<corpus>__<pool>.cells.parquet`, `.certcells.parquet`, `.clusters.parquet`: endpoint
tables (grids with one design per cell); `.sec.parquet`: secondary summary table;
`<corpus>.manifest.json`: run manifest (code commit, data and protocol hashes, pool sizes,
skipped cells, machine).

Where the pools ran: ANES, Beemo, M4, HC3-Chinese, DAIGT in the sandbox (commit of each
pool in its manifest); RAID on Kaggle CPU (primary, small-budget, large-N: K07 at 785fb1d;
verification error: K08-0..9 at bcacbd7; mixtures and designs: K09-00..19 at bcacbd7).
Simulation code is identical in these commits. Six RAID mixtures pools run in both places
agree to 1e-12. The ANES tables contain cell-level error summaries only, no respondent data.
