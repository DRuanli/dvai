# Result files

Small summaries are committed; raw replication files (`results/raw/`) are rebuilt by the
scripts and archived on Zenodo with the submission.

| File | Script | Status |
|---|---|---|
| `E0_*` | `run_e0_reproduction.py`, `run_e0_hc3.py` | Current: exact reproduction of the first submission (stage S3) |
| `S2_anes_pool_summary.json` | `build_anes_human_pool.py` | Current |
| `S4_variance_dev_summary.csv` | `run_s4_variance_dev.py` | Interval-rule comparison, round 1. Rows for SRS, proportional and top-decile designs are valid; rows for `adaptive` and `neyman_n_strata5` used the faulty two-stage weights (F15) and are superseded |
| `S4_design_dev_summary.csv`, `S4_tuning_dev_summary.csv` | `run_s4_design_dev.py`, `run_s4_tuning_dev.py` | Superseded (round 1; two-stage weights of F15, Hájek fallback) |
| `S4_policy_dev_*` | `run_s4_policy_dev.py` | Superseded (round 1 design) |
| `S4_explore_propensity_variance.csv` | `s4_explore_propensity_variance.py` | Exploratory (top-decile, one detector); superseded for the fold choice by S4b |
| `S4b_design_dev_summary.csv`, `S4b_design_dev_cells.csv` | `run_s4b_design_dev.py` | Current: audit design and number of folds (round 2) |
| `S4c_*` | `run_s4c_final_dev.py` | Current: policy, certifier, feasibility threshold and interval rule on the chosen design |
