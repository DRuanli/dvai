# D2VI — Detect, then Validly Infer

Replication package and toolkit for *Detect, Then What? Valid Social Measurement from
AI-Contaminated Response Data* (Neurocomputing, under revision, NEUCOM-D-26-20926).

When machine-written responses contaminate survey or text data, the usual remedy —
run an AI-text detector and delete what it flags — is itself an estimator whose error
depends on detector operating points that the analyst cannot observe. This package
estimates the human-population mean from a small **verification audit** instead:

- **RC-PPI / RC-PPI++** — a ratio-form, cross-fitted prediction-powered estimator whose
  consistency follows from the audit design alone (known inclusion probabilities), not
  from assumptions about the detector;
- **audit designs** with exact budgets — simple random, proportional score-stratified,
  top-decile split, two-stage Neyman, label-adaptive two-stage, score-rank PPS;
- **design-based standard errors** (linearisation with finite-population correction and a
  small-sample interval rule) and a Rao–Wu rescaling bootstrap;
- **certification of deletion** by an equivalence test (TOST), whose false-certification
  rate is measured on held-out data;
- an **audit-anchored selection policy** among naive, deletion and RC-PPI++;
- **`d2vi-report`**, a command that applies the protocol to your own response set.

## Status (23 November 2026)

The held-out evaluation is complete (13 November). Every tunable choice was made on two
development corpora and frozen at commit `d091b5b` (9 November) before six held-out
corpora were run once: RAID, DAIGT, M4, ANES 2024, Beemo and
HC3-Chinese; 153 pools, 12,012 cells, 1,000 replications per cell
(`docs/EXPERIMENT_PROTOCOL.md`). The four pre-specified endpoints
(`results/S5_endpoints.json`):

| Endpoint | Held-out result |
|---|---|
| PE1: RC-PPI++ against flag-PPI on the same audit | MSE ratio 0.61 intention-to-treat, 0.51 per protocol; below 1 in all six corpora; above 1 in 26 of 153 pools |
| PE2: 95% intervals | mean coverage 0.977; 97.5% of cells at 0.90 or more; lowest cell 0.614; median width 1.36 times the oracle width |
| PE3: certification of deletion (TOST, 0.2 SD) | certified wrongly in 1.05% of replications with a bias beyond tolerance (3.69% near the boundary); 60 of 1,188 cells exceed 7.5% |
| PE4: worst-case excess standardised RMSE | policy 0.197, RC-PPI++ 0.223, flag-PPI 0.681, deletion 1.520, naive 1.569 |

Deletion is the most accurate single estimator in half of the cells and fails badly in
others; `results/S5_posthoc_descriptives.json` has that description, which was defined
after the results were seen and is labelled post hoc. The manuscript is being revised;
stage status is in `docs/STAGES.md`. The first submission's Tables 2, 3 and D.2 are
reproduced exactly by the legacy code path (`docs/reproduction_report.md`), and the
problems found in that code are listed in `docs/FINDINGS_V1.md`.

## Install

```bash
python -m pip install -e ".[dev]"        # core + tests
python -m pip install -e ".[text]"       # VADER / textstat outcome scorers
python -m pip install -e ".[gpu]"        # neural detectors (Colab, Kaggle)
```

## Test

```bash
pytest -q          # unit, theory and legacy tests (under a minute)
ruff check src tests scripts
```

`tests/theory` checks the paper's results numerically on synthetic populations: the
identities exactly and the limit statements by simulation (table at the end of
`paper/theory.tex`). Synthetic data are used only in tests; every result in the paper
comes from real corpora.

## Use it on your own data

Verification happens outside the software, so the protocol takes two commands.

```bash
# 1. choose which responses to verify (needs only the detector scores)
d2vi-report plan responses.csv --score score --budget 0.05 --out audit.csv
#    verify every listed record and fill the `label` column (0 = human, 1 = machine)
# 2. estimate and report
d2vi-report estimate responses.csv audit.csv --outcome rating --score score \
    --flag flagged --length words --out report.md --json report.json
```

The report gives the estimated contamination rate, the detector's operating point, the
naive, deletion and audit-based estimates of the human mean with a 95% interval, whether
deletion is certified as safe, the estimate the selection policy reports, and a
disclosure paragraph. Defaults are those of the frozen protocol. Each command takes about
one second for 900 and for 100,000 responses (2-vCPU x86-64 container). The command
wraps the classes the simulations call; it was not used to produce the paper's results.

## Reproduce the held-out results

Per-pool reductions of the replication files are in `results/compact/S5`.

```bash
python scripts/report_endpoints.py results/compact/S5/primary --out results/S5_endpoints.json
python scripts/summarise_secondary.py results/compact/S5 --tag final
python scripts/secondary_tests.py results/compact/S5
python scripts/heldout_breakdowns.py results/compact/S5
python scripts/posthoc_descriptives.py results/compact/S5        # post hoc, descriptive
python scripts/make_tables.py                                    # paper/generated/*.tex
python scripts/make_figures.py paper/generated results/S5_final_secondary.csv \
    results/compact/S5/primary
```

Each command reproduces the committed file exactly (`results/README.md` lists them).
The simulations themselves are rerun with `scripts/run_s5.py configs/s5/<corpus>.yaml
--grid <grid>`; seeds depend on the cell only, so a rerun gives the same numbers. Two
things cannot be rebuilt from this repository alone: the ANES 2024 feature table, because
ANES records may not be redistributed (`docs/dataset_cards/anes2024.md` says how to build
it), and `results/S5_all_estimators.csv`, which needs the raw replication files.

Development corpora and the first submission:

```bash
# Study 1 features on CPU (about 2 minutes); see the script header for downloads
python scripts/compute_features_ghostbuster.py --root data/raw/ghostbuster-data \
    --subj-dir data/raw/subjectivity --out data/features/ghostbuster_tfidf.parquet
# E0: v1 code path and corrected pipeline, then the report
python scripts/run_e0_reproduction.py --seeds 200 --reps 200 --jobs 2
python scripts/report_e0.py
```

GPU features are computed in the notebooks under `colab/`. A single simulation
experiment can also be run from a YAML file with `d2vi-run` (format in
`src/d2vi/sim/cli.py`).

## Layout

```
src/d2vi/
  data/        ObservedData (analyst view) and ResponseSet (with ground truth); corpus loaders
  features/    detectors and outcome scorers
  sampling/    mixture samplers, audit designs, verifiers
  estimators/  naive, deletion, moment inversion, Hajek, flag-PPI, RC-PPI(++), GREG, legacy v1
  inference/   design-based variance, Rao-Wu bootstrap
  decision/    TOST certification, MSE selection policy
  sim/         seeded Monte Carlo runner, metrics, CLI
  analysis/    endpoints PE1-PE4; post hoc descriptive summaries
  report/      d2vi-report
configs/       frozen protocol and held-out corpus configurations (YAML)
docs/          stage plan, experiment protocol, v1 findings, dataset cards
paper/         theory section with proofs; generated tables, figures and number macros
results/       result tables (index in results/README.md)
tests/         unit, theory and legacy tests
```

## Reproducibility rules

1. Every number in the paper is generated by a script from the result tables in
   `results/`; `scripts/make_tables.py` writes the tables and number macros.
2. Parameters chosen on the development corpora are frozen in `configs/protocol/` before
   the held-out corpora are run; later changes go to the Deviations table of
   `docs/EXPERIMENT_PROTOCOL.md`.
3. Estimators never fall back silently; infeasibility is reported with a reason.
4. Seeds derive from a root seed and the cell key, so designs are compared on identical
   response sets (common random numbers).
5. `d2vi.estimators.legacy` reproduces the first submission's code, including its
   problems, only to quantify how the corrected pipeline changes the published numbers
   (`docs/FINDINGS_V1.md`).
6. Analyses defined after the held-out results were seen are labelled post hoc.

## License and citation

MIT. See `CITATION.cff`.
