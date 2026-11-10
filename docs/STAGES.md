# Revision stages

The critical path is S0 → S1 → S3 → S4 → S5 → S7 → S8 → S9 (about 22 working days);
S2 (features) and S6 (theory) run in parallel. Every stage ends with a gate; work does
not move on until the gate passes or the deviation is written down.

| Stage | Work | Output | Gate |
|---|---|---|---|
| S0 Setup | Repository, license, packaging, CI, docs skeleton; author metadata; data accounts (ANES, Kaggle, LLM APIs); triage of reviewer-suggested references | This repository | CI green |
| S1 Core library | Data containers, mixture samplers, five audit designs, cross-fitted propensity, estimators (incl. legacy v1), design-based variance, Rao–Wu bootstrap, TOST certification, selection policy, metrics, runner | `src/d2vi` | G1: unit, equivalence and theory tests pass; independent code review |
| S2 Features | Loader, download script, checksum and dataset card per corpus; Colab notebooks for neural detectors; development corpora first (Ghostbuster, HC3), then held-out corpora; synthetic ANES respondents | Feature parquet files + manifests | G2: operating points plausible; no duplicates; no overlap between detector training data and evaluation data |
| S3 Reproduction (E0) | Run the legacy v1 pipeline on the development corpora; compare with the published tables; write the reproduction report | `docs/reproduction_report.md` | G3: agreement within Monte Carlo error, or every gap explained |
| S4 Development and freeze | E1–E6 on development corpora; choose defaults (audit design, feasibility threshold, power tuning, variance method, policy rule); write `configs/protocol/frozen.yaml` and the protocol; tag `protocol-freeze` | Frozen protocol | G4: independent review before tagging |
| S5 Held-out evaluation | E1–E14 on held-out corpora, run once; deviations logged | `results/raw/` | G5: all cells complete; Monte Carlo SE on target; PE1–PE4 computed as registered |
| S6 Theory | Statements and proofs of Lemma 1, Propositions 1–6, Theorems 1–2, Corollary 1; a numerical check per result | Appendices A–B, `tests/theory` | G6: checks pass; independent cross-reading |
| S7 Analysis | Generated LaTeX tables, PDF figures, regime map; claim–evidence matrix | `paper/generated/` | G7: every claim has evidence |
| S8 Writing | Revised manuscript with highlighted changes; highlights; AI-use declaration; point-by-point response with page/line references; language editing | Manuscript, response letter | G8: full cross-reading |
| S9 Check and submit | Clean-clone reproduction of at least one main table; independent referee-style audit; Zenodo DOI; submission | Submission package | G9: no critical issue open |

## Status

| Stage | Status (10 November 2026) |
|---|---|
| S0, S1 | Done |
| S2 | Development corpora done; ANES human pool built; synthetic respondents and detector scores running on Kaggle (K03); RAID and Beemo subsets defined, features to be computed on Kaggle (K05); multilingual and other registered corpora not prepared |
| S3 | Done: Tables 2, D.2 and 3 of the first submission reproduced exactly |
| S4 | Done: protocol frozen (commit `d091b5b`) after two G4 review rounds (findings F14-F17); one deviation since (F18) |
| S5 | Infrastructure written and rehearsed on development data before any held-out run (`scripts/run_s5.py`, `scripts/report_endpoints.py`, `configs/s5/`); no held-out run yet |
| S6 | Done: `paper/theory.tex`, two rounds of independent review (finding F18), 34 checks in `tests/theory` |
| S7 | Pending |
| S8 | Started: response letter skeleton and result-independent manuscript sections (kept outside this repository) |
| S9 | Pending |
