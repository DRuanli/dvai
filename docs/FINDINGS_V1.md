# Findings on the first-submission code (v1)

Audit of the v1 repository (`d2vi`, commit `658b1ff`) and its four Colab notebooks, done
at the start of the revision. Each item is reproduced in `d2vi.estimators.legacy` where it
affects numbers, so that its effect on the published tables can be measured (stage S3).

| ID | Finding | Evidence | Effect | Fixed in |
|---|---|---|---|---|
| F1 | No code for the main tables, the policy/regret evaluation, coverage and the inflation factor, split-text, the exogenous outcome or the two-detector calibration | README lists `experiments/`, `results/`, `paper/`, `docs/`; none exist | Results not reproducible | This repository |
| F2 | "Stratified RC-PPI" column reports flag-PPI when RC-PPI abstains | `rc_strat = rs if not np.isnan(rs) else p` in all notebooks; identical values in Table D.2 (e.g. essay/claude/0.05: 0.0983 in both columns) | Mixes two estimators in the main tables, the primary endpoint and the regret | `EstimateResult` has no fallback; only the policy substitutes, and it records the choice |
| F3 | Flag-PPI and stratified RC-PPI use different audits | Bernoulli(5%) audit of random size vs fixed `max(round(0.05 n), 20)` stratified audit from another random stream | Comparisons not budget-matched or paired | Designs return exactly `m` records; all estimators share one audit per replication |
| F4 | Cross-fitting inconsistent with cross-PPI | Unaudited records predicted by the full-audit model, residuals by fold models | Small finite-sample bias | Fold-averaged predictions for unaudited records |
| F5 | The report tool's bootstrap is not a bootstrap | `_rc(df, pd.Index(np.unique(bi)), w, ...)` keeps unique records (~63% of the audit) with unchanged HT weights; not stratified | Wrong intervals in the tool | Linearised design variance; Rao–Wu bootstrap |
| F6 | Operating points in the report are unweighted | `s_hat`, `f_hat` are plain means over a stratified audit | Biased `s`, `f` under stratified audits | Design-weighted estimates |
| F7 | Certification rule has no error control | Certify if `abs(b) < 2 sd` and `2 sd < tau`, `tau = 0.15 SD(Y)` hard-coded | Unknown false-certification rate | TOST (Proposition 5) |
| F8 | HC3 style stratum built from true labels | `G = len > median(len of human texts)` | Violates the observable-strata assumption | Strata from observables only |
| F9 | Configurations differ across studies and from the paper | N = 600/300/450 vs 900; 6 seeds vs 8/10; audit 7.5%; `min_df = 2` not in code; Fast-DetectGPT uses a 95th-percentile threshold, not `c = 0.5` | Paper misdescribes settings | One configuration file per experiment |
| F10 | Few replications | 6–10 seeds per cell | Large Monte Carlo error; per-cell tests underpowered | 1,000–2,000 replications per cell |
| F11 | Metadata inconsistent | README mentions another venue and "12x"; package metadata lists a different co-author from the manuscript | Confusing for reviewers | Metadata rewritten; authors Nguyen Le and My Vu, as in the manuscript |
| F12 | Package and paper diverge | Core module implements moment correction and a simulator; RC-PPI only inside the CLI | Tool does not implement the paper | RC-PPI is the core estimator |
| F13 | Undisclosed power tuning | With a random audit, `ppi_py.ppi_mean_pointestimate` is called with its default `lam=None` (PPI++); the stratified branch uses `lam = 1` | Paper describes both as standard PPI | `FlagPPI(tune=...)` and `RCPPI(tune=...)` are explicit |
