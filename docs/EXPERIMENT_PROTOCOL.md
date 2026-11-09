# Experiment protocol — CANDIDATE FOR FREEZE (second G4 review pending)

This protocol was fixed at the end of stage S4, on the development corpora only, before any
held-out corpus was run. The machine-readable version is `configs/protocol/frozen.yaml`;
`tests/unit/test_protocol_defaults.py` checks that the code defaults equal it. Every later
change goes to the Deviations table (section 8) and the results it affects are labelled
exploratory.

## 1. Corpora

Selection rule (written before running): public corpora of genuine human and genuine
machine text with documented provenance, at least 300 human documents after length
filtering, covering the reviewers' requests (more generators and domains, hybrid
authorship, non-English text, creative or sarcastic text, a real survey outcome).

| Role | Corpus | Use |
|---|---|---|
| Development | Ghostbuster (essay, Reuters, WritingPrompts; human / GPT / Claude) | Original Study 1 |
| Development | HC3 English (three largest sources) | Original Study 2 |
| Held-out | RAID (training split, subset with attacks) | Generators, domains, adversarial edits |
| Held-out | Beemo | Human-edited and LLM-humanised machine text |
| Held-out | APT-Eval | Machine-polished human text, graded |
| Held-out | ANES 2024 open-ended answers + synthetic respondents from three open LLMs | Real closed-ended survey outcome (Study 3) |
| Held-out | HC3-Chinese; M4 (at least two further languages) | Multilingual |
| Held-out | PERSUADE 2.0 + DAIGT | Student responses, many generators |
| Held-out | CoAuthor | Continuous machine share |
| Held-out | SARC-based sarcastic set | Non-literal text |
| Held-out | Real vs generated images (small) | Multimodal demonstration |

Disclosure: before the freeze the ANES human pool was built and its outcome means and
standard deviations were computed (`results/S2_anes_pool_summary.json`); no estimator,
detector or design was run on ANES data.

## 2. Pools, splits and detectors (held-out corpora)

* **No detector sees its evaluation records.** Supervised detectors are trained, and
  zero-shot thresholds calibrated, only on documents that never enter an evaluation pool:
  the other domains of the same corpus when it has at least two domains (the rule of Study
  1), otherwise a seeded 30% calibration split stratified by class, generator and condition
  (`d2vi.data.splits.calibration_split`). Evaluation pools use the remaining records.
* **Detectors.** Every English text corpus is paired with (i) Fast-DetectGPT with
  GPT-Neo-2.7B (400 tokens; threshold = 95th percentile of human calibration scores, the v1
  rule) and (ii) TF-IDF + logistic regression (word 1-2 grams, sublinear tf, 30,000
  features, `min_df = 1`, threshold 0.5) trained on the calibration documents of one
  generator, so that other generators are a generator shift. The RoBERTa OpenAI detector
  (zero-shot, threshold 0.5) is added where Study 2 used it (HC3-type corpora, ANES). RADAR
  and Binoculars (a pair of small open models that fits a T4 GPU; a deviation from the
  Falcon-7B pair) are secondary, where compute allows. Non-English corpora use an XLM-R
  detector trained on English calibration data and applied cross-lingually. The detector's
  continuous score is the propensity model's input; the flag is score > threshold.
* **Unscorable records** (Fast-DetectGPT: fewer than 20 tokens) are excluded from that
  detector's pools; the share is reported.
* **Pools.** One pool per corpus x domain x generator (or source) x detector, as in Studies
  1-2. Mixtures draw without replacement from the evaluation records; a cell runs only when
  the pool holds at least (1 − ε) N human and ε N machine records.
* **ANES (Study 3).** Human records: the 4,926 respondents of
  `results/S2_anes_pool_summary.json`. Machine records: synthetic respondents of three open
  generators (Qwen2.5-7B-Instruct, OLMo-2-1124-7B-Instruct, Phi-3.5-mini-instruct) under four
  conditions (notebook F03), parsed replies with text only. One pool per generator (four
  conditions together) and detector (Fast-DetectGPT, RoBERTa, TF-IDF trained on the human
  and Qwen calibration records). Outcomes: the four thermometers. Descriptive: detector flag
  rates by interview mode; sensitivity analysis with interviewer-administered respondents
  only.

## 3. Methods compared

Naive; deletion; moment inversion (Rogan–Gladen + 2×2); two-phase Hájek; flag-PPI and
flag-PPI++; RC-PPI and RC-PPI++; GREG; the selection policy; certification (TOST and the v1
rule). Audit designs, always at identical budgets and with common random numbers: simple
random; proportional (10 score strata); top-decile split; two-stage Neyman (5 strata);
adaptive constrained Neyman (4 strata); score-rank PPS. Two-stage designs use stage-wise
weights (finding F15). Default design: proportional stratified sampling over 10 detector-score quantile strata (at least two audited records per stratum; fewer strata when m < 20).

**Intention-to-treat (ITT).** "RC-PPI++" in every endpoint means RC-PPI++ when it is
feasible (at least 4 audited records of each class, 2 per training fold)
and flag-PPI on the same audit otherwise; its point estimate and interval are both taken
from the estimator actually used. Replications are never dropped.

## 4. Grid

Primary grid (all primary endpoints): default design; N = 900; iid mixtures; contamination
0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.45; audit budget 2, 5 and 10% (m = 18, 45, 90); 1,000
replications per cell (Monte Carlo SE of a coverage near 0.95 at most 0.007). Every
estimator in a replication uses the same audit; common random numbers across designs.

| Experiment | Design | Contamination | Budget | N | Mixture | Replications |
|---|---|---|---|---|---|---|
| E2, E4, E5, E6 (primary) | default | all seven | 2, 5, 10% | 900 | iid | 1,000 |
| E2b small budget | default | all seven | 1% (m = 9) | 900 | iid | 500 |
| E2c large N | default | 0.05, 0.20 | 5% | 2,000 where pools allow | iid | 500 |
| E3 designs | all six | 0.02, 0.05, 0.10, 0.20, 0.30 | 5% | 900 | iid | 500 |
| E8 contamination patterns | default | 0.05, 0.20 | 5% | 900 | adversarial, clustered, multi-generator | 500 |
| E11 verification error | default | 0.05, 0.20 | 5% | 900 | iid; flips φ0, φ1 ∈ {0, 0.01, 0.03, 0.05, 0.10} | 500 |

The revision plan had 2,000 replications for coverage cells and N = 2,000 throughout; both
were reduced before the freeze for compute.

## 5. Metrics

Standardised error = (estimate − human mean of the response set) / SD of the human
outcome in the pool. Per cell: bias, standardised RMSE (sRMSE), coverage of the 95%
interval with its Monte Carlo SE, interval width relative to the Monte Carlo oracle width
2 · 1.96 · SD(estimate), feasibility rate, false-certification rate, certification power.
Regret of a rule in a cell = its sRMSE minus the lowest sRMSE among the single estimators
(naive, deletion, Hájek, flag-PPI, RC-PPI++) in that cell, in human-SD units ("excess
sRMSE"); the ratio is reported as secondary because it is unstable when the best sRMSE is
near zero.

## 6. Primary endpoints

Scope: the primary grid of section 4 (default design; N = 900; iid mixtures; budgets 2, 5
and 10%; all seven contamination rates) on every held-out text corpus x detector pool and
on ANES. Other designs, mixtures, budgets and N are secondary.

| ID | Claim | Test | Threshold |
|---|---|---|---|
| PE1 | RC-PPI++ (ITT) is non-inferior to flag-PPI on the same audit | Ratio of mean cell standardised MSE; Wilcoxon signed-rank on the per-corpus log ratio; cluster bootstrap by corpus | Upper one-sided 95% bound of the ratio < 1.10 |
| PE2 | The intervals of RC-PPI++ (ITT) are valid | Mean coverage over cells; share of cells with coverage ≥ 0.90; median width ratio | Mean ≥ 0.93 and ≥ 90% of cells ≥ 0.90; width ratio reported |
| PE3 | TOST (τ = 0.2 SD) controls false certification | Certification rate over replications with \|b_del\| ≥ τ; near the boundary (τ ≤ \|b_del\| < 1.1 τ); worst cell | Cluster-bootstrap upper 95% bound ≤ 0.075 overall and near the boundary |
| PE4 | The policy has the lowest worst-case regret | Maximum excess sRMSE over cells, policy vs each single estimator; cluster bootstrap by corpus | Descriptive |

Secondary comparisons: Holm correction within each family; Friedman then Wilcoxon–Holm
across corpora; effect sizes with intervals.

## 7. Choices made on the development corpora

Two rounds. Round 1 (morning of 9 November) is superseded where it used the two-stage
designs, whose weights were wrong (finding F15, found by the independent G4 review); see
`results/README.md`. Round 2 (`run_s4b_design_dev.py`) chose the design and the folds;
round 3 (`run_s4c_final_dev.py`) checked the interval rule and chose the policy, the
certifier settings and the feasibility threshold on that design. Both covered the primary
regime (15 pools: Ghostbuster x TF-IDF and Fast-DetectGPT, HC3 x RoBERTa; N = 900; iid;
contamination 0.01-0.45; budgets 2, 5, 10%). Selection rules were written in the scripts
before they ran.

| Parameter | Candidates | Choice | Evidence (development data) |
|---|---|---|---|
| Audit design | SRS; proportional (10 strata); top-decile split; two-stage Neyman (5); adaptive constrained Neyman (k_target 0, 8, 16; 16 with four second-stage records) | Proportional, 10 score-quantile strata, at least 2 records per stratum | `S4b_design_dev_summary.csv`. Rule: lowest mean cell ITT sRMSE among designs whose ITT interval has mean coverage ≥ 0.93 and ≥ 90% of cells ≥ 0.90. Proportional is the only design that passes at all three budgets (cells ≥ 0.90: 100%, 100%, 98%) and has the lowest sRMSE among those that pass (0.100, 0.058, 0.034 at 2, 5, 10%). SRS is about as accurate (0.100, 0.055, 0.035) but its intervals fail (84%, 74%, 83% of cells ≥ 0.90); top-decile 0.119, 0.058, 0.037 with 89%, 88%, 82% |
| Cross-fitting folds | 2, 5 | 5 | `S4b_design_dev_cells.csv`: with the proportional design, five folds give lower ITT sRMSE in 67% of cells (mean ratio 0.96) and are feasible more often (each training fold holds 4/5 of the audit) |
| Regularisation | C ∈ {1, 0.1} | C = 1 | Round 1, exploratory (`S4_explore_propensity_variance.csv`): C = 0.1 raised RMSE by 10-30% |
| Feasibility threshold | 4 per class (2 per fold); 6 (3 per fold) | 4 / 2 | `S4c_feasibility.csv`: mean cell ITT sRMSE 0.064 vs 0.066; lower in 52% of cells, higher in 26% |
| Fallback when RC-PPI++ is infeasible | two-phase Hájek; flag-PPI | Flag-PPI on the same audit | Recomputed from the round-3 replications: mean cell ITT sRMSE 0.064 (flag-PPI) vs 0.122 (Hájek); at 1% contamination 0.047 vs 0.159. RC-PPI++ is feasible in 0%, 0%, 10%, 69% and 100% of replications at 1, 2, 5, 10 and ≥ 20% contamination (budget 5%) |
| Interval rule | linearised (df = m − H); Satterthwaite df; shrinkage to the pooled stratum variance; shrinkage to the anticipated variance with Jeffreys stratum rates (round 1 also: empirical-Bayes rates, a floor, Rao-Wu) | Shrinkage (one pseudo-record) to the variance anticipated under a Bernoulli origin model with Jeffreys stratum rates; Satterthwaite df | `S4c_interval_rules.csv`, proportional design: the only rule valid for both estimators the ITT interval can come from. Cells ≥ 0.90 at 2/5/10%: flag-PPI 100/99/96% (linearised 20/39/67%, pooled 28/54/77%); feasible RC-PPI++ 100/100/99% (pooled 91/95/96%). Cost: median width 1.65, 1.22, 1.10 times the oracle width for RC-PPI++. Round 1 on the valid designs (SRS, top-decile): empirical-Bayes rates never improved the worst cell (e.g. HC3 top-decile 0.857 vs 0.880) |
| Policy | squared bias plugin / debiased; certification required or not | Debiased, certification not required | `S4c_regret.csv`. Rule: lowest worst-cell excess sRMSE, ties by pooled sRMSE. Debiased 0.166 SD worst excess, pooled sRMSE 0.072 (lowest of all rules; RC-PPI++ ITT 0.209 / 0.079, flag-PPI 0.284 / 0.100, naive 0.755 / 0.161, deletion 0.846 / 0.185); requiring certification 0.167 / 0.074; plugin 0.190 / 0.074. The interval the policy reports covers 97.9% on average, 98.9% of cells ≥ 0.90 |
| Certifier | TOST τ ∈ {0.1, 0.2, 0.3} SD, Student or normal quantile, with or without fallback; v1 rule | TOST, τ = 0.2 SD (substantive, not tuned), Student quantile with the anchor's Satterthwaite df, flag-PPI anchor when RC-PPI++ is infeasible | `S4c_certification.csv`: false certification 0.38% of 22,668 replications with \|b_del\| ≥ τ (cluster-bootstrap upper bound 0.45%); near the boundary 1.4% (upper 1.7%); worst cell 5.1%; certifies 72% of replications with \|b_del\| < τ/2. The v1 rule certifies 0.1% of all replications |

### Changes made on development data before the freeze

* F14 (morning): the first stratum-variance shrinkage shrank towards a pooled sum of
  squares; found when HC3 intervals were about nine times too wide; corrected and covered
  by a regression test.
* F15 (G4 review): the two-stage designs used weights that are not inverse inclusion
  probabilities; the round-1 choice of the adaptive design was void. Stage-wise weights
  and label-correlated unbiasedness tests were added and the design comparison was rerun
  (round 2), which chose the proportional design.
* PE2 first required mean coverage in [0.93, 0.97]. Coverage above nominal is not a
  validity failure, so PE2 keeps the lower bounds and reports the width separately.
* The fallback was first the two-phase Hájek estimator; it dominated the error at 1-2%
  contamination, where RC-PPI++ is mostly infeasible.
* Primary endpoints were restricted to the regime tested on development data (G4 review,
  M3); PE4 is measured as excess sRMSE in human-SD units (M4); the detector and threshold
  rule for held-out corpora was written down (M5); the policy default in the code was
  aligned with the protocol and a test now checks all defaults (M1).
* Known limitations: (i) at 1-2% contamination RC-PPI++ is almost never feasible with
  18-90 audited records, so the ITT rule reports flag-PPI there; (ii) the chosen interval
  rule is conservative at small budgets (median width 1.65 times the oracle at 2%);
  (iii) in the worst development cells coverage is 0.80-0.83 (TF-IDF and Fast-DetectGPT,
  10% budget, 1-2% contamination, where the fallback flag-PPI interval is used).

## 8. Deviations

| Date | Change | Reason | Results affected (labelled exploratory) |
|---|---|---|---|
