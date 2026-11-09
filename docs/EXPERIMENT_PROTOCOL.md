# Experiment protocol — DRAFT (not frozen)

This protocol is frozen at the end of stage S4, before any held-out corpus is run. The
frozen version is committed together with `configs/protocol/frozen.yaml` and tagged
`protocol-freeze`. Later changes go to the Deviations table.

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
| Held-out | ANES open-ended answers + synthetic respondents from at least three current LLMs | Real closed-ended survey outcome |
| Held-out | HC3-Chinese; M4 (at least two further languages) | Multilingual |
| Held-out | PERSUADE 2.0 + DAIGT | Student responses, many generators |
| Held-out | CoAuthor | Continuous machine share |
| Held-out | SARC-based sarcastic set | Non-literal text |
| Held-out | Real vs generated images (small) | Multimodal demonstration |

## 2. Methods compared

Naive; deletion; moment inversion (Rogan–Gladen + 2×2); two-phase Hájek; flag-PPI and
flag-PPI++; RC-PPI and RC-PPI++; GREG; selection policy (variants); certification (TOST
and legacy rule). Audit designs: simple random, proportional (10 strata), top-decile
split, two-stage Neyman (5 strata), score-rank PPS — always at identical budgets.

## 3. Grid

Contamination rate ∈ {0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.45}; budget ∈ {1, 2, 5, 10}%;
N ∈ {900, 2000} where pools allow; mixtures: iid, adversarial, clustered, multi-generator.
Replications: 1,000 per cell for point metrics, 2,000 for coverage.

## 4. Metrics

Bias, RMSE, standardised RMSE (divided by the human SD of the pool), coverage with Monte
Carlo SE, interval width, feasibility rate, false-certification rate, certification
power, regret (max, mean, P90, ratio) against the best non-oracle candidate per cell.

## 5. Primary endpoints (held-out corpora only)

| ID | Claim | Test | Threshold |
|---|---|---|---|
| PE1 | RC-PPI++ is non-inferior to flag-PPI on the same audit | Ratio of pooled MSE; Wilcoxon signed-rank on per-corpus log ratio; cluster bootstrap by corpus | Margin 1.10, one-sided α = 0.05 |
| PE2 | Design-based intervals are valid | Mean coverage over held-out cells; share of cells ≥ 0.90; median width relative to the cell's Monte Carlo oracle width 2·1.96·SD(estimate) | Mean ≥ 0.93 and ≥ 90% of cells ≥ 0.90; median width ratio reported (target ≤ 1.25) |
| PE3 | TOST controls false certification | Certification rate in cells with \|b_del\| ≥ τ | Upper 95% bound ≤ 0.075 at α = 0.05 |
| PE4 | The policy has the lowest observed worst-case regret | Max regret vs each single estimator; bootstrap by corpus | Descriptive |

Secondary comparisons: Holm correction within each family; Friedman then Wilcoxon–Holm
across corpora; effect sizes with intervals.

## 6. Parameters to be chosen on development corpora

Default audit design; feasibility threshold `k`; power-tuning clip; interval rule;
policy variant (`bias_sq`, certification required or not). The certification tolerance
is a substantive choice, not tuned: τ = 0.2 SD(Y), with sensitivity τ ∈ {0.1, 0.3}.

Choices on the development corpora (Ghostbuster with TF-IDF and Fast-DetectGPT; HC3 with
RoBERTa), 9 November 2026 (`configs/protocol/draft.yaml`):

| Parameter | Candidates | Choice | Evidence |
|---|---|---|---|
| Interval rule | linearised (df = m − H); Satterthwaite df; stratum variances shrunk to the pooled variance; shrunk to the variance anticipated under a Bernoulli origin model with Jeffreys or empirical-Bayes (0.5, 1, 2, 4 pseudo-records) stratum rates; floor at the anticipated variance; Rao–Wu bootstrap | Shrinkage (one pseudo-record) to the anticipated variance with Jeffreys stratum rates, Satterthwaite df | `results/S4_variance_dev_summary.csv`; RC-PPI++ under the adaptive design: mean coverage 0.966 / 0.983 / 0.970, cells ≥ 0.90: 94% / 100% / 98%, median width ratio 1.14 / 1.17 / 1.13 (Fast-DetectGPT / HC3 / TF-IDF). Empirical-Bayes rates give narrower intervals at 5% contamination but worst-cell coverage 0.70–0.80; the floor is 1.4–2.3 times wider than needed |
| Audit design | SRS; top-decile split; two-stage Neyman; adaptive constrained Neyman | Adaptive, `k_target = 16` | `results/S4_design_dev_summary.csv`, `results/S4_tuning_dev_summary.csv`; intention-to-treat RC-PPI++ (flag-PPI when infeasible), mean standardised RMSE over cells 0.044 (adaptive) vs 0.054 (SRS), 0.054 (Neyman), 0.057 (top-decile); adaptive is more than 10% better than SRS in 62% of cells and more than 10% worse in 3% (TF-IDF, essays, Claude, 5% contamination). `k_target` 12 and 16 tie (0.054 both) |
| Feasibility threshold | 6 (3 per fold); 4 (2 per fold) | 4 (2 per fold) | `results/S4_tuning_dev_summary.csv`: lower intention-to-treat error in every design, corpus and contamination level; coverage unchanged |
| Cross-fitting folds, regularisation | K ∈ {2, 5}, C ∈ {1, 0.1} | K = 2, C = 1 | `results/S4_explore_propensity_variance.csv`: in feasible cells 5 folds lower RMSE by about 4% and C = 0.1 raises it by 10–30%; neither fixes coverage. Two audited records of each class per fold need 4 machine records with K = 2 but 10 with K = 5: under the adaptive design at 5% contamination that is 91% versus 60% of replications |
| Fallback when RC-PPI++ is infeasible | two-phase Hájek; flag-PPI | Flag-PPI (same audit) | `results/S4_policy_dev_regret_hajekfb.csv`: at 1% contamination RC-PPI++ is feasible in about 22% of replications; standardised RMSE with the Hájek fallback 0.22–0.23, with the flag-PPI fallback 0.012–0.115 |
| Policy variant | `bias_sq` plugin / debiased; certification required or not | Debiased, certification not required | `results/S4_policy_dev_regret.csv`: pooled standardised RMSE 0.052 (lowest of all rules; RC-PPI++ 0.058, flag-PPI 0.084, naive 0.112, deletion 0.173), worst excess over the best single estimator of a cell 0.189 SD (RC-PPI++ 0.231, naive 0.481, deletion 0.847); plugin 0.055 / 0.210; requiring certification 0.052 / 0.193 |
| Certifier | TOST τ ∈ {0.1, 0.2, 0.3} SD, Student or normal quantile, with or without fallback; v1 rule | TOST, τ = 0.2 SD, Student quantile, flag-PPI fallback | `results/S4_policy_dev_certification.csv`: false certification 0.35% (upper 95% bound 0.48%) of 7,788 replications with \|b_del\| ≥ τ; certifies 92% of replications with \|b_del\| < τ/2. The v1 rule never certifies with 45 audited records |

Changes made on development data before the freeze (no held-out corpus had been run):

* An interim choice of the morning of 9 November ("shrunk stratum variances +
  Satterthwaite") rested on an implementation error that shrank towards a pooled sum of
  squares (finding F14). It was found when HC3 intervals came out about nine times wider
  than needed, corrected, covered by a regression test, and the comparison was rerun.
* PE2 originally required mean coverage in [0.93, 0.97]. On the development corpora the
  valid rules are conservative at 5% contamination under the adaptive design, because a
  stratum with few audited records cannot tell a small machine share from zero. Coverage
  above nominal is not a validity failure, so PE2 now has a validity criterion (unchanged
  lower bounds) and reports the cost in interval width separately.
* The fallback was first set to the two-phase Hájek estimator. The policy run on
  development data showed that this fallback dominates the error at 1–2% contamination,
  where RC-PPI++ is mostly infeasible with 45 audited records; flag-PPI is design-unbiased
  on the same audit and its intervals are valid under the chosen interval rule.
* Known limitation: in the worst development cells (Fast-DetectGPT, WritingPrompts, GPT,
  readability; TF-IDF, essays, Claude, readability) coverage of RC-PPI++ is 0.85–0.89. The
  standard error is close to the Monte Carlo SD there; the misses are one-sided (skewed
  errors of a ratio estimator with rare machine records), which a symmetric interval
  cannot correct.

## 7. Deviations

| Date | Change | Reason | Results affected (labelled exploratory) |
|---|---|---|---|
