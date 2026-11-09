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
| PE2 | Linearised intervals have near-nominal coverage | Mean coverage over held-out cells; share of cells ≥ 0.90 | Mean in [0.93, 0.97]; ≥ 90% of cells ≥ 0.90 |
| PE3 | TOST controls false certification | Certification rate in cells with \|b_del\| ≥ τ | Upper 95% bound ≤ 0.075 at α = 0.05 |
| PE4 | The policy has the lowest observed worst-case regret | Max regret vs each single estimator; bootstrap by corpus | Descriptive |

Secondary comparisons: Holm correction within each family; Friedman then Wilcoxon–Holm
across corpora; effect sizes with intervals.

## 6. Parameters to be chosen on development corpora

Default audit design; feasibility threshold `k`; power-tuning clip; variance method;
policy variant (`bias_sq`, certification required or not). The certification tolerance
is a substantive choice, not tuned: τ = 0.2 SD(Y), with sensitivity τ ∈ {0.1, 0.3}.

## 7. Deviations

| Date | Change | Reason | Results affected (labelled exploratory) |
|---|---|---|---|
