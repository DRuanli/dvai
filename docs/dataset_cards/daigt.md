# Dataset card: DAIGT v2 with PERSUADE (held-out)

Status: loader fixed on 10 November 2026 (`d2vi.data.multilingual.load_daigt`) for
`train_v2_drcat_02.csv` of the Kaggle dataset "DAIGT V2 Train Dataset" (thedrcat). **The
file could not be opened from the development machine**; the loader requires the columns
`text`, `label`, `prompt_name`, `source` and notebook K06 prints the sources before anything
is scored.

* Source: essays of the PERSUADE 2.0 corpus (students, grades 6-12) as the human class, and
  essays written by several LLMs for the same prompts, collected for the Kaggle competition
  "LLM - Detect AI Generated Text".
* Subset: at most 4,000 human essays with source `persuade_corpus`; the eight largest
  machine sources with at least 700 essays, 1,000 essays each; all by hash of the text.
* Outcomes and detectors: as for the English corpora (four text outcomes; Fast-DetectGPT and
  TF-IDF). The TF-IDF detector is trained on the calibration essays of the largest machine
  source (ties: first by name), so the other sources are generator shifts.
* Calibration: single-domain rule, 30% split stratified by source.
* Pools: machine source x detector. Unit for PE1: one corpus.
