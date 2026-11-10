# Dataset card: Beemo (held-out)

Status: loader fixed on 10 November 2026 (`d2vi.data.heldout.load_beemo`), file
`dataset.parquet` of github.com/Toloka/beemo (SHA-256 5df88f79...a188db, 2,187 prompts).
Features are built by notebook K05 on Kaggle.

* Source: Beemo (Artemova et al., NAACL 2025). Prompts and human answers from No Robots
  (CC-BY-NC-4.0); outputs of ten open instruction-tuned LLMs; expert edits (MIT).
* Versions per prompt: human answer (a = 0); raw LLM output; expert-edited output; output
  edited by Llama 3.1 70B; output edited by GPT-4o (a = 1 for all four). For the two
  LLM-edited versions one of the three editing prompts is chosen per item by hash.
* After the 60-character filter: 2,145 human, 2,178 raw, 2,156 expert-edited, 2,182 and
  2,171 LLM-edited documents.
* Calibration: Beemo's task categories are too small to serve as domains (48-551 human
  answers of at least 30 words), so it is treated as one domain: 30% of the prompts of every
  category form the calibration set, all versions of a prompt together. The TF-IDF detector
  is trained on calibration humans and raw LLM outputs; the edited versions are shifts.
* Pools: version (4) x detector; the ten generators together.
* Edited machine text is labelled machine (a = 1), following the benchmark; the paper
  discusses this choice with the graded-authorship corpora.
