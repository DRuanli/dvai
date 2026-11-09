# Dataset card — Ghostbuster (development corpus, Study 1)

| Field | Value |
|---|---|
| Source | Verma et al. (2024), *Ghostbuster: Detecting Text Ghostwritten by Large Language Models*; data repository `vivek3141/ghostbuster-data` |
| Version | commit `86ebd72590556a81622986fab736ab9227a948af` (branch `master`) |
| License | CC BY 3.0 (repository `LICENSE`) |
| Role | Development (Dev): used to choose defaults before the protocol is frozen |
| Unit | One document |
| Classes | Human; ChatGPT (`gpt`, generated on matched prompts); Claude (`claude`, generator shift) |
| Domains | Argumentative essays (`essay`), Reuters news (`reuter`, one sub-folder per author), WritingPrompts stories (`wp`) |
| Size after filtering | 1,000 documents per domain and class, except 993 human essays (7 shorter than 61 characters) |
| Filter | Text length > 60 characters after stripping, same rule for every class |
| Duplicates | None (checked on SHA-256 of the text) |
| Detector | TF-IDF (1-2 grams, 30,000 terms, sublinear tf, `min_df = 1`) + logistic regression, trained on the two non-target domains (human vs GPT), threshold 0.5 |
| Leakage control | The target domain never enters detector training; Claude text never enters training |
| Outcomes | VADER compound; Flesch reading ease / 100 (textstat with CMUdict); NRC trust share (lexicon bundled with `nrclex` 4.1.0); Pang-Lee subjectivity classifier (TF-IDF + LR on the NLTK `subjectivity` corpus); second-half sentiment and readability for the split-text design |
| Published file | `data/features/ghostbuster_tfidf.parquet` (ids, scores, outcomes; no text) and its manifest |
| Reproduction | Operating points equal Table 1 of the first submission; the v1 code path reproduces Tables 2 and D.2 (see `docs/reproduction_report.md`) |
