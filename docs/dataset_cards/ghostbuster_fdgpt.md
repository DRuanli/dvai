# Dataset card — Ghostbuster x Fast-DetectGPT (development corpus, detector pairing)

| Field | Value |
|---|---|
| Documents | Same 8,993 Ghostbuster documents as `ghostbuster.md`; 6 too short for the scorer (< 20 tokens) are dropped |
| Detector | Fast-DetectGPT, analytic criterion, `EleutherAI/gpt-neo-2.7B` as sampling and scoring model, first 400 tokens (Bao et al., 2024) |
| Computed in | Colab notebook `colab/F02_ghostbuster_fastdetectgpt.ipynb` (GPU), output `ghostbuster_fdgpt_scores.parquet` (manifest records versions) |
| Threshold | Per target domain, 95th percentile of human raw scores in the two other domains (v1 rule); `flag_v1` repeats this on the v1 sample (first 600 files per domain and class) |
| Split-text | `score_h1` scores the first half of each document, with its own threshold; outcomes for the second half come from the TF-IDF table |
| Outcomes | Joined from `ghostbuster_tfidf.parquet` by `doc_id` and text hash |
| File | `data/features/ghostbuster_fdgpt.parquet` + manifest (built by `scripts/build_fdgpt_features.py`) |
| Reproduction | Generator-shift sensitivity loss of 24-38 points matches the first submission |
