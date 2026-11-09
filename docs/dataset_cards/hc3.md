# Dataset card — HC3 English (development corpus, Study 2)

| Field | Value |
|---|---|
| Source | Guo et al. (2023), *How Close is ChatGPT to Human Experts?*; `Hello-SimpleAI/HC3`, file `all.jsonl` (SHA-256 in the manifest) |
| License | CC BY-SA 4.0 (dataset card on Hugging Face) |
| Role | Development (Dev) |
| Unit | One answer to a question |
| Classes | Human answers; ChatGPT answers |
| Sources used | The three largest: `reddit_eli5`, `finance`, `open_qa` |
| Sampling | Answers longer than 60 characters; up to 1,500 per class and source, `DataFrame.sample(random_state=0)` as in v1 (`open_qa` has 1,151 human answers) |
| Duplicates | 34 duplicated texts across the sample (inherited from HC3, which repeats some answers); kept so that the v1 sample is reproduced exactly; flagged for the held-out analyses |
| Detector | `openai-community/roberta-base-openai-detector`, zero-shot, first 2,500 characters, threshold 0.5; no detector trained on HC3 is used |
| Outcomes | As for Ghostbuster (VADER, Flesch/100, NRC trust, Pang-Lee subjectivity; second-half variants) |
| Computed in | Colab notebook `colab/F01_hc3_roberta.ipynb` (GPU) |
| File | `data/features/hc3_roberta.parquet` + manifest |
| Reproduction | Operating points equal Table 1 of the first submission; the v1 code path reproduces all 32 cells of Table 3 |
