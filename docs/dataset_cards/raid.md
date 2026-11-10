# Dataset card: RAID subset (held-out)

Status: subset fixed on 10 November 2026 (`d2vi.data.heldout`), after the protocol freeze and
before any RAID document was scored. Features are built by notebook K05 on Kaggle.

* Source: RAID (Dugan et al., ACL 2024), training split (the test split has no labels);
  files `train_none.csv` and `train.csv` from dataset.raid-bench.xyz. MIT licence.
* Domains (5): news, Reddit posts, IMDb reviews, poetry, book summaries. Poetry and books
  serve the request for creative text.
* Human records: every human document of these domains (at most 2,000 per domain), without
  attack.
* Machine records: ChatGPT, GPT-4, Llama 2 70B chat, Mistral 7B chat without attack; and
  the paraphrase, synonym-swap and homoglyph attacks on ChatGPT and Llama 2 chat. 600
  documents per domain x generator x attack, all decoding settings together, chosen by a
  hash of the document id (order-independent, seed 20261109).
* Calibration (protocol, section 2): cross-domain. For a target domain the Fast-DetectGPT
  threshold is the 95th percentile of human scores in the other four domains; the TF-IDF
  detector is trained on the humans and the unattacked ChatGPT documents of the other four
  domains, so GPT-4, Llama, Mistral and all attacks are shifts.
* Pools: domain x generator x attack (50) x detector.
* Not verified from the build machine: the column names of the RAID files (the host is not
  reachable from it). The loader checks them and the notebook has a format-check cell.
