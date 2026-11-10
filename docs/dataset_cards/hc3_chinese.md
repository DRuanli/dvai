# Dataset card: HC3-Chinese (held-out)

Status: loader fixed on 10 November 2026 (`d2vi.data.multilingual.load_hc3_chinese`) for
`all.jsonl` of huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese. **The file could not be
opened from the development machine**; the loader assumes the format of HC3 English
(`question`, `human_answers`, `chatgpt_answers`, `source`) and notebook K06 checks it before
anything is scored.

* Source: HC3 (Guo et al., 2023), Chinese part: questions with human answers and ChatGPT
  answers.
* Subset: the three sources with the most questions (the rule of the development corpus HC3
  English); at most 2,000 human and 2,000 ChatGPT answers per source, chosen by hash.
* Outcomes, detectors, length filter and calibration as for M4 (addendum A1); answers to the
  same question stay on the same side of the calibration split.
* Pools: source x detector. Unit for PE1: one corpus.
