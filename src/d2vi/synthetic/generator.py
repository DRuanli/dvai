"""Generation of synthetic respondents with open-weight chat models (GPU, run in Colab).

:class:`HFChatGenerator` wraps a Hugging Face chat model (``torch`` and ``transformers``
are imported lazily). :class:`SyntheticSurvey` drives it: one persona per respondent, one
reply per persona, a resumable CSV cache, one retry with a new seed for replies that do not
parse, and a table with one row per synthetic respondent.

Models used in Study 3 are ungated and fit a free Colab T4 GPU (16 GB): 7-8B models in
4-bit NF4, smaller models in float16.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from .instrument import SurveyInstrument
from .persona import Persona
from .prompt import RespondentPrompt, parse_answer, respondent_document

__all__ = ["HFChatGenerator", "SyntheticSurvey", "GENERATORS"]

# model id -> load in 4 bit?
GENERATORS = {
    "Qwen/Qwen2.5-7B-Instruct": True,
    "allenai/OLMo-2-1124-7B-Instruct": True,
    "microsoft/Phi-3.5-mini-instruct": False,
}


class HFChatGenerator:
    """Batched sampling from a chat model."""

    def __init__(self, model_id: str, load_4bit: bool | None = None, batch_size: int = 16,
                 max_new_tokens: int = 450, temperature: float = 0.9, top_p: float = 0.95):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        load_4bit = GENERATORS.get(model_id, True) if load_4bit is None else load_4bit
        kw = {"device_map": "auto", "torch_dtype": torch.float16}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        self.batch_size = int(batch_size)
        self.gen_kw = {"max_new_tokens": int(max_new_tokens), "do_sample": True,
                       "temperature": float(temperature), "top_p": float(top_p),
                       "pad_token_id": self.tok.pad_token_id}
        self.settings = {"model": model_id, "load_4bit": bool(load_4bit), **{
            k: v for k, v in self.gen_kw.items() if k != "pad_token_id"}}

    def generate(self, conversations: list[list[dict]], seed: int) -> list[str]:
        import torch

        out = []
        for b in range(0, len(conversations), self.batch_size):
            chunk = conversations[b:b + self.batch_size]
            texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                     for c in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
            torch.manual_seed(seed + b)
            with torch.no_grad():
                gen = self.model.generate(**enc, **self.gen_kw)
            new = gen[:, enc["input_ids"].shape[1]:]
            out.extend(self.tok.batch_decode(new, skip_special_tokens=True))
        return out


@dataclass
class SyntheticSurvey:
    instrument: SurveyInstrument
    prompt: RespondentPrompt
    cache_path: str
    seed: int = 0

    def _cache(self) -> pd.DataFrame:
        if os.path.exists(self.cache_path):
            return pd.read_csv(self.cache_path, dtype={"source_id": str}, keep_default_na=False)
        return pd.DataFrame(columns=["resp_id", "source_id", "attempt", "raw"])

    def run(self, generator: HFChatGenerator, personas: list[Persona], prefix: str,
            chunk: int = 64) -> pd.DataFrame:
        """Generate (resumably) and return one parsed row per persona."""
        cache = self._cache()
        ids = [f"{prefix}-{i:05d}" for i in range(len(personas))]
        by_id = dict(zip(ids, personas, strict=True))
        for attempt in (0, 1):
            parsed = {r.resp_id: parse_answer(r.raw, self.instrument).ok
                      for r in cache.itertuples() if r.attempt == attempt}
            if attempt == 0:
                todo = [i for i in ids if i not in parsed]
            else:
                failed0 = [i for i in ids if not _ok_any(cache, i, self.instrument)]
                todo = [i for i in failed0 if i not in parsed]
            for s in range(0, len(todo), chunk):
                part = todo[s:s + chunk]
                replies = generator.generate([self.prompt.messages(by_id[i]) for i in part],
                                             seed=self.seed + 1_000_003 * attempt + s)
                new = pd.DataFrame({"resp_id": part,
                                    "source_id": [by_id[i].source_id for i in part],
                                    "attempt": attempt, "raw": replies})
                cache = pd.concat([cache, new], ignore_index=True)
                cache.to_csv(self.cache_path, index=False)
        return self.table(cache, ids, generator.model_id)

    def table(self, cache: pd.DataFrame, ids: list[str], model_id: str) -> pd.DataFrame:
        rows = []
        for rid in ids:
            got = cache[cache.resp_id == rid].sort_values("attempt")
            best = None
            for r in got.itertuples():
                p = parse_answer(r.raw, self.instrument)
                if p.ok:
                    best = (r, p)
                    break
            if best is None:
                r = got.iloc[-1]
                rows.append({"resp_id": rid, "source_id": r.source_id, "parse_ok": False,
                             "error": parse_answer(r.raw, self.instrument).error})
                continue
            r, p = best
            rows.append({"resp_id": rid, "source_id": r.source_id, "parse_ok": True,
                         "attempt": r.attempt, **p.thermometers,
                         **{f"text_{k}": v for k, v in p.open.items()},
                         "text": respondent_document(p.open, self.instrument),
                         "generator": model_id, "condition": self.prompt.condition})
        return pd.DataFrame(rows)


def _ok_any(cache: pd.DataFrame, rid: str, instrument: SurveyInstrument) -> bool:
    return any(parse_answer(r, instrument).ok for r in cache.loc[cache.resp_id == rid, "raw"])
