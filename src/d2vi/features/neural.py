"""Neural detectors (GPU, run in Colab). ``torch`` and ``transformers`` are imported lazily.

* :class:`RobertaOpenAIDetector` - the GPT-2-era supervised detector of Solaiman et al.
  (2019), ``openai-community/roberta-base-openai-detector``, applied zero-shot (Study 2).
* :class:`FastDetectGPT` - the analytic Fast-DetectGPT criterion (Bao et al., 2024) with
  the same model as sampling and scoring model (GPT-Neo-2.7B in the first submission).

Settings mirror the v1 notebooks so that the development results can be reproduced:
texts are cut to 2,500 characters for RoBERTa (512 tokens) and to 400 tokens for
Fast-DetectGPT.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RobertaOpenAIDetector", "FastDetectGPT"]


class RobertaOpenAIDetector:
    model_id = "openai-community/roberta-base-openai-detector"

    def __init__(self, batch_size: int = 64, max_chars: int = 2500, device: int | None = None):
        import torch
        from transformers import pipeline

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline("text-classification", model=self.model_id, device=device,
                              truncation=True, max_length=512, batch_size=batch_size)
        self.max_chars = int(max_chars)

    def predict(self, texts: list[str]) -> np.ndarray:
        out = self._pipe([t[: self.max_chars] for t in texts])
        # label "Fake" = machine-written; return P(machine)
        return np.array([o["score"] if o["label"].lower().startswith("fake") else 1 - o["score"]
                         for o in out])


class FastDetectGPT:
    """Analytic Fast-DetectGPT score (higher = more machine-like); ``nan`` if < 20 tokens."""

    def __init__(self, model_id: str = "EleutherAI/gpt-neo-2.7B", max_tokens: int = 400):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                                          device_map="cuda")
        self.model.eval()
        self.max_tokens = int(max_tokens)
        self.model_id = model_id

    def score_one(self, text: str) -> float:
        torch = self._torch
        with torch.no_grad():
            ids = self.tok(text, return_tensors="pt", truncation=True,
                           max_length=self.max_tokens).input_ids.cuda()
            if ids.shape[1] < 20:
                return float("nan")
            logits = self.model(ids).logits[0, :-1].float()
            labels = ids[0, 1:]
            lp = torch.log_softmax(logits, -1)
            ll = lp.gather(-1, labels[:, None]).squeeze(-1)
            mu = (lp.exp() * lp).sum(-1)
            var = (lp.exp() * lp.pow(2)).sum(-1) - mu.pow(2)
            return float(((ll - mu).sum() / var.sum().clamp_min(1e-6).sqrt()).item())

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.array([self.score_one(t) for t in texts])
