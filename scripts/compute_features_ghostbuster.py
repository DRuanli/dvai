"""Compute the Study 1 feature table for Ghostbuster (CPU only).

Usage::

    git clone --depth 1 https://github.com/vivek3141/ghostbuster-data data/raw/ghostbuster-data
    curl -sSo data/raw/subjectivity.zip \
        https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/subjectivity.zip
    unzip -q data/raw/subjectivity.zip -d data/raw
    # textstat >= 0.7.10 counts syllables with CMUdict through NLTK
    curl -sSo data/raw/nltk_data/corpora/cmudict.zip \
        https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/cmudict.zip
    unzip -q data/raw/nltk_data/corpora/cmudict.zip -d data/raw/nltk_data/corpora
    export NLTK_DATA=data/raw/nltk_data
    python scripts/compute_features_ghostbuster.py --root data/raw/ghostbuster-data \
        --subj-dir data/raw/subjectivity --out data/features/ghostbuster_tfidf.parquet --min-df 1

For each target domain the TF-IDF detector is trained on the two other domains, human vs
GPT only, and applied to every document of the target domain (human, GPT and Claude), so
the Claude documents form the generator-shift condition. Outcomes are computed on the
full text; the split-text variant scores the detector on the first half of a document and
the outcomes on the second half. No text is written to the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from d2vi import __version__
from d2vi.data.loaders import GHOSTBUSTER_DOMAINS, load_ghostbuster
from d2vi.features.detectors import TfidfLRDetector
from d2vi.features.outcomes import (
    FleschReadingEase,
    NRCTrust,
    PangLeeSubjectivity,
    VaderSentiment,
    split_halves,
)


def _version(pkg: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(pkg)
    except PackageNotFoundError:  # pragma: no cover
        return "missing"


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:  # pragma: no cover - not a git clone
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--subj-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-df", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    t0 = time.time()

    docs = load_ghostbuster(args.root)
    texts = docs["text"].tolist()
    halves = [split_halves(t) for t in texts]
    first, second = [h[0] for h in halves], [h[1] for h in halves]

    scorers = [VaderSentiment(), FleschReadingEase(), NRCTrust(),
               PangLeeSubjectivity(args.subj_dir)]
    feats = pd.DataFrame({"length": [len(t.split()) for t in texts]})
    for sc in scorers:
        feats[f"y_{sc.name}"] = sc.score(texts)
    for sc in scorers[:2]:
        feats[f"y_{sc.name}_h2"] = sc.score(second)

    feats["score"] = np.nan
    feats["score_h1"] = np.nan
    detector_params = None
    for dom in GHOSTBUSTER_DOMAINS:
        train = docs[(docs["domain"] != dom) & docs["source"].isin(["human", "gpt"])]
        det = TfidfLRDetector(min_df=args.min_df).fit(train["text"].tolist(),
                                                      train["a"].to_numpy())
        detector_params = det.params
        target = np.flatnonzero(docs["domain"].to_numpy() == dom)
        feats.loc[target, "score"] = det.predict([texts[i] for i in target])
        feats.loc[target, "score_h1"] = det.predict([first[i] for i in target])
    feats["flag"] = (feats["score"] > args.threshold).astype(int)
    feats["flag_h1"] = (feats["score_h1"] > args.threshold).astype(int)

    keep = ["doc_id", "corpus", "domain", "source", "a", "generator", "text_sha256"]
    out = pd.concat([docs[keep].reset_index(drop=True), feats], axis=1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    ops = []
    for (dom, src), g in out.groupby(["domain", "source"]):
        ops.append({"domain": dom, "source": src, "n": int(len(g)),
                    "flag_rate": round(float(g["flag"].mean()), 4)})
    manifest = {
        "corpus": "ghostbuster-data",
        "corpus_commit": _git_head(Path(args.root)),
        "detector": "tfidf_lr (cross-domain, trained on human vs gpt of the other two domains)",
        "detector_params": detector_params,
        "threshold": args.threshold,
        "outcomes": [s.name for s in scorers],
        "flag_rates": ops,
        "rows": int(len(out)),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "d2vi_version": __version__,
        "sklearn": sklearn.__version__,
        "textstat": _version("textstat"),
        "vaderSentiment": _version("vaderSentiment"),
        "nrclex": _version("nrclex"),
        "python": platform.python_version(),
        "seconds": round(time.time() - t0, 1),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(ops).pivot(index="domain", columns="source", values="flag_rate"))
    print(out_path, manifest["rows"], "rows", manifest["seconds"], "s")


if __name__ == "__main__":
    main()
