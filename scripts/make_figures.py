"""S7 figures from compact cell tables (primary grid) and the secondary summary.

    python scripts/make_figures.py OUT_DIR SECONDARY_CSV CELLS_DIR [CELLS_DIR ...]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
INK, MUTED = "#222222", "#888888"
RULES = {"naive": ("Naive", "#9e9e9e"), "deletion": ("Detect-and-delete", "#d55e00"),
         "flag_ppi": ("Flag-PPI", "#56b4e9"), "itt": ("RC-PPI++ (ITT)", "#0072b2"),
         "policy_debiased": ("Policy", "#009e73")}
NAMES = {"anes2024": "ANES 2024", "beemo": "Beemo", "m4": "M4", "hc3zh": "HC3-Chinese",
         "daigt": "DAIGT", "raid": "RAID"}


def load(dirs):
    files = [f for d in dirs for f in glob.glob(f"{d}/*.cells.parquet")]
    c = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    c["mse_itt"] = c["mse_rc_ppi_pp"]
    return c


def regime_map(c, out):
    eps, bud = sorted(c.eps.unique()), sorted(c.budget.unique())
    rules = ["naive", "deletion", "flag_ppi", "itt"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    win = np.zeros((len(eps), len(bud)), dtype=int)
    share = np.zeros_like(win, dtype=float)
    cov = np.zeros_like(share)
    for i, e in enumerate(eps):
        for j, b in enumerate(bud):
            s = c[(c.eps == e) & (c.budget == b)]
            best = s[[f"mse_{r}" for r in rules]].values.argmin(axis=1)
            counts = np.bincount(best, minlength=len(rules))
            win[i, j] = counts.argmax()
            share[i, j] = counts.max() / counts.sum()
            cov[i, j] = s.cover_itt.mean()
    ax = axes[0]
    for i in range(len(eps)):
        for j in range(len(bud)):
            name, col = RULES[rules[win[i, j]]]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=col, alpha=0.35 + 0.6 * share[i, j]))
            ax.text(j + .5, i + .5, f"{name.split(' ')[0]}\n{share[i, j]:.0%}", ha="center",
                    va="center", fontsize=6.5, color=INK)
    ax.set_xlim(0, len(bud)); ax.set_ylim(0, len(eps))
    ax.set_xticks(np.arange(len(bud)) + .5, [f"{b:.0%}" for b in bud])
    ax.set_yticks(np.arange(len(eps)) + .5, [f"{e:.0%}" for e in eps])
    ax.set_xlabel("audit budget"); ax.set_ylabel("AI share ε")
    ax.set_title("(a) rule with the lowest MSE (share of cells)", loc="left")
    ax = axes[1]
    im = ax.imshow(cov, origin="lower", cmap="Blues", vmin=0.85, vmax=1.0, aspect="auto")
    for i in range(len(eps)):
        for j in range(len(bud)):
            ax.text(j, i, f"{cov[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if cov[i, j] > 0.95 else INK)
    ax.set_xticks(range(len(bud)), [f"{b:.0%}" for b in bud])
    ax.set_yticks(range(len(eps)), [f"{e:.0%}" for e in eps])
    ax.set_xlabel("audit budget"); ax.set_title("(b) coverage of the ITT 95% interval", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.savefig(out / "fig_regime_map.pdf"); fig.savefig(out / "fig_regime_map.png")


def corpus_ratios(c, out):
    c = c.assign(dom=c.pool.str.split("__").str[1].str.split("-").str[0])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    order = ["raid", "daigt", "beemo", "m4", "anes2024", "hc3zh"]
    for k, corp in enumerate(o for o in order if o in set(c.corpus)):
        s = c[c.corpus == corp]
        per_pool = s.groupby("pool")[["mse_itt", "mse_flag_ppi"]].mean()
        r = per_pool.mse_itt / per_pool.mse_flag_ppi
        ax.scatter(r, np.full(len(r), k) + np.random.default_rng(k).uniform(-.18, .18, len(r)),
                   s=6, color=RULES["itt"][1], alpha=.35, lw=0)
        tot = s.mse_itt.mean() / s.mse_flag_ppi.mean()
        ax.plot([tot], [k], "|", ms=14, mew=2, color=INK)
        ax.text(1.32, k, f"{tot:.2f}", va="center", fontsize=7)
    ax.axvline(1, color=MUTED, lw=.8, ls="--")
    ax.axvline(1.10, color="#d55e00", lw=.8, ls=":")
    ax.set_yticks(range(k + 1), [NAMES[o] for o in order if o in set(c.corpus)])
    ax.set_xlim(0.2, 1.3); ax.set_xlabel("MSE of RC-PPI++ (ITT) / MSE of flag-PPI")
    ax.set_title("PE1: dots = pools, bar = corpus", loc="left")
    fig.savefig(out / "fig_pe1_corpora.pdf"); fig.savefig(out / "fig_pe1_corpora.png")


def secondary(sec, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"wspace": 0.45})
    d = sec[sec.grid == "designs"].copy()
    d["lab"] = d.level.str.split("_").str[0].replace({"top": "top decile", "adaptive": "adaptive",
                                                       "proportional": "proportional"})
    d = d.sort_values("width_mean_sd")
    ax = axes[0]
    ax.barh(d.lab, d.cover_mean, color=RULES["itt"][1], alpha=.8)
    ax.scatter(d.cover_min, d.lab, color="#d55e00", s=12, zorder=3, label="lowest cell")
    for y, (w, f) in enumerate(zip(d.width_mean_sd, d.feasible)):
        ax.text(0.22, y, f"width {w:.2f} SD · RC-PPI++ feasible {f:.0%}", va="center", fontsize=6, color="white")
    ax.axvline(.95, color=MUTED, lw=.8, ls="--"); ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("ITT coverage (mean over cells)"); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22), fontsize=6)
    ax.set_title("(a) audit designs, budget 5%", loc="left")
    v = sec[sec.grid == "verification_error"].copy()
    v[["p0", "p1"]] = v.level.str.extract(r"phi0=([\d.]+),phi1=([\d.]+)").astype(float)
    ax = axes[1]
    for p1, col in zip(sorted(v.p1.unique()), plt.cm.Blues(np.linspace(.35, 1, 5))):
        s = v[v.p1 == p1].sort_values("p0")
        ax.plot(s.p0, s.srmse_itt, "o-", ms=3, color=col, label=f"φ1={p1:g}")
    s = v[v.p1 == 0].sort_values("p0")
    ax.plot(s.p0, s.srmse_deletion, "s--", ms=3, color=RULES["deletion"][1], label="detect-and-delete")
    ax.set_xlabel("φ0 (human recorded as AI)"); ax.set_ylabel("standardised RMSE")
    ax.set_title("(b) verification error", loc="left"); ax.legend(frameon=False, fontsize=6)
    fig.savefig(out / "fig_secondary.pdf"); fig.savefig(out / "fig_secondary.png")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    sec = pd.read_csv(sys.argv[2])
    cells = load(sys.argv[3:])
    regime_map(cells, out); corpus_ratios(cells, out); secondary(sec, out)
    print("cells", len(cells), "pools", cells.pool.nunique())
