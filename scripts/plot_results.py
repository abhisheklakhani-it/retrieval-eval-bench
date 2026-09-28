"""Draw the README charts from results/ (light and dark versions).

  python scripts/plot_results.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reteval.stats import bootstrap_ci  # noqa: E402

RESULTS, DOCS = ROOT / "results", ROOT / "docs"

# Colour-blind-safe categorical palette (validated), one colour + one marker shape per family.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "lexical": "#2a78d6", "dense": "#eb6834", "hybrid": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#383835",
             "lexical": "#3987e5", "dense": "#d95926", "hybrid": "#199e70"},
}
FAMILY_MARKER = {"lexical": "o", "dense": "s", "hybrid": "D"}
FAMILY_LABEL = {"lexical": "Lexical (word matching)", "dense": "Dense (embeddings)", "hybrid": "Hybrid (both)"}


def family(name: str) -> str:
    return "hybrid" if name.startswith("Hybrid") else "dense" if name.startswith("Dense") else "lexical"


def style_axes(ax, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)
    ax.grid(color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def ndcg_dot_plot(theme: str) -> None:
    t = THEMES[theme]
    per_query = pd.read_csv(RESULTS / "per_query_ndcg.csv")
    systems = [c for c in per_query.columns if c != "query_id"]
    rows = []
    for name in systems:
        scores = per_query[name].tolist()
        low, high = bootstrap_ci(scores, [0.0] * len(scores))  # CI of the mean
        rows.append((name, sum(scores) / len(scores), low, high))
    rows.sort(key=lambda r: r[1])  # best at the top

    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200, facecolor=t["surface"])
    style_axes(ax, t)
    ax.grid(axis="y", visible=False)
    for y, (name, mean, low, high) in enumerate(rows):
        f = family(name)
        ax.plot([low, high], [y, y], color=t[f], linewidth=2, solid_capstyle="round", alpha=0.55)
        ax.plot(mean, y, marker=FAMILY_MARKER[f], markersize=9, color=t[f],
                markeredgecolor=t["surface"], markeredgewidth=2)
        ax.text(high + 0.004, y, f"{mean:.3f}", va="center", fontsize=10, color=t["text"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=t["text"], fontsize=10)
    ax.set_xlim(0.58, 0.82)
    ax.set_xlabel("nDCG@10 (higher is better) · lines show the 95% bootstrap confidence interval",
                  color=t["muted"], fontsize=9.5)
    ax.set_title("Retrieval quality on SciFact (300 test queries)",
                 loc="left", color=t["text"], fontsize=12, fontweight="bold", pad=28)
    handles = [plt.Line2D([], [], marker=FAMILY_MARKER[f], linestyle="", markersize=8, color=t[f], label=FAMILY_LABEL[f])
               for f in ["lexical", "dense", "hybrid"]]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
              fontsize=9.5, labelcolor=t["text"], handletextpad=0.3, columnspacing=1.4, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(DOCS / f"ndcg_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def alpha_curve(theme: str) -> None:
    t = THEMES[theme]
    tuning = pd.read_csv(RESULTS / "tuning.csv")
    tuning = tuning[tuning["step"] == "hybrid alpha"]
    alphas = tuning["setting"].str.replace("alpha=", "").astype(float).tolist()
    ndcg = tuning["train_ndcg@10"].tolist()
    best = max(range(len(ndcg)), key=ndcg.__getitem__)

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=200, facecolor=t["surface"])
    style_axes(ax, t)
    ax.grid(axis="x", visible=False)
    ax.plot(alphas, ndcg, color=t["hybrid"], linewidth=2, marker="o", markersize=6,
            markeredgecolor=t["surface"], markeredgewidth=1.5)
    ax.plot(alphas[best], ndcg[best], marker="o", markersize=10, color=t["hybrid"],
            markeredgecolor=t["surface"], markeredgewidth=2)
    ax.annotate(f"best: α = {alphas[best]}  ({ndcg[best]:.3f})", (alphas[best], ndcg[best]),
                xytext=(0, 12), textcoords="offset points", ha="center", color=t["text"], fontsize=10)
    ax.annotate(f"BM25 only\n{ndcg[0]:.3f}", (alphas[0], ndcg[0]), xytext=(8, -4), textcoords="offset points",
                va="top", color=t["text"], fontsize=9.5)
    ax.annotate(f"dense only\n{ndcg[-1]:.3f}", (alphas[-1], ndcg[-1]), xytext=(-8, -8), textcoords="offset points",
                ha="right", va="top", color=t["text"], fontsize=9.5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(min(ndcg) - 0.02, max(ndcg) + 0.02)
    ax.set_xlabel("α  (weight of the dense score; 1 − α goes to BM25)", color=t["muted"], fontsize=9.5)
    ax.set_ylabel("nDCG@10 on train", color=t["muted"], fontsize=9.5)
    ax.set_title("Mixing BM25 and BGE beats either one alone (tuned on the 809 train queries)",
                 loc="left", color=t["text"], fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(DOCS / f"alpha_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    for theme in THEMES:
        ndcg_dot_plot(theme)
        alpha_curve(theme)
    print("Saved charts to", DOCS.relative_to(ROOT))


if __name__ == "__main__":
    main()
