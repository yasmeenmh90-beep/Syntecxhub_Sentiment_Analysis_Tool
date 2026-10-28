"""Evaluation charts saved as PNG (matplotlib, no display needed)."""

import matplotlib.pyplot as plt  # charts are saved and closed, never shown, so any backend works
import numpy as np

# Palette validated for colour-blind separation and contrast on white
POSITIVE = "#7c3aed"   # violet
NEGATIVE = "#db2777"   # pink
MUTED_BAR = "#d9d3ee"
INK = "#1d1838"
INK_2 = "#5b5677"
GRID = "#ebe8f4"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.title.set_color(INK)


def model_comparison(results: list, best_name: str, path, folds: int = 5):
    """Horizontal bars of cross-validated macro-F1, best model highlighted."""
    rows = sorted(results, key=lambda r: r["cv_f1_mean"])
    names = [r["config"].name for r in rows]
    vals = [r["cv_f1_mean"] for r in rows]
    errs = [r["cv_f1_std"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.48 * len(rows) + 1.4), dpi=150)
    colors = [POSITIVE if n == best_name else MUTED_BAR for n in names]
    ax.barh(names, vals, xerr=errs, color=colors, height=0.62,
            error_kw=dict(ecolor=INK_2, elinewidth=1, capsize=2))
    lo = max(0.0, min(vals) - 0.08)
    ax.set_xlim(lo, min(1.0, max(vals) + 0.06))
    for i, v in enumerate(vals):
        ax.text(v + 0.004 + errs[i], i, f"{v:.3f}", va="center", fontsize=9,
                color=INK if names[i] == best_name else INK_2,
                fontweight="bold" if names[i] == best_name else "normal")
    ax.set_xlabel(f"Macro F1 ({folds}-fold cross-validation, mean ± std)", color=INK_2, fontsize=9)
    ax.set_title("Model comparison", loc="left", fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def confusion(cm, labels, path, title="Confusion matrix (test set)"):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=150)
    ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted", color=INK_2)
    ax.set_ylabel("Actual", color=INK_2)
    totals = cm.sum(axis=1, keepdims=True)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            share = cm[i, j] / totals[i, 0] if totals[i, 0] else 0
            ax.text(j, i, f"{cm[i, j]}\n{share:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2)
    ax.title.set_color(INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def top_words(features: dict, path, n: int = 12):
    """Two panels: words that most push towards each class."""
    if len(features) != 2:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.36 * n + 1.4), dpi=150)
    for ax, (label, items) in zip(axes, features.items()):
        items = items[:n][::-1]
        color = POSITIVE if label.startswith("pos") else NEGATIVE
        ax.barh([w for w, _ in items], [s for _, s in items], color=color, height=0.62)
        ax.set_title(f"Strongest {label} words", loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Model weight", color=INK_2, fontsize=9)
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True
