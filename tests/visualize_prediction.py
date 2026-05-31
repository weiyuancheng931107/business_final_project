#!/usr/bin/env python3
"""
visualize_prediction.py
───────────────────────
Generates ALL analysis charts from prediction_results.csv.

Charts
──────
Existing (unchanged):
  1. confusion_matrix.png
  2. sentiment_distribution.png
  3. sentiment_vs_return.png
  4. cumulative_return.png
  5. accuracy_bar.png            (per-sentiment accuracy)
  6. return_distribution.png

New (multi-horizon analysis):
  7. accuracy_by_horizon.png     Accuracy across 1/3/5/10/20/60d
  8. f1_by_horizon.png           Weighted F1 score across horizons
  9. winrate_by_horizon.png      Win-rate across horizons

Usage
─────
    python tests/visualize_prediction.py
    python tests/visualize_prediction.py --horizon 5   # primary for existing charts
    python tests/visualize_prediction.py --csv path/to/results.csv
    python tests/visualize_prediction.py --metrics path/to/metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

from tests.utils import (
    FUTURE_DAYS, get_logger, info, ok, err, warn, results_path,
)

log = get_logger("visualize_prediction")

# ──────────────────────────────────────────────────────────────
# Design tokens  (Neumorphism palette)
# ──────────────────────────────────────────────────────────────
BG     = "#E7E5E4"
FG     = "#1E2938"
TEAL   = "#006666"
GREEN  = "#00A63D"
RED    = "#FF2157"
ORANGE = "#FE9900"
BLUE   = "#2563EB"
GREY   = "#64748B"
BORDER = "#c4c2c1"

SENTIMENT_COLOURS = {
    "positive":          GREEN,
    "slightly_positive": "#8BC34A",
    "neutral":           GREY,
    "slightly_negative": ORANGE,
    "negative":          RED,
}
ALL_SENTIMENTS = ["positive", "slightly_positive", "neutral", "slightly_negative", "negative"]
ALL_DIRECTIONS = ["bullish", "weak_bullish", "neutral", "weak_bearish", "bearish"]
DIRECTION_COLOURS = {
    "bullish":      GREEN,
    "weak_bullish": "#8BC34A",
    "neutral":      GREY,
    "weak_bearish": ORANGE,
    "bearish":      RED,
}

# Horizon colour ramp (cool → warm as horizon lengthens)
_HORIZON_COLOURS = {
    "1d":  "#2563EB",
    "3d":  "#006666",
    "5d":  "#00A63D",
    "10d": "#8BC34A",
    "20d": "#FE9900",
    "60d": "#FF2157",
}

plt.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     BG,
    "axes.edgecolor":     BORDER,
    "axes.labelcolor":    FG,
    "xtick.color":        FG,
    "ytick.color":        FG,
    "text.color":         FG,
    "grid.color":         BORDER,
    "grid.linestyle":     "--",
    "grid.alpha":         0.5,
    "font.family":        "sans-serif",
    "axes.unicode_minus": False,
})


def _save(fig: plt.Figure, name: str) -> str:
    path = results_path(name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Saved → %s", path)
    print(ok(f"  ✓  {name}"))
    return path


def _styled_ax(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", color=FG, pad=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BORDER)
    ax.tick_params(colors=FG, labelsize=9)


def _legend(ax: plt.Axes) -> None:
    leg = ax.legend(facecolor=BG, edgecolor=BORDER, fontsize=8)
    for t in leg.get_texts():
        t.set_color(FG)


# ══════════════════════════════════════════════════════════════
# EXISTING CHARTS  (unchanged behaviour)
# ══════════════════════════════════════════════════════════════

# 1 ─ Confusion Matrix
def plot_confusion_matrix(df: pd.DataFrame, ret_col: str) -> None:
    eval_df = df.dropna(subset=["ai_sentiment", ret_col]).copy()
    if eval_df.empty:
        log.warning("Skipping confusion matrix – no evaluable rows.")
        return

    from tests.utils import SENTIMENT_TO_DIRECTION, return_to_direction
    eval_df["pred"] = eval_df["ai_sentiment"].map(
        lambda s: SENTIMENT_TO_DIRECTION.get(s, "neutral")
    )
    eval_df["true"] = eval_df[ret_col].apply(return_to_direction)

    labels    = ALL_DIRECTIONS
    label_idx = {l: i for i, l in enumerate(labels)}
    cm        = np.zeros((len(labels), len(labels)), dtype=int)
    for _, row in eval_df.iterrows():
        t, p = row["true"], row["pred"]
        if t in label_idx and p in label_idx:
            cm[label_idx[t], label_idx[p]] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.where(cm.sum(axis=1, keepdims=True) > 0,
                           cm / cm.sum(axis=1, keepdims=True), 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    for idx, (data, title) in enumerate([
        (cm,      "Confusion Matrix (counts)"),
        (cm_norm, "Confusion Matrix (normalised)"),
    ]):
        ax = axes[idx]
        im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=max(data.max(), 1))
        for i in range(len(labels)):
            for j in range(len(labels)):
                v     = data[i, j]
                text  = f"{v:.0%}" if idx == 1 else str(int(v))
                color = "white" if v > (data.max() or 1) * 0.6 else FG
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        _styled_ax(ax, title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Directional Prediction – Confusion Matrix",
                 fontsize=14, fontweight="bold", color=FG, y=1.02)
    plt.tight_layout()
    _save(fig, "confusion_matrix.png")


# 2 ─ Sentiment Distribution
def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    counts = df["ai_sentiment"].value_counts().reindex(ALL_SENTIMENTS, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    bars = ax.bar(
        counts.index, counts.values,
        color=[SENTIMENT_COLOURS.get(s, GREY) for s in counts.index],
        edgecolor=BORDER, linewidth=0.8, width=0.6,
    )
    for bar, val in zip(bars, counts.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(int(val)), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=FG)
    ax.set_xlabel("AI Sentiment Label")
    ax.set_ylabel("Number of Articles")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.4)
    _styled_ax(ax, "AI Sentiment Distribution")
    plt.tight_layout()
    _save(fig, "sentiment_distribution.png")


# 3 ─ Sentiment vs Average Return
def plot_sentiment_vs_return(df: pd.DataFrame, ret_col: str) -> None:
    if ret_col not in df.columns:
        log.warning("Column %s missing; skipping sentiment_vs_return.", ret_col)
        return
    stats = (df.groupby("ai_sentiment")[ret_col]
               .agg(["mean", "median", "std", "count"])
               .reindex(ALL_SENTIMENTS).dropna(how="all"))
    if stats.empty:
        return

    x = np.arange(len(stats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    bar_mean = ax.bar(x - w / 2, stats["mean"], width=w,
                      color=[SENTIMENT_COLOURS.get(s, GREY) for s in stats.index],
                      edgecolor=BORDER, linewidth=0.8, label="Mean")
    bar_med  = ax.bar(x + w / 2, stats["median"], width=w,
                      color=[SENTIMENT_COLOURS.get(s, GREY) for s in stats.index],
                      alpha=0.55, edgecolor=BORDER, linewidth=0.8,
                      label="Median", hatch="///")
    ax.axhline(0, color=FG, linewidth=0.8)
    for bar in list(bar_mean) + list(bar_med):
        h = bar.get_height()
        if h == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, h + (0.05 if h >= 0 else -0.25),
                f"{h:+.1f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold",
                color=GREEN if h >= 0 else RED)
    ax.set_xticks(x)
    ax.set_xticklabels(stats.index, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(f"Average Return ({ret_col.split('_')[-1]})")
    ax.set_xlabel("AI Sentiment Label")
    ax.grid(axis="y", alpha=0.4)
    _legend(ax)
    _styled_ax(ax, f"Sentiment vs Average Future Return ({ret_col})")
    plt.tight_layout()
    _save(fig, "sentiment_vs_return.png")


# 4 ─ Cumulative Return
def plot_cumulative_return(df: pd.DataFrame, ret_col: str) -> None:
    if ret_col not in df.columns:
        return
    df_s = df.dropna(subset=[ret_col, "ai_sentiment"]).sort_values("date").reset_index(drop=True)
    if df_s.empty:
        return

    from tests.utils import SENTIMENT_TO_DIRECTION, predicted_is_bullish
    df_s["signal"] = df_s["ai_sentiment"].map(
        lambda s: predicted_is_bullish(SENTIMENT_TO_DIRECTION.get(s, "neutral"))
    )
    df_s["trade_ret"] = df_s.apply(
        lambda r: r[ret_col] if r["signal"] is True
        else (-r[ret_col] if r["signal"] is False else 0.0),
        axis=1,
    )
    cum_strategy = (1 + df_s["trade_ret"] / 100).cumprod() - 1
    cum_bh       = (1 + df_s[ret_col] / 100).cumprod() - 1

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    ax.plot(df_s.index, cum_strategy * 100, color=TEAL,   linewidth=2.0,
            label="Sentiment Strategy (long/short)")
    ax.plot(df_s.index, cum_bh * 100,       color=ORANGE, linewidth=1.5,
            linestyle="--", label="Buy-and-Hold (equal weight)")
    ax.axhline(0, color=BORDER, linewidth=0.8)
    ax.fill_between(df_s.index, 0, cum_strategy * 100,
                    where=cum_strategy >= 0, alpha=0.15, color=GREEN)
    ax.fill_between(df_s.index, 0, cum_strategy * 100,
                    where=cum_strategy < 0,  alpha=0.15, color=RED)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
    ax.set_xlabel("Trade Index (sorted by date)")
    ax.set_ylabel("Cumulative Return (%)")
    ax.grid(alpha=0.4)
    _legend(ax)
    _styled_ax(ax, f"Cumulative Return – Sentiment Strategy vs Buy-and-Hold ({ret_col})")
    plt.tight_layout()
    _save(fig, "cumulative_return.png")


# 5 ─ Accuracy Bar (per-sentiment)
def plot_accuracy_bar(df: pd.DataFrame, ret_col: str) -> None:
    if ret_col not in df.columns:
        return
    from tests.utils import SENTIMENT_TO_DIRECTION, return_to_direction, binary_correct
    ev = df.dropna(subset=["ai_sentiment", ret_col]).copy()
    if ev.empty:
        return
    ev["pred_dir"] = ev["ai_sentiment"].map(
        lambda s: SENTIMENT_TO_DIRECTION.get(s, "neutral")
    )
    ev["act_dir"]  = ev[ret_col].apply(return_to_direction)
    ev["correct"]  = ev.apply(
        lambda r: binary_correct(r["pred_dir"], r["act_dir"]), axis=1
    )
    acc_by_sent = (ev.groupby("ai_sentiment")["correct"]
                     .agg(["mean", "count"])
                     .reindex(ALL_SENTIMENTS).dropna(how="all"))

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    x  = np.arange(len(acc_by_sent))
    cs = [GREEN if v >= 0.55 else (ORANGE if v >= 0.45 else RED)
          for v in acc_by_sent["mean"].astype(float).fillna(0.0)]
    bars = ax.bar(x, acc_by_sent["mean"] * 100,
                  color=cs, edgecolor=BORDER, linewidth=0.8, width=0.55)
    ax.axhline(50, color=GREY, linewidth=1.2, linestyle="--",
               label="Random baseline (50%)")
    for bar, (_, row) in zip(bars, acc_by_sent.iterrows()):
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.0f}%\n(n={int(row['count'])})",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=FG)
    ax.set_xticks(x)
    ax.set_xticklabels(acc_by_sent.index, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Binary Directional Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", alpha=0.4)
    _legend(ax)
    _styled_ax(ax, "Prediction Accuracy by AI Sentiment Label")
    plt.tight_layout()
    _save(fig, "accuracy_bar.png")


# 6 ─ Return Distribution
def plot_return_distribution(df: pd.DataFrame, ret_col: str) -> None:
    if ret_col not in df.columns:
        return
    data = df[ret_col].dropna()
    if data.empty:
        return
    sentiments = [s for s in ALL_SENTIMENTS if s in df["ai_sentiment"].values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)

    ax = axes[0]
    n_bins = min(30, max(10, len(data) // 3))
    ax.hist(data, bins=n_bins, color=TEAL, edgecolor=BORDER, alpha=0.85)
    ax.axvline(data.mean(),   color=RED,    linewidth=1.8, linestyle="--",
               label=f"Mean {data.mean():+.2f}%")
    ax.axvline(data.median(), color=ORANGE, linewidth=1.8, linestyle=":",
               label=f"Median {data.median():+.2f}%")
    ax.axvline(0, color=FG, linewidth=0.8)
    ax.set_xlabel(f"Future Return ({ret_col})")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.4)
    _legend(ax)
    _styled_ax(ax, f"Overall Return Distribution ({ret_col})")

    ax = axes[1]
    for sent in sentiments:
        sub    = df[df["ai_sentiment"] == sent][ret_col].dropna()
        colour = SENTIMENT_COLOURS.get(sent, GREY)
        if len(sub) < 2:
            continue
        if _HAS_SEABORN:
            import seaborn as sns
            sns.kdeplot(sub, ax=ax, color=colour, linewidth=2.0,
                        fill=True, alpha=0.25, label=f"{sent} (n={len(sub)})")
        else:
            ax.hist(sub, bins=10, color=colour, alpha=0.4,
                    edgecolor=BORDER, linewidth=0.6, density=True,
                    label=f"{sent} (n={len(sub)})")
    ax.axvline(0, color=FG, linewidth=0.8)
    ax.set_xlabel(f"Future Return ({ret_col})")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.4)
    _legend(ax)
    _styled_ax(ax, "Return Distribution by Sentiment")
    plt.tight_layout()
    _save(fig, "return_distribution.png")


# ══════════════════════════════════════════════════════════════
# NEW CHARTS  (multi-horizon analysis)
# ══════════════════════════════════════════════════════════════

def _extract_horizon_series(
    df: pd.DataFrame,
    metric_fn,   # callable(df, ret_col) → float
) -> tuple[list[str], list[float], list[int]]:
    """Helper: compute one metric per horizon."""
    labels, values, counts = [], [], []
    for h in FUTURE_DAYS:
        ret_col = f"future_return_{h}d"
        if ret_col not in df.columns:
            continue
        sub = df.dropna(subset=["ai_sentiment", ret_col])
        if sub.empty:
            continue
        val, n = metric_fn(sub, ret_col)
        labels.append(f"{h}d")
        values.append(val)
        counts.append(n)
    return labels, values, counts


# 7 ─ Accuracy by Horizon
def plot_accuracy_by_horizon(df: pd.DataFrame, metrics_json: Optional[dict] = None) -> None:
    """
    Bar chart: binary directional accuracy at each prediction horizon,
    with 95% Wilson confidence-interval error bars.
    """
    from tests.utils import (
        SENTIMENT_TO_DIRECTION, return_to_direction, binary_correct, wilson_ci,
    )

    h_labels, accs, ns, ci_lows, ci_highs = [], [], [], [], []
    for h in FUTURE_DAYS:
        ret_col = f"future_return_{h}d"
        if ret_col not in df.columns:
            continue
        sub = df.dropna(subset=["ai_sentiment", ret_col]).copy()
        if sub.empty:
            continue
        sub["pred_dir"] = sub["ai_sentiment"].map(
            lambda s: SENTIMENT_TO_DIRECTION.get(s, "neutral")
        )
        sub["act_dir"]  = sub[ret_col].apply(return_to_direction)
        sub["correct"]  = sub.apply(
            lambda r: binary_correct(r["pred_dir"], r["act_dir"]), axis=1
        )
        bin_df  = sub.dropna(subset=["correct"])
        n_bin   = len(bin_df)
        n_corr  = int(bin_df["correct"].sum())
        acc     = n_corr / n_bin if n_bin > 0 else 0.0
        lo, hi  = wilson_ci(n_corr, n_bin)

        h_labels.append(f"{h}d")
        accs.append(acc * 100)
        ns.append(n_bin)
        ci_lows.append((acc - lo) * 100)
        ci_highs.append((hi - acc) * 100)

    if not h_labels:
        log.warning("No horizon data for accuracy_by_horizon chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    x       = np.arange(len(h_labels))
    colours = [_HORIZON_COLOURS.get(l, TEAL) for l in h_labels]

    bars = ax.bar(x, accs, color=colours, edgecolor=BORDER, linewidth=0.8,
                  width=0.55, zorder=3)
    ax.errorbar(x, accs, yerr=[ci_lows, ci_highs],
                fmt="none", color=FG, capsize=5, linewidth=1.5, zorder=4)
    ax.axhline(50, color=GREY, linewidth=1.4, linestyle="--",
               label="Random baseline (50%)", zorder=2)

    for bar, acc, n in zip(bars, accs, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(ci_highs) + 0.5,
                f"{acc:.1f}%\n(n={n})",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=FG)

    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, fontsize=11)
    ax.set_xlabel("Prediction Horizon (trading days)")
    ax.set_ylabel("Binary Directional Accuracy (%)")
    ax.set_ylim(0, min(100, max(accs) + max(ci_highs) + 15))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", alpha=0.4, zorder=1)
    _legend(ax)
    _styled_ax(ax, "GPT Sentiment Prediction Accuracy by Horizon (95% CI)")
    plt.tight_layout()
    _save(fig, "accuracy_by_horizon.png")


# 8 ─ F1 by Horizon
def plot_f1_by_horizon(df: pd.DataFrame) -> None:
    """
    Line chart: weighted F1 score (5-class direction) at each horizon.
    """
    try:
        from sklearn.metrics import f1_score
        _has_sk = True
    except ImportError:
        log.warning("scikit-learn not available; skipping f1_by_horizon.")
        return

    from tests.utils import SENTIMENT_TO_DIRECTION, return_to_direction

    h_labels, f1s, ns = [], [], []
    for h in FUTURE_DAYS:
        ret_col = f"future_return_{h}d"
        if ret_col not in df.columns:
            continue
        sub = df.dropna(subset=["ai_sentiment", ret_col]).copy()
        if len(sub) < 5:
            continue
        sub["pred_dir"] = sub["ai_sentiment"].map(
            lambda s: SENTIMENT_TO_DIRECTION.get(s, "neutral")
        )
        sub["act_dir"]  = sub[ret_col].apply(return_to_direction)
        f1_val = f1_score(
            sub["act_dir"].tolist(),
            sub["pred_dir"].tolist(),
            labels=ALL_DIRECTIONS, average="weighted", zero_division=0,
        )
        h_labels.append(f"{h}d")
        f1s.append(float(f1_val))
        ns.append(len(sub))

    if not h_labels:
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    x       = np.arange(len(h_labels))
    colours = [_HORIZON_COLOURS.get(l, TEAL) for l in h_labels]

    ax.plot(x, f1s, color=TEAL, linewidth=2.5, marker="o",
            markersize=9, zorder=3, label="Weighted F1")
    for i, (xi, yi, n) in enumerate(zip(x, f1s, ns)):
        colour = colours[i]
        ax.scatter(xi, yi, color=colour, s=80, zorder=4)
        ax.annotate(
            f"{yi:.3f}\n(n={n})",
            (xi, yi), textcoords="offset points",
            xytext=(0, 12), ha="center", fontsize=9, color=FG,
        )

    ax.axhline(0.2, color=GREY, linewidth=1.0, linestyle="--",
               label="Baseline reference")
    ax.set_xticks(x)
    ax.set_xticklabels(h_labels, fontsize=11)
    ax.set_xlabel("Prediction Horizon (trading days)")
    ax.set_ylabel("Weighted F1 Score (5-class direction)")
    ax.set_ylim(0, min(1.05, max(f1s) + 0.2))
    ax.grid(axis="y", alpha=0.4)
    _legend(ax)
    _styled_ax(ax, "Weighted F1 Score by Prediction Horizon")
    plt.tight_layout()
    _save(fig, "f1_by_horizon.png")


# 9 ─ Win-Rate by Horizon
def plot_winrate_by_horizon(df: pd.DataFrame) -> None:
    """
    Win-rate (% of articles where future return > 0) per sentiment per horizon.
    Shows whether positive-sentiment articles more often precede price rises
    at longer horizons.
    """
    sentiments_present = [s for s in ALL_SENTIMENTS if s in df["ai_sentiment"].values]
    horizons_present   = [h for h in FUTURE_DAYS if f"future_return_{h}d" in df.columns]
    if not sentiments_present or not horizons_present:
        log.warning("Skipping winrate_by_horizon – insufficient data.")
        return

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
    x       = np.arange(len(horizons_present))
    x_labels = [f"{h}d" for h in horizons_present]

    line_colours = {
        "positive":          GREEN,
        "slightly_positive": "#8BC34A",
        "neutral":           GREY,
        "slightly_negative": ORANGE,
        "negative":          RED,
    }

    for sent in sentiments_present:
        sub_sent = df[df["ai_sentiment"] == sent]
        win_rates = []
        for h in horizons_present:
            ret_col = f"future_return_{h}d"
            sub_h   = sub_sent[ret_col].dropna()
            wr = float((sub_h > 0).mean()) * 100 if len(sub_h) > 0 else 0.0
            win_rates.append(wr)

        ax.plot(x, win_rates, color=line_colours.get(sent, GREY),
                linewidth=2.0, marker="o", markersize=7,
                label=sent, zorder=3)

    ax.axhline(50, color=GREY, linewidth=1.2, linestyle="--",
               label="Random baseline (50%)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Prediction Horizon (trading days)")
    ax.set_ylabel("Win Rate (% of articles with future return > 0)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", alpha=0.4)
    _legend(ax)
    _styled_ax(ax, "Win-Rate by Sentiment × Prediction Horizon")
    plt.tight_layout()
    _save(fig, "winrate_by_horizon.png")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate backtest visualisation charts")
    p.add_argument("--csv",     type=str, default=None)
    p.add_argument("--metrics", type=str, default=None,
                   help="Path to prediction_metrics.json (for CI bars)")
    p.add_argument("--horizon", type=int, default=3,
                   choices=[1, 3, 5, 10, 20, 60],
                   help="Primary horizon for existing charts (default: 3)")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    horizon  = args.horizon
    ret_col  = f"future_return_{horizon}d"
    csv_path = args.csv or results_path("prediction_results.csv")
    json_path = args.metrics or results_path("prediction_metrics.json")

    print()
    print(info("══════════════════════════════════════════════"))
    print(info("  Financial Sentiment Backtest – Visualiser"))
    print(info("══════════════════════════════════════════════"))
    print()

    if not os.path.exists(csv_path):
        print(err(f"  ✗  {csv_path} not found."))
        print(warn("  Run  python tests/evaluate_prediction.py  first."))
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(info(f"  Loaded {len(df)} rows from {csv_path}"))
    print(info(f"  Primary horizon: {horizon}d"))
    print()

    # Restore boolean columns
    for col in ["is_correct"]:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: True  if str(v).lower() in ("true",  "1") else
                         (False if str(v).lower() in ("false", "0") else np.nan)
            )

    # Load metrics JSON if available
    metrics_json: Optional[dict] = None
    if os.path.exists(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                metrics_json = json.load(f)
        except Exception:
            pass

    print(info("  Generating charts…"))

    # Existing charts
    plot_confusion_matrix(df, ret_col)
    plot_sentiment_distribution(df)
    plot_sentiment_vs_return(df, ret_col)
    plot_cumulative_return(df, ret_col)
    plot_accuracy_bar(df, ret_col)
    plot_return_distribution(df, ret_col)

    # New multi-horizon charts
    plot_accuracy_by_horizon(df, metrics_json)
    plot_f1_by_horizon(df)
    plot_winrate_by_horizon(df)

    print()
    print(ok(f"  ✓  All 9 charts saved to: {results_path('')}"))
    print()


if __name__ == "__main__":
    main()