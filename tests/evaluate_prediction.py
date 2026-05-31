#!/usr/bin/env python3
"""
evaluate_prediction.py
──────────────────────
Runs the full backtesting pipeline and prints a detailed summary.

Usage
─────
    # Run with all horizons (default primary=3d)
    python tests/evaluate_prediction.py

    # Choose a different primary horizon for the existing charts
    python tests/evaluate_prediction.py --horizon 5

    # Just check price-data availability without evaluating
    python tests/evaluate_prediction.py --dry-run

    # Use a custom dataset file
    python tests/evaluate_prediction.py --dataset path/to/dataset.json
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tests.utils import (
    FUTURE_DAYS, bold, err, format_ci, get_logger, header, info,
    ok, print_metric, print_section, results_path, warn, accent,
)
from tests.backtest_engine import (
    load_dataset, run_backtest, compute_metrics, export_results,
    print_summary, get_future_returns,
)

log = get_logger("evaluate_prediction")


# ──────────────────────────────────────────────────────────────
# Dry-run
# ──────────────────────────────────────────────────────────────

def dry_run(dataset: list[dict], primary_horizon: int) -> None:
    print_section("DRY RUN – Price Data Availability Check")
    ret_col = f"future_return_{primary_horizon}d"
    rows    = []

    symbols = sorted({r["symbol"] for r in dataset})
    print(info(f"  Symbols : {', '.join(symbols)}"))
    print(info(f"  Records : {len(dataset)}"))
    print()

    for item in dataset:
        sym   = item["symbol"]
        date  = item["date"]
        title = item["title"][:55]
        prices = get_future_returns(sym, date)
        ret   = prices.get(ret_col)

        status = ok("✓") if ret is not None else err("✗")
        ret_s  = (ok(f"{ret:+.2f}%") if ret is not None and ret >= 0
                  else (err(f"{ret:+.2f}%") if ret is not None else warn("N/A")))
        print(f"  {status} [{sym}] {date}  {ret_s:>14}  {title}")
        rows.append({"symbol": sym, "date": date, "title": item["title"],
                     ret_col: ret, "has_price": ret is not None})

    df = pd.DataFrame(rows)
    available = int(df["has_price"].sum())
    total     = len(df)
    print()
    pct = available / total * 100 if total > 0 else 0
    print(bold(f"\n  Price coverage: {available}/{total} = {pct:.0f}%"))

    if available == 0:
        print(err("\n  ⚠  No price data – check yfinance connectivity."))
    elif available < total * 0.7:
        print(warn("\n  ⚠  < 70% coverage – results may be unreliable."))
    else:
        print(ok("\n  ✓  Sufficient price data for backtesting."))


# ──────────────────────────────────────────────────────────────
# Per-symbol breakdown
# ──────────────────────────────────────────────────────────────

def print_per_symbol(df: pd.DataFrame, metrics: dict, primary_horizon: int) -> None:
    ret_col   = f"future_return_{primary_horizon}d"
    per_stock = metrics.get("per_stock", {})
    print_section("Per-Symbol Breakdown")

    header_line = (
        f"  {'Symbol':<14} {'n':>5}  "
        + "  ".join(f"{h}d".center(16) for h in FUTURE_DAYS)
    )
    print(info(header_line))
    sub_line = (
        f"  {'':14} {'':>5}  "
        + "  ".join("Acc  [95% CI]    ".center(16) for _ in FUTURE_DAYS)
    )
    print(info(sub_line))
    print(info("  " + "─" * (20 + 18 * len(FUTURE_DAYS))))

    for sym, sym_stats in sorted(per_stock.items()):
        n       = sym_stats.get("count", 0)
        horizons = sym_stats.get("horizons", {})
        cols    = []
        for h in FUTURE_DAYS:
            h_data = horizons.get(f"{h}d", {})
            if not h_data:
                cols.append(warn("  —  "))
                continue
            acc    = h_data.get("accuracy", 0)
            ci_lo  = h_data.get("ci_lower", 0)
            ci_hi  = h_data.get("ci_upper", 0)
            acc_s  = ok(f"{acc*100:.0f}%") if acc >= 0.55 else warn(f"{acc*100:.0f}%")
            ci_s   = f"[{ci_lo*100:.0f}%,{ci_hi*100:.0f}%]"
            cols.append(f"{acc_s} {ci_s}")
        sym_s = accent(f"{sym:<14}")
        print(f"  {sym_s} {n:>5}  " + "  ".join(f"{c:<16}" for c in cols))


# ──────────────────────────────────────────────────────────────
# Per-stock CSV summary (console)
# ──────────────────────────────────────────────────────────────

def print_stock_metrics_table(metrics: dict) -> None:
    per_stock = metrics.get("per_stock", {})
    if not per_stock:
        return

    print_section("Per-Stock Summary Table (primary horizon for each)")

    # Find best horizon per stock by accuracy
    print(info(f"  {'Symbol':<14} {'n':>5}  "
               f"{'Best Horizon':>14}  {'Accuracy':>10}  {'95% CI':<22}  "
               f"{'WinRate':>9}  {'MeanRet':>9}"))
    print(info("  " + "─" * 85))

    for sym, sym_stats in sorted(per_stock.items()):
        n        = sym_stats.get("count", 0)
        horizons = sym_stats.get("horizons", {})
        if not horizons:
            print(f"  {accent(sym):<22} {n:>5}  {warn('no data'):>14}")
            continue

        best_h    = max(horizons, key=lambda h: horizons[h].get("accuracy", 0))
        best_data = horizons[best_h]
        acc       = best_data.get("accuracy", 0)
        ci_lo     = best_data.get("ci_lower", 0)
        ci_hi     = best_data.get("ci_upper", 0)
        wr        = best_data.get("win_rate", 0)
        mr        = best_data.get("mean_return", 0)

        acc_s = ok(f"{acc*100:.1f}%") if acc >= 0.55 else warn(f"{acc*100:.1f}%")
        ci_s  = f"[{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]"
        mr_s  = ok(f"{mr:+.2f}%") if mr >= 0 else err(f"{mr:+.2f}%")

        print(f"  {accent(sym):<22} {n:>5}  {best_h:>14}  "
              f"{acc_s:>20}  {ci_s:<22}  "
              f"{wr*100:>8.1f}%  {mr_s:>14}")


# ──────────────────────────────────────────────────────────────
# Sample prediction table
# ──────────────────────────────────────────────────────────────

def print_prediction_table(df: pd.DataFrame, n: int = 15,
                            primary_horizon: int = 3) -> None:
    ret_col = f"future_return_{primary_horizon}d"
    print_section(f"Sample Predictions (last {n} records, {primary_horizon}d horizon)")

    hdr = (f"  {'Symbol':<12} {'Date':<12} {'AI Sentiment':<22} "
           f"{'Pred Dir':<16} {'Actual Dir':<16} {'Return':>8}  {'OK?':<5}")
    print(info(hdr))
    print(info("  " + "─" * 100))

    for _, row in df.tail(n).iterrows():
        sym  = row["symbol"]
        date = row["date"]
        sent = row.get("ai_sentiment", "?")
        pred = row.get("predicted_direction", "?")
        act  = row.get("actual_direction", "—")
        ret  = row.get(ret_col)
        corr = row.get("is_correct")

        ret_s  = (ok(f"{ret:+.2f}%") if ret is not None and ret >= 0
                  else (err(f"{ret:+.2f}%") if ret is not None else warn("N/A")))
        corr_s = (ok("✓") if corr is True else (err("✗") if corr is False else info("?")))

        print(f"  {sym:<12} {date:<12} {sent:<22} "
              f"{pred:<16} {str(act):<16} {ret_s:>14}  {corr_s}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Financial Sentiment Backtest – Evaluation Runner"
    )
    p.add_argument(
        "--horizon", type=int, default=3, choices=[1, 3, 5, 10, 20, 60],
        help="Primary horizon for legacy charts (default: 3)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Check price data availability without running evaluation",
    )
    p.add_argument(
        "--dataset", type=str, default=None,
        help="Path to a custom sentiment_dataset.json",
    )
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    horizon = args.horizon

    print()
    print(header("╔══════════════════════════════════════════════════════════╗"))
    print(header("║      Financial Sentiment Backtesting System              ║"))
    print(header("║      台股新聞情緒 × 股價預測回測引擎                    ║"))
    print(header("╚══════════════════════════════════════════════════════════╝"))
    print()

    dataset = load_dataset(args.dataset) if args.dataset else load_dataset()

    print_metric("Records in dataset", len(dataset))
    print_metric("All horizons",       str(FUTURE_DAYS))
    print_metric("Primary horizon",    f"{horizon}d")
    print()

    if args.dry_run:
        dry_run(dataset, horizon)
        return

    df = run_backtest(dataset, primary_horizon=horizon)

    if df.empty:
        print(err("No records were processed."))
        sys.exit(1)

    metrics = compute_metrics(df, primary_horizon=horizon)
    export_results(df, metrics)

    print_summary(metrics)
    print_per_symbol(df, metrics, primary_horizon=horizon)
    print_stock_metrics_table(metrics)
    print_prediction_table(df, n=15, primary_horizon=horizon)

    print()
    print(ok("  ✓ Evaluation complete."))
    print(info(f"  Prediction CSV    → {results_path('prediction_results.csv')}"))
    print(info(f"  Metrics JSON      → {results_path('prediction_metrics.json')}"))
    print(info(f"  Stock metrics CSV → {results_path('stock_metrics.csv')}"))
    print(info(f"  Classification    → {results_path('classification_report.txt')}"))
    print()
    print(info("  Next:  python tests/visualize_prediction.py"))
    print()


if __name__ == "__main__":
    main()