"""
backtest_engine.py
──────────────────
Core Financial Sentiment Backtesting Engine.

Workflow
────────
1. Load sentiment_dataset.json  (built by build_dataset.py)
2. For every record, fetch price returns at horizons: 1, 3, 5, 10, 20, 60 days
3. Map sentiment → predicted_direction
4. Map return → actual_direction
5. Compute per-horizon and per-stock metrics with 95% Wilson CI
6. Export prediction_results.csv, prediction_metrics.json, stock_metrics.csv
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

import pandas as pd
import numpy as np

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from tqdm import tqdm as _tqdm
    def tqdm(iterable, **kw): return _tqdm(iterable, **kw)
except ImportError:
    def tqdm(iterable, **kw): return iterable

from tests.utils import (
    FUTURE_DAYS,
    SENTIMENT_TO_DIRECTION,
    binary_correct,
    err, get_logger, info, ok, print_section, print_metric, bold, warn,
    predicted_is_bullish,
    results_path,
    return_to_direction,
    sentiment_label_pretty,
    wilson_ci, format_ci,
)

log = get_logger("backtest_engine")

DATASET_PATH = os.path.join(_HERE, "sentiment_dataset.json")
FLASK_BASE   = "http://127.0.0.1:2330"

# All direction labels (ordered worst → best)
ALL_DIRECTIONS = ["bullish", "weak_bullish", "neutral", "weak_bearish", "bearish"]
ALL_SENTIMENTS = ["positive", "slightly_positive", "neutral", "slightly_negative", "negative"]


# ──────────────────────────────────────────────────────────────
# Step 1 – Load dataset
# ──────────────────────────────────────────────────────────────

def load_dataset(path: str = DATASET_PATH) -> list[dict]:
    log.info("Loading dataset from: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise RuntimeError(
            "sentiment_dataset.json is empty. Run: python tests/build_dataset.py"
        )
    log.info("  → %d records loaded.", len(data))
    return data


# ──────────────────────────────────────────────────────────────
# Flask / OpenAI (used only by build_dataset; kept here for compat)
# ──────────────────────────────────────────────────────────────

def _flask_available() -> bool:
    try:
        import requests
        r = requests.get(f"{FLASK_BASE}/api/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────
# Step 2 – Future price returns for ALL horizons
# ──────────────────────────────────────────────────────────────

_PRICE_CACHE: dict[str, pd.DataFrame] = {}


def _get_cached_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    key = f"{symbol}:{period}"
    if key not in _PRICE_CACHE:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist   = ticker.history(period=period)
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            hist.index = hist.index.normalize()
            _PRICE_CACHE[key] = hist
        except Exception as exc:
            log.warning("yfinance failed for %s: %s", symbol, exc)
            _PRICE_CACHE[key] = pd.DataFrame()
    return _PRICE_CACHE[key]


def get_future_returns(symbol: str, news_date_str: str) -> dict:
    """
    Compute percentage returns at every horizon in FUTURE_DAYS.

    Returns dict with keys: close_on_date, future_return_1d, …, future_return_60d.
    Any horizon where insufficient price data exists is returned as None.
    """
    results: dict[str, Optional[float]] = {
        "close_on_date": None,
        **{f"future_return_{h}d": None for h in FUTURE_DAYS},
    }

    hist = _get_cached_history(symbol)
    if hist.empty or "Close" not in hist.columns:
        return results

    news_date    = pd.Timestamp(news_date_str)
    trading_days = hist.index[hist.index >= news_date]

    if len(trading_days) == 0:
        log.debug("No trading days on/after %s for %s", news_date_str, symbol)
        return results

    t0 = trading_days[0]
    c0 = float(hist.loc[t0, "Close"])
    results["close_on_date"] = c0

    for h in FUTURE_DAYS:
        if len(trading_days) > h:
            t_h = trading_days[h]
            c_h = float(hist.loc[t_h, "Close"])
            results[f"future_return_{h}d"] = round((c_h - c0) / c0 * 100.0, 4)

    return results


# ──────────────────────────────────────────────────────────────
# Step 3 – Run the full backtest
# ──────────────────────────────────────────────────────────────

def run_backtest(
    dataset: Optional[list[dict]] = None,
    primary_horizon: int = 3,
    **_kwargs,
) -> pd.DataFrame:
    """
    Build a backtest DataFrame from the pre-analyzed dataset.

    Reads sentiment_type from each record (set by build_dataset.py).
    Fetches price data and computes returns at all FUTURE_DAYS horizons.
    No AI calls are made here.

    Parameters
    ----------
    dataset         : list of dataset records; loaded from file if None
    primary_horizon : horizon used for the legacy is_correct / actual_direction
                      columns (for backward compat with existing charts)

    Returns
    -------
    pd.DataFrame – one row per news record with all return columns
    """
    if dataset is None:
        dataset = load_dataset()

    rows = []
    print_section(f"Running backtest – {len(dataset)} records, "
                  f"horizons: {FUTURE_DAYS}")

    for item in tqdm(dataset, desc="Fetching prices", unit="item"):
        symbol   = item["symbol"]
        date_str = item["date"]
        title    = item["title"]

        ai_sentiment = (
            item.get("sentiment_type")
            or item.get("expected_sentiment")
        )
        if not ai_sentiment:
            log.warning(
                "Skipping [%s %s]: no sentiment_type. Run build_dataset.py first.",
                symbol, date_str,
            )
            continue

        price_data           = get_future_returns(symbol, date_str)
        predicted_direction  = SENTIMENT_TO_DIRECTION.get(ai_sentiment, "neutral")

        # Primary-horizon direction + binary correctness (backward compat)
        primary_ret = price_data.get(f"future_return_{primary_horizon}d")
        actual_direction = (
            return_to_direction(primary_ret) if primary_ret is not None else None
        )
        is_correct = (
            binary_correct(predicted_direction, actual_direction)
            if actual_direction is not None else None
        )

        row = {
            # Identification
            "symbol":              symbol,
            "name":                item.get("name", symbol),
            "date":                date_str,
            "title":               title,
            "source":              item.get("source", ""),
            "url":                 item.get("url", "#"),
            # Sentiment
            "ai_sentiment":        ai_sentiment,
            "sentiment_label":     item.get("sentiment_label", ""),
            "reason":              item.get("reason", ""),
            # Primary-horizon prediction (backward compat)
            "predicted_direction": predicted_direction,
            "actual_direction":    actual_direction,
            "is_correct":          is_correct,
            # Price baseline
            "close_on_date":       price_data.get("close_on_date"),
        }

        # All horizon returns
        for h in FUTURE_DAYS:
            row[f"future_return_{h}d"] = price_data.get(f"future_return_{h}d")

        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("Backtest complete. %d records processed.", len(df))
    return df


# ──────────────────────────────────────────────────────────────
# Step 4 – Compute metrics  (all horizons + per-stock)
# ──────────────────────────────────────────────────────────────

def _horizon_stats(df: pd.DataFrame, h: int) -> dict:
    """
    Compute full metric set for one prediction horizon.

    Parameters
    ----------
    df : backtest DataFrame
    h  : horizon in trading days (e.g. 3)

    Returns
    -------
    dict with accuracy, precision, recall, f1, win_rate, mean_return,
    sharpe_like, confusion_matrix, per_class
    """
    ret_col = f"future_return_{h}d"
    if ret_col not in df.columns:
        return {}

    sub = df.dropna(subset=["ai_sentiment", ret_col]).copy()
    if sub.empty:
        return {"n": 0}

    sub["pred_dir_h"]    = sub["ai_sentiment"].map(
        lambda s: SENTIMENT_TO_DIRECTION.get(s, "neutral")
    )
    sub["actual_dir_h"]  = sub[ret_col].apply(return_to_direction)
    sub["is_correct_h"]  = sub.apply(
        lambda r: binary_correct(r["pred_dir_h"], r["actual_dir_h"]),
        axis=1,
    )

    n         = len(sub)
    ret_series = sub[ret_col]
    y_pred    = sub["pred_dir_h"].tolist()
    y_true    = sub["actual_dir_h"].tolist()

    stats: dict = {"n": n}

    # Binary accuracy & win rate
    bin_df   = sub.dropna(subset=["is_correct_h"])
    n_corr   = int(bin_df["is_correct_h"].sum())
    n_bin    = len(bin_df)
    bin_acc  = n_corr / n_bin if n_bin > 0 else 0.0
    ci_lo, ci_hi = wilson_ci(n_corr, n_bin)

    stats["binary_accuracy"] = round(bin_acc, 4)
    stats["win_rate"]        = round(bin_acc, 4)
    stats["ci_lower"]        = round(ci_lo, 4)
    stats["ci_upper"]        = round(ci_hi, 4)
    stats["n_binary"]        = n_bin

    # 5-class sklearn metrics
    if _HAS_SKLEARN and n >= 5:
        stats["direction_accuracy"] = round(accuracy_score(y_true, y_pred), 4)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=ALL_DIRECTIONS,
            average="weighted", zero_division=0,
        )
        stats["direction_precision"] = round(float(p),  4)
        stats["direction_recall"]    = round(float(r),  4)
        stats["direction_f1"]        = round(float(f1), 4)

        p_cl, r_cl, f1_cl, sup_cl = precision_recall_fscore_support(
            y_true, y_pred, labels=ALL_DIRECTIONS,
            average=None, zero_division=0,
        )
        stats["per_class"] = {
            lbl: {
                "precision": round(float(p_cl[i]),  4),
                "recall":    round(float(r_cl[i]),  4),
                "f1":        round(float(f1_cl[i]), 4),
                "support":   int(sup_cl[i]),
            }
            for i, lbl in enumerate(ALL_DIRECTIONS)
        }
        cm = confusion_matrix(y_true, y_pred, labels=ALL_DIRECTIONS)
        stats["confusion_matrix"] = {
            "labels": ALL_DIRECTIONS,
            "matrix": cm.tolist(),
        }

    # Return-based metrics
    stats["mean_return_pct"]   = round(float(ret_series.mean()),   4)
    stats["median_return_pct"] = round(float(ret_series.median()), 4)
    stats["std_return_pct"]    = round(float(ret_series.std()),    4)

    # Sharpe-like (annualised)
    tpy = 252 / h  # periods per year
    if ret_series.std() > 0:
        sharpe = (ret_series.mean() / ret_series.std()) * (tpy ** 0.5)
    else:
        sharpe = 0.0
    stats["sharpe_like"] = round(float(sharpe), 4)

    return stats


def _per_stock_stats(df: pd.DataFrame) -> dict:
    """Compute per-symbol metrics across all horizons with Wilson CI."""
    per_stock: dict = {}

    for sym, grp in df.groupby("symbol"):
        sym_stats: dict = {
            "count":    len(grp),
            "name":     grp.iloc[0].get("name", sym) if len(grp) > 0 else sym,
            "horizons": {},
        }
        for h in FUTURE_DAYS:
            ret_col = f"future_return_{h}d"
            if ret_col not in grp.columns:
                continue
            sub = grp.dropna(subset=["ai_sentiment", ret_col]).copy()
            if sub.empty:
                continue

            sub["pred_dir"] = sub["ai_sentiment"].map(
                lambda s: SENTIMENT_TO_DIRECTION.get(s, "neutral")
            )
            sub["act_dir"]  = sub[ret_col].apply(return_to_direction)
            sub["correct"]  = sub.apply(
                lambda r: binary_correct(r["pred_dir"], r["act_dir"]), axis=1
            )

            bin_df = sub.dropna(subset=["correct"])
            n_bin  = len(bin_df)
            n_corr = int(bin_df["correct"].sum())
            acc    = n_corr / n_bin if n_bin > 0 else 0.0
            ci     = wilson_ci(n_corr, n_bin)

            sym_stats["horizons"][f"{h}d"] = {
                "n":           n_bin,
                "accuracy":    round(acc, 4),
                "ci_lower":    round(ci[0], 4),
                "ci_upper":    round(ci[1], 4),
                "win_rate":    round(float((sub[ret_col] > 0).mean()), 4),
                "mean_return": round(float(sub[ret_col].mean()), 4),
            }

        per_stock[str(sym)] = sym_stats

    return per_stock


def compute_metrics(
    df: pd.DataFrame,
    primary_horizon: int = 3,
) -> dict:
    """
    Compute all evaluation metrics.

    Returns enriched dict:
    {
      "total_records": N,
      "primary_horizon": "3d",
      "horizon_metrics": {
        "1d":  {accuracy, f1, win_rate, sharpe, confusion_matrix, ...},
        "3d":  {...},
        "5d":  {...},
        "10d": {...},
        "20d": {...},
        "60d": {...},
      },
      "per_stock": {
        "2330.TW": {
          "count": N,
          "horizons": {"1d": {accuracy, ci_lower, ci_upper, win_rate, mean_return}, ...}
        },
        ...
      },
      "sentiment_returns": {
        "positive": {count, mean_return, median_return, positive_rate},
        ...
      }
    }
    """
    primary_ret_col = f"future_return_{primary_horizon}d"
    metrics: dict   = {
        "total_records":   len(df),
        "primary_horizon": f"{primary_horizon}d",
    }

    if df.empty:
        log.warning("Empty DataFrame – no metrics to compute.")
        return metrics

    # Date range
    if "date" in df.columns:
        metrics["date_range"] = {
            "start": str(df["date"].min()),
            "end":   str(df["date"].max()),
        }

    # ── Per-horizon metrics ───────────────────────────────────
    horizon_metrics: dict = {}
    for h in FUTURE_DAYS:
        h_stats = _horizon_stats(df, h)
        if h_stats:
            horizon_metrics[f"{h}d"] = h_stats
    metrics["horizon_metrics"] = horizon_metrics

    # ── Per-stock metrics ────────────────────────────────────
    metrics["per_stock"] = _per_stock_stats(df)

    # ── Sentiment → average return (primary horizon) ─────────
    if primary_ret_col in df.columns:
        sent_returns: dict = {}
        for sent, grp in df.groupby("ai_sentiment"):
            ret = grp[primary_ret_col].dropna()
            if not ret.empty:
                sent_returns[str(sent)] = {
                    "count":         int(len(ret)),
                    "mean_return":   round(float(ret.mean()),   4),
                    "median_return": round(float(ret.median()), 4),
                    "positive_rate": round(float((ret > 0).mean()), 4),
                }
        metrics["sentiment_returns"] = sent_returns

    # ── Classification report text (primary horizon) ─────────
    if _HAS_SKLEARN and primary_ret_col in df.columns:
        eval_df = df.dropna(subset=["ai_sentiment", primary_ret_col]).copy()
        eval_df["pred_dir"] = eval_df["ai_sentiment"].map(
            lambda s: SENTIMENT_TO_DIRECTION.get(s, "neutral")
        )
        eval_df["act_dir"]  = eval_df[primary_ret_col].apply(return_to_direction)

        if len(eval_df) >= 5:
            report = classification_report(
                eval_df["act_dir"].tolist(),
                eval_df["pred_dir"].tolist(),
                labels=ALL_DIRECTIONS,
                zero_division=0,
            )
            rpath = results_path("classification_report.txt")
            with open(rpath, "w", encoding="utf-8") as f:
                f.write(f"Primary horizon: {primary_horizon}d  |  "
                        f"Records: {len(eval_df)}\n\n")
                f.write(report)
            log.info("Classification report saved → %s", rpath)

    # ── Legacy fields (for backward compat with older charts) ─
    primary_h_stats = horizon_metrics.get(f"{primary_horizon}d", {})
    for k in ("binary_accuracy", "win_rate", "direction_accuracy",
              "direction_f1_weighted", "direction_precision_weighted",
              "direction_recall_weighted", "mean_return_pct",
              "median_return_pct", "std_return_pct", "sharpe_like",
              "confusion_matrix", "per_class"):
        legacy_key = k
        if k == "direction_f1_weighted":
            legacy_key = "direction_f1"
        elif k == "direction_precision_weighted":
            legacy_key = "direction_precision"
        elif k == "direction_recall_weighted":
            legacy_key = "direction_recall"
        val = primary_h_stats.get(legacy_key) or primary_h_stats.get(k)
        if val is not None:
            metrics[k] = val

    metrics["evaluated"] = primary_h_stats.get("n_binary", primary_h_stats.get("n", 0))

    return metrics


# ──────────────────────────────────────────────────────────────
# Step 5 – Export
# ──────────────────────────────────────────────────────────────

def export_results(df: pd.DataFrame, metrics: dict) -> None:
    """Save prediction_results.csv, prediction_metrics.json, stock_metrics.csv."""

    csv_path = results_path("prediction_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("Results CSV → %s", csv_path)

    json_path = results_path("prediction_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log.info("Metrics JSON → %s", json_path)

    # ── stock_metrics.csv ────────────────────────────────────
    per_stock = metrics.get("per_stock", {})
    stock_rows = []
    for sym, sym_stats in per_stock.items():
        base = {
            "symbol": sym,
            "name":   sym_stats.get("name", sym),
            "count":  sym_stats.get("count", 0),
        }
        for h_label, h_stats in sym_stats.get("horizons", {}).items():
            row = {**base, "horizon": h_label, **h_stats}
            stock_rows.append(row)

    if stock_rows:
        stock_df  = pd.DataFrame(stock_rows)
        stock_path = results_path("stock_metrics.csv")
        stock_df.to_csv(stock_path, index=False, encoding="utf-8-sig")
        log.info("Stock metrics CSV → %s", stock_path)


# ──────────────────────────────────────────────────────────────
# Pretty-print summary
# ──────────────────────────────────────────────────────────────

def print_summary(metrics: dict) -> None:
    print_section("Backtest Metrics Summary")

    print_metric("Total records",    metrics.get("total_records", "—"))
    primary = metrics.get("primary_horizon", "3d")
    print_metric("Primary horizon",  primary)
    if "date_range" in metrics:
        dr = metrics["date_range"]
        print_metric("Date range", f"{dr.get('start','?')} → {dr.get('end','?')}")

    # ── Horizon comparison table ─────────────────────────────
    horizon_metrics = metrics.get("horizon_metrics", {})
    if horizon_metrics:
        print()
        print(info("  Prediction accuracy by horizon:"))
        print(info(f"  {'Horizon':<8} {'n':>5} {'BinAcc':>8} "
                   f"{'95% CI':<22} {'WinRate':>8} {'F1':>7} "
                   f"{'MeanRet':>9} {'Sharpe':>7}"))
        print(info("  " + "─" * 78))

        for h in FUTURE_DAYS:
            key   = f"{h}d"
            stats = horizon_metrics.get(key, {})
            if not stats:
                continue

            acc  = stats.get("binary_accuracy", 0)
            n    = stats.get("n_binary", stats.get("n", 0))
            ci_l = stats.get("ci_lower", 0)
            ci_h = stats.get("ci_upper", 0)
            wr   = stats.get("win_rate", 0)
            f1   = stats.get("direction_f1", "—")
            mr   = stats.get("mean_return_pct", 0)
            sh   = stats.get("sharpe_like", 0)

            acc_s = ok(f"{acc*100:.1f}%") if acc >= 0.55 else warn(f"{acc*100:.1f}%")
            ci_s  = f"[{ci_l*100:.1f}%, {ci_h*100:.1f}%]"
            mr_s  = ok(f"{mr:+.2f}%") if mr >= 0 else err(f"{mr:+.2f}%")
            f1_s  = f"{f1:.4f}" if isinstance(f1, float) else str(f1)

            print(
                f"  {key:<8} {n:>5} {acc_s:>20} "
                f"{ci_s:<22} {wr*100:>7.1f}% "
                f"{f1_s:>7} {mr_s:>14} {sh:>7.3f}"
            )

    # ── Sentiment → return (primary horizon) ─────────────────
    sent_returns = metrics.get("sentiment_returns", {})
    if sent_returns:
        print()
        print(info(f"  Average return by sentiment ({primary} horizon):"))
        for sent in ALL_SENTIMENTS:
            stats = sent_returns.get(sent)
            if not stats:
                continue
            r  = stats["mean_return"]
            r_s = ok(f"{r:+.2f}%") if r >= 0 else err(f"{r:+.2f}%")
            label = sentiment_label_pretty(sent)
            print(f"    {label:<42}  avg: {r_s}  (n={stats['count']})")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main(primary_horizon: int = 3) -> tuple[pd.DataFrame, dict]:
    dataset = load_dataset()
    df      = run_backtest(dataset, primary_horizon=primary_horizon)
    metrics = compute_metrics(df, primary_horizon=primary_horizon)
    export_results(df, metrics)
    print_summary(metrics)
    return df, metrics


if __name__ == "__main__":
    main()