#!/usr/bin/env python3
"""
build_dataset.py
─────────────────
Auto-build sentiment_dataset.json from real historical news + GPT analysis.

How it works
────────────
1. For each stock in portfolio.json (or --symbols), call
   fetcher.get_stock_news_range() which iterates day-by-day over the
   requested date range, making one FinMind API request per calendar day.

2. Collect every article found (no article-count cap).

3. Split articles into batches of --batch-size (default 20) and call GPT
   once per batch.  This reduces GPT calls by ~20× vs. one-per-article.

4. Save each article with its AI-assigned sentiment_type to the dataset.

5. Resume support: a progress file tracks which (symbol, date) pairs have
   already been queried, so interrupted runs restart where they left off.

API budget estimate (10 stocks, 365-day range)
───────────────────────────────────────────────
FinMind  : 365 × 10 = 3,650 requests
           @ 0.5 s delay  →  ~30 min
           @ 0.12 s delay →  ~7 min (requires FinMind token)

GPT      : depends on article density.
           Typical: ~5–15 articles/day/stock × 10 stocks × 365 days
           ≈ 18,000–55,000 articles total ÷ 20 per batch ≈ 900–2,750 calls

Usage
─────
# Full 365-day backfill using all stocks in portfolio.json
python tests/build_dataset.py --start-date 2025-01-01 --end-date 2025-12-31

# Specific stocks only
python tests/build_dataset.py --symbols 2330.TW 2454.TW \\
    --start-date 2025-01-01 --end-date 2025-12-31

# Resume interrupted run (skips already-fetched dates automatically)
python tests/build_dataset.py --start-date 2025-01-01 --end-date 2025-12-31

# Faster with a FinMind registered token
python tests/build_dataset.py --start-date 2025-01-01 --end-date 2025-12-31 \\
    --delay 0.12

# Rebuild from scratch (ignore existing dataset + progress)
python tests/build_dataset.py --start-date 2025-01-01 --end-date 2025-12-31 \\
    --fresh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from tqdm import tqdm
except ImportError:
    class DummyTqdm:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, *args, **kwargs):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

        def close(self):
            pass

    def tqdm(*args, **kwargs):
        return DummyTqdm()

from tests.utils import (
    bold,
    err,
    get_logger,
    info,
    ok,
    print_section,
    print_metric,
    warn,
    accent,
    sentiment_label_pretty,
    retry,
)
log = get_logger("build_dataset")

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PORTFOLIO_FILE   = os.path.join(_PROJECT_ROOT, "portfolio.json")
DATASET_OUT      = os.path.join(_HERE, "sentiment_dataset.json")
PROGRESS_FILE    = os.path.join(_HERE, "sentiment_dataset_progress.json")
FLASK_BASE       = "http://127.0.0.1:2330"
DEFAULT_DELAY    = 0.5    # seconds between FinMind requests
DEFAULT_BATCH    = 20     # articles per GPT call
GPT_CALL_DELAY   = 1.0    # seconds between GPT calls


# ──────────────────────────────────────────────────────────────
# Portfolio
# ──────────────────────────────────────────────────────────────

def load_portfolio() -> list[dict]:
    if not os.path.exists(PORTFOLIO_FILE):
        log.warning("portfolio.json not found at %s", PORTFOLIO_FILE)
        return []
    with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    stocks = data.get("stocks", [])
    log.info("Loaded %d stocks from portfolio.json", len(stocks))
    return stocks


# ──────────────────────────────────────────────────────────────
# Progress tracking (resume support)
# ──────────────────────────────────────────────────────────────

def load_progress() -> dict[str, set]:
    """
    Load the progress file.
    Returns dict: symbol → set of date strings already queried.
    """
    if not os.path.exists(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {sym: set(dates) for sym, dates in raw.items()}
    except Exception as exc:
        log.warning("Could not load progress file: %s", exc)
        return {}


def save_progress(progress: dict[str, set]) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({sym: sorted(dates) for sym, dates in progress.items()},
                  f, ensure_ascii=False, indent=2)


def _mark_done(progress: dict[str, set], symbol: str, date_str: str) -> None:
    progress.setdefault(symbol, set()).add(date_str)


# ──────────────────────────────────────────────────────────────
# News fetcher
# ──────────────────────────────────────────────────────────────

def fetch_news_range(
    symbol: str,
    start_date: str,
    end_date: str,
    request_delay: float,
    skip_dates: set,
) -> list[dict]:
    """
    Wrapper around fetcher.get_stock_news_range with tqdm progress.
    """
    from analysis.fetcher import get_stock_news_range

    total_days = (
        datetime.strptime(end_date, "%Y-%m-%d")
        - datetime.strptime(start_date, "%Y-%m-%d")
    ).days + 1

    bar = tqdm(
        total=total_days,
        desc=f"  {symbol} FinMind",
        unit="day",
        leave=False,
        ncols=80,
    )

    def _cb(date_str, day_idx, total, collected):
        bar.set_postfix({"date": date_str, "found": collected}, refresh=False)
        bar.update(1)

    try:
        articles = get_stock_news_range(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            request_delay=request_delay,
            skip_dates=skip_dates,
            progress_callback=_cb,
        )
    finally:
        bar.close()

    return articles


# ──────────────────────────────────────────────────────────────
# GPT sentiment analysis  (batched)
# ──────────────────────────────────────────────────────────────

def _flask_available() -> bool:
    try:
        import requests
        r = requests.get(f"{FLASK_BASE}/api/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@retry(max_attempts=3, delay=2.0, backoff=2.0)
def _call_flask_batch(symbol: str, news_list: list[dict]) -> dict:
    import requests
    resp = requests.post(
        f"{FLASK_BASE}/api/stock/{symbol}/analyze_news",
        json={"news": news_list},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


@retry(max_attempts=3, delay=2.0, backoff=2.0)
def _call_openai_batch(symbol: str, news_list: list[dict]) -> dict:
    from openai import OpenAI
    import config
    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_API_BASE)

    news_text = "\n".join(f"- {n['title']}" for n in news_list)
    prompt = f"""你是一位台股分析師。

以下是 {symbol} 的新聞（共 {len(news_list)} 篇）：

{news_text}

請分析每篇新聞情緒。情緒類型只能是以下五種之一：
positive / slightly_positive / neutral / slightly_negative / negative

請只輸出 JSON（不含任何 markdown code block），格式如下：
{{
  "summary": "整體摘要",
  "details": [
    {{
      "title": "新聞標題（完整原文）",
      "sentiment_label": "正向",
      "sentiment_type": "positive",
      "reason": "判斷原因（一句話）"
    }}
  ]
}}

注意：
1. details 陣列長度必須等於輸入新聞數量 ({len(news_list)})。
2. 若新聞標題或內容包含雙引號 (")，請務必加上反斜線跳脫 (\\\")，或改用單引號，確保 JSON 格式完全合法。"""

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4096,
    )
    raw = response.choices[0].message.content
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.decoder.JSONDecodeError as e:
        # 如果解析失敗，印出最後的字串幫助 debug，然後將錯誤往上拋給 retry 處理
        print(f"\n[JSONDecodeError] LLM 回傳無法解析的 JSON:\n{raw}\n")
        raise e


def _parse_sentiment_result(result) -> Optional[str]:
    try:
        details = result.get("details", [])
        if details:
            return details[0].get("sentiment_type", "neutral")
    except Exception:
        pass
    return None


def _build_detail_map(result: dict) -> dict[str, dict]:
    """Build title → detail lookup from GPT response."""
    detail_map: dict[str, dict] = {}
    for d in result.get("details", []):
        t = str(d.get("title", "")).strip()
        if t:
            detail_map[t] = d
    return detail_map


def _fuzzy_match(title: str, detail_map: dict[str, dict]) -> Optional[dict]:
    """Exact match first, then first-N-chars prefix match."""
    if title in detail_map:
        return detail_map[title]
    prefix = title[:10]
    for key, val in detail_map.items():
        if key.startswith(prefix):
            return val
    return None


def analyze_batch(
    symbol: str,
    news_batch: list[dict],
    use_flask: bool,
) -> dict:
    """Call GPT for a batch of news articles. Returns raw API result."""
    if use_flask:
        try:
            result = _call_flask_batch(symbol, news_batch)
            if result.get("details"):
                return result
            log.warning("Flask returned empty details; falling back to OpenAI direct.")
        except Exception as exc:
            log.warning("Flask failed (%s); falling back to OpenAI direct: %s", symbol, exc)

    return _call_openai_batch(symbol, news_batch)


# ──────────────────────────────────────────────────────────────
# Core builder
# ──────────────────────────────────────────────────────────────

def _load_existing_dataset() -> list[dict]:
    if not os.path.exists(DATASET_OUT):
        return []
    try:
        with open(DATASET_OUT, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_dataset(records: list[dict]) -> None:
    records_sorted = sorted(records, key=lambda r: r.get("date", ""), reverse=True)
    with open(DATASET_OUT, "w", encoding="utf-8") as f:
        json.dump(records_sorted, f, ensure_ascii=False, indent=2)


def _existing_keys(records: list[dict]) -> set[tuple]:
    """Return set of (symbol, title) tuples already analyzed."""
    return {(r["symbol"], r["title"]) for r in records if "symbol" in r and "title" in r}


def build_dataset(
    stocks: list[dict],
    start_date: str,
    end_date: str,
    request_delay: float = DEFAULT_DELAY,
    batch_size: int = DEFAULT_BATCH,
    use_flask: bool = False,
    fresh: bool = False,
) -> list[dict]:
    """
    Main dataset builder.

    Parameters
    ----------
    stocks        : list of {symbol, name} dicts
    start_date    : "YYYY-MM-DD"
    end_date      : "YYYY-MM-DD"
    request_delay : seconds between FinMind requests
    batch_size    : articles per GPT call (default 20)
    use_flask     : prefer Flask API for sentiment
    fresh         : if True, ignore existing dataset + progress
    """
    # ── Load existing state ──────────────────────────────────
    existing_records: list[dict] = [] if fresh else _load_existing_dataset()
    progress: dict[str, set]     = {} if fresh else load_progress()
    analyzed_keys                = _existing_keys(existing_records)
    new_records: list[dict]      = []
    errors: list[str]            = []

    flask_ok = use_flask and _flask_available()
    if use_flask and not flask_ok:
        log.warning("Flask not available; using OpenAI direct.")

    print_section(f"Building Dataset: {start_date} → {end_date}")
    print_metric("Stocks",          len(stocks))
    print_metric("Date range",      f"{start_date} → {end_date}")
    print_metric("FinMind delay",   f"{request_delay}s per request")
    print_metric("GPT batch size",  batch_size)
    print_metric("GPT mode",        "Flask API" if flask_ok else "OpenAI direct")
    print_metric("Incremental",     "No (--fresh)" if fresh else "Yes")
    print()

    total_days = (
        datetime.strptime(end_date, "%Y-%m-%d")
        - datetime.strptime(start_date, "%Y-%m-%d")
    ).days + 1

    for stock in stocks:
        symbol = stock["symbol"]
        name   = stock.get("name", symbol)

        print()
        print(info(f"━━━  {name} ({symbol})  ━━━"))

        # ── 1. Fetch all news for date range ─────────────────
        already_done = progress.get(symbol, set())
        skipping     = len(already_done)
        remaining    = total_days - skipping

        if remaining <= 0:
            print(ok(f"  All {total_days} days already fetched for {symbol}; skipping FinMind."))
            print_metric("  Existing analyzed records",
                         sum(1 for r in existing_records if r.get("symbol") == symbol))
            continue

        if skipping:
            print(info(f"  Resuming: skipping {skipping} already-fetched days, "
                       f"querying {remaining} remaining."))

        articles = fetch_news_range(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            request_delay=request_delay,
            skip_dates=already_done,
        )

        # Mark every day in range as done (including days with 0 articles)
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt  = datetime.strptime(end_date, "%Y-%m-%d")
        while current <= end_dt:
            _mark_done(progress, symbol, current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        save_progress(progress)

        print(ok(f"  FinMind: fetched {len(articles)} articles"))

        # ── 2. Filter out already-analyzed articles ──────────
        new_articles = [
            a for a in articles
            if (symbol, a["title"]) not in analyzed_keys
        ]
        skipped_analyzed = len(articles) - len(new_articles)
        if skipped_analyzed:
            print(info(f"  Skipping {skipped_analyzed} already-analyzed articles (incremental)."))

        if not new_articles:
            print(ok("  Nothing new to analyze."))
            continue

        print(info(f"  Sending {len(new_articles)} articles to GPT "
                   f"in batches of {batch_size}…"))

        # ── 3. Batch GPT analysis ────────────────────────────
        n_batches   = (len(new_articles) + batch_size - 1) // batch_size
        gpt_success = 0
        gpt_errors  = 0

        for i in tqdm(range(n_batches), desc=f"  {symbol} GPT", unit="batch", ncols=80):
            batch = new_articles[i * batch_size:(i + 1) * batch_size]

            try:
                result     = analyze_batch(symbol, batch, use_flask=flask_ok)
                detail_map = _build_detail_map(result)
                summary    = result.get("summary", "")

                for article in batch:
                    title  = article["title"].strip()
                    detail = _fuzzy_match(title, detail_map)

                    if detail is None:
                        detail = {
                            "sentiment_type":  "neutral",
                            "sentiment_label": "中性",
                            "reason":          "AI 未回傳對應分析",
                        }

                    record = {
                        "symbol":          symbol,
                        "name":            name,
                        "date":            article.get("date", ""),
                        "title":           title,
                        "source":          article.get("source", ""),
                        "url":             article.get("url", "#"),
                        "sentiment_type":  detail.get("sentiment_type", "neutral"),
                        "sentiment_label": detail.get("sentiment_label", "中性"),
                        "reason":          detail.get("reason", ""),
                        "summary":         summary,
                        "analyzed_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    new_records.append(record)
                    analyzed_keys.add((symbol, title))
                    gpt_success += 1

                time.sleep(GPT_CALL_DELAY)

            except Exception as exc:
                log.error("GPT batch %d/%d failed for %s: %s", i + 1, n_batches, symbol, exc)
                gpt_errors += 1

        print(ok(f"  GPT: {gpt_success} articles analyzed, {gpt_errors} batch errors"))

        if gpt_errors:
            errors.append(f"{symbol} ({gpt_errors} GPT batch errors)")

        # ── 4. Save incrementally after each stock ───────────
        all_records = existing_records + new_records
        _save_dataset(all_records)
        log.info("Saved %d total records (after %s).", len(all_records), symbol)

    # ── Final save + report ──────────────────────────────────
    all_records = existing_records + new_records
    _save_dataset(all_records)

    _print_dataset_report(all_records, stocks, start_date, end_date)

    if errors:
        print(warn(f"\n  Errors encountered: {'; '.join(errors)}"))

    return all_records


# ──────────────────────────────────────────────────────────────
# Dataset report
# ──────────────────────────────────────────────────────────────

def _print_dataset_report(
    records: list[dict],
    stocks: list[dict],
    start_date: str,
    end_date: str,
) -> None:
    import pandas as pd

    print()
    print(bold("=" * 52))
    print(bold("  DATASET SUMMARY"))
    print(bold("=" * 52))

    df = pd.DataFrame(records) if records else pd.DataFrame()

    if df.empty:
        print(warn("  No records in dataset."))
        return

    total_articles  = len(df)
    unique_symbols  = df["symbol"].nunique() if "symbol" in df.columns else 0
    date_min        = df["date"].min() if "date" in df.columns else "?"
    date_max        = df["date"].max() if "date" in df.columns else "?"

    for stock in stocks:
        sym  = stock["symbol"]
        name = stock.get("name", sym)
        sub  = df[df["symbol"] == sym] if "symbol" in df.columns else pd.DataFrame()

        if sub.empty:
            print(f"\n  {accent(sym)} ({name})")
            print(warn("    No articles collected."))
            continue

        sym_min  = sub["date"].min() if "date" in sub.columns else "?"
        sym_max  = sub["date"].max() if "date" in sub.columns else "?"
        sym_cnt  = len(sub)

        # Sentiment breakdown
        if "sentiment_type" in sub.columns:
            counts = sub["sentiment_type"].value_counts()
            breakdown = " | ".join(
                f"{k}: {v}"
                for k, v in counts.items()
            )
        else:
            breakdown = "—"

        print(f"\n  {accent(sym)} ({name})")
        print(f"    Articles   : {ok(str(sym_cnt))}")
        print(f"    Date range : {sym_min} ~ {sym_max}")
        print(f"    Sentiment  : {breakdown}")

    print()
    print(bold("─" * 52))
    print(bold(f"  Total Articles : {ok(str(total_articles))}"))
    print(bold(f"  Unique Stocks  : {unique_symbols}"))
    print(bold(f"  Date Coverage  : {date_min} ~ {date_max}"))
    print(bold("=" * 52))
    print()
    print(info(f"  Saved → {DATASET_OUT}"))
    print(info(f"  Progress → {PROGRESS_FILE}"))
    print()


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    today     = datetime.now().strftime("%Y-%m-%d")
    one_yr    = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    p = argparse.ArgumentParser(
        description="Auto-build sentiment_dataset.json from real historical news + GPT"
    )
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Symbols to process (default: all in portfolio.json)")
    p.add_argument("--start-date", default=one_yr,
                   help=f"Start date YYYY-MM-DD (default: 1 year ago = {one_yr})")
    p.add_argument("--end-date", default=today,
                   help=f"End date YYYY-MM-DD (default: today = {today})")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                   help=f"Seconds between FinMind requests (default: {DEFAULT_DELAY})")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                   help=f"Articles per GPT call (default: {DEFAULT_BATCH})")
    p.add_argument("--flask", action="store_true",
                   help="Use Flask API for sentiment (needs app.py running)")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore existing dataset + progress, rebuild from scratch")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print()
    print(bold("╔══════════════════════════════════════════════════════╗"))
    print(bold("║  Dataset Builder  –  台股新聞情緒自動標記工具       ║"))
    print(bold("╚══════════════════════════════════════════════════════╝"))
    print()

    # Validate dates
    try:
        s = datetime.strptime(args.start_date, "%Y-%m-%d")
        e = datetime.strptime(args.end_date,   "%Y-%m-%d")
        if s > e:
            print(err("--start-date must be before --end-date"))
            sys.exit(1)
    except ValueError as exc:
        print(err(f"Invalid date format: {exc}"))
        sys.exit(1)

    # Load stocks
    if args.symbols:
        stocks = [
            {"symbol": s if "." in s else s + ".TW", "name": s}
            for s in args.symbols
        ]
    else:
        stocks = load_portfolio()
        if not stocks:
            print(err("portfolio.json is empty or not found."))
            print(warn("Use --symbols 2330.TW 2454.TW ... to specify manually."))
            sys.exit(1)

    build_dataset(
        stocks       = stocks,
        start_date   = args.start_date,
        end_date     = args.end_date,
        request_delay= args.delay,
        batch_size   = args.batch_size,
        use_flask    = args.flask,
        fresh        = args.fresh,
    )

    print(info("  Next step:  python tests/evaluate_prediction.py"))
    print()


if __name__ == "__main__":
    main()