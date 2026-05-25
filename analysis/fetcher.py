"""
fetcher.py
負責從 yfinance 抓取台股資料。
"""
import json
import os
import urllib.parse
import urllib.request
import re
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
import numpy as np


NEWS_LIMIT = 8
FINMIND_LOOKBACK_DAYS = 7
MAX_FINMIND_HISTORY_DAYS = 30
PERIOD_TO_DAYS = {
    "5d": 5,
    "1mo": 31,
    "3mo": 93,
    "6mo": 186,
    "1y": 366,
    "2y": 732,
    "3y": 1098,
    "5y": 1830,
    "10y": 3650,
}

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YFINANCE_CACHE_DIR = os.path.join(_BASE_DIR, ".yfinance_cache")

try:
    os.makedirs(_YFINANCE_CACHE_DIR, exist_ok=True)
    yf.set_tz_cache_location(_YFINANCE_CACHE_DIR)
except Exception as e:
    print(f"[WARN] yfinance cache setup failed: {e}")


def get_tw_stock_name(ticker_symbol: str) -> str:
    """
    透過爬取奇摩股市取得台股的中文名稱。
    若非台股（如美股）或爬取失敗，則回傳原始代號。
    """
    code = ticker_symbol.split(".")[0]
    if not code.isdigit():
        return ticker_symbol

    url = f"https://tw.stock.yahoo.com/quote/{code}"
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8')
            title_match = re.search(r'<title>(.*?)</title>', html)
            if title_match:
                title = title_match.group(1)
                name_match = re.search(r'^([^\s\(]+)', title)
                if name_match:
                    return name_match.group(1)
    except Exception as e:
        print(f"[WARN] Unable to fetch Chinese name for {ticker_symbol}: {e}")
    return ticker_symbol


def get_stock_info(ticker_symbol: str) -> dict:
    """
    抓取股票基本面資訊（本益比、ROE、EPS 等）。
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info or {}
        if not isinstance(info, dict):
            info = {}
    except Exception as e:
        print(f"[WARN] yfinance stock info failed ({ticker_symbol}): {e}")
        info = {}

    # 取得中文名稱
    chinese_name = get_tw_stock_name(ticker_symbol)
    name = chinese_name if chinese_name != ticker_symbol else info.get("shortName", "N/A")
    current_price = _first_number(
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
        info.get("previousClose"),
    )

    if current_price is None:
        history = get_price_history(ticker_symbol, period="1mo")
        if not history.empty and "Close" in history.columns:
            current_price = _to_number(history["Close"].dropna().iloc[-1])

    return {
        "symbol": ticker_symbol,
        "name": name,
        "currency": info.get("currency", "N/A"),
        "current_price": current_price,
        "pe_ratio": _to_number(info.get("trailingPE")),
        "roe": _to_number(info.get("returnOnEquity")),
        "eps": _to_number(info.get("trailingEps")),
    }


def _to_number(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _first_number(*values):
    for value in values:
        number = _to_number(value)
        if number is not None:
            return number
    return None


def _period_date_range(period: str) -> tuple[str, str]:
    today = datetime.now()
    if period == "ytd":
        start = datetime(today.year, 1, 1)
    else:
        start = today - timedelta(days=PERIOD_TO_DAYS.get(period, PERIOD_TO_DAYS["1y"]))
    return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


def _fetch_yfinance_price_history(ticker_symbol: str, period: str) -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker_symbol)
        return stock.history(period=period)
    except Exception as e:
        print(f"[WARN] yfinance price history failed ({ticker_symbol}, {period}): {e}")
        return pd.DataFrame()


def _fetch_finmind_price_history(ticker_symbol: str, period: str) -> pd.DataFrame:
    stock_id = ticker_symbol.split(".")[0]
    if not stock_id.isdigit():
        return pd.DataFrame()

    start_date, end_date = _period_date_range(period)
    params = urllib.parse.urlencode({
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
    })
    url = f"https://api.finmindtrade.com/api/v4/data?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[WARN] FinMind price history failed ({ticker_symbol}, {period}): {e}")
        return pd.DataFrame()

    if res_data.get("status") != 200 or not res_data.get("data"):
        return pd.DataFrame()

    df = pd.DataFrame(res_data["data"])
    required_columns = {"date", "open", "max", "min", "close"}
    if not required_columns.issubset(df.columns):
        return pd.DataFrame()

    history = pd.DataFrame()
    history["Open"] = pd.to_numeric(df["open"], errors="coerce")
    history["High"] = pd.to_numeric(df["max"], errors="coerce")
    history["Low"] = pd.to_numeric(df["min"], errors="coerce")
    history["Close"] = pd.to_numeric(df["close"], errors="coerce")
    history["Volume"] = pd.to_numeric(df.get("Trading_Volume", 0), errors="coerce").fillna(0)
    history.index = pd.to_datetime(df["date"], errors="coerce")
    history = history[history.index.notna()].dropna(subset=["Close"]).sort_index()
    return history


def get_price_history(ticker_symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    抓取股票的歷史股價資料。
    """
    history = _fetch_yfinance_price_history(ticker_symbol, period)
    if history.empty:
        history = _fetch_finmind_price_history(ticker_symbol, period)
    return history


def get_historical_pe(ticker_symbol: str, period: str = "3y") -> pd.DataFrame:
    """
    計算歷史本益比 (P/E Ratio)。
    優先使用 quarterly earnings 數據計算季度的 TTM EPS（滾動四季加總），
    並以階梯式 (step-wise) 對齊歷史股價，使估值河流在財報公布日產生階梯波動。
    """
    stock = yf.Ticker(ticker_symbol)
    history = get_price_history(ticker_symbol, period=period)
    if history.empty:
        return pd.DataFrame()

    history.index = history.index.tz_localize(None)
    daily_eps = pd.Series(dtype=float, index=history.index)
    used_quarterly = False

    try:
        earnings = stock.earnings_dates
        if earnings is not None and "Reported EPS" in earnings.columns:
            q_eps = earnings["Reported EPS"].dropna()
            if not q_eps.empty:
                q_eps.index = pd.to_datetime(q_eps.index).tz_localize(None).normalize()
                q_eps = q_eps.sort_index()

                # TTM EPS = 滾動 4 季 Reported EPS 總和
                ttm_eps = q_eps.rolling(4).sum().dropna()

                if not ttm_eps.empty:
                    union_idx = history.index.union(ttm_eps.index).sort_values()
                    daily_eps = ttm_eps.reindex(union_idx).ffill().bfill()
                    daily_eps = daily_eps.reindex(history.index)
                    used_quarterly = True
    except Exception as e:
        print(f"[WARN] Failed to fetch quarterly TTM EPS: {e}")

    if not used_quarterly:
        # 簡單 fallback
        try:
            info = stock.info or {}
            current_eps = _to_number(info.get("trailingEps")) or 1.0
        except Exception as e:
            print(f"[WARN] yfinance EPS fallback failed ({ticker_symbol}): {e}")
            current_eps = 1.0
        daily_eps = daily_eps.fillna(current_eps)

    df = pd.DataFrame(index=history.index)
    df["Close"] = history["Close"]
    df["EPS"] = daily_eps
    df["PE"] = df["Close"] / df["EPS"]

    return df


def _parse_news_date(value) -> str:
    """Normalize provider date values to YYYY-MM-DD."""
    if value is None:
        return ""

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            return ""

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return ""
        try:
            return datetime.fromisoformat(cleaned.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except Exception:
            return cleaned.split("T")[0].split(" ")[0]

    return ""


def _nested_url(value) -> str:
    if isinstance(value, dict):
        return value.get("url") or value.get("href") or ""
    if isinstance(value, str):
        return value
    return ""


def _normalize_yfinance_news_item(item: dict) -> dict | None:
    """
    yfinance has used both flat and nested news payloads. Convert either shape
    to the frontend contract: title/source/date/url.
    """
    if not isinstance(item, dict):
        return None

    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}

    title = item.get("title") or content.get("title") or content.get("headline")
    if not title:
        return None

    source = (
        item.get("publisher")
        or provider.get("displayName")
        or provider.get("name")
        or "Yahoo Finance"
    )
    date_value = (
        item.get("providerPublishTime")
        or item.get("pubDate")
        or content.get("pubDate")
        or content.get("displayTime")
    )
    url = (
        item.get("link")
        or _nested_url(content.get("canonicalUrl"))
        or _nested_url(content.get("clickThroughUrl"))
        or "#"
    )

    return {
        "title": str(title).strip(),
        "source": str(source).strip() or "Yahoo Finance",
        "date": _parse_news_date(date_value),
        "url": url,
    }


def _normalize_finmind_news_item(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None

    title = item.get("title") or item.get("headline")
    if not title:
        return None

    return {
        "title": str(title).strip(),
        "source": item.get("source") or "FinMind",
        "date": _parse_news_date(item.get("date") or item.get("publish_date")),
        "url": item.get("link") or item.get("url") or "#",
    }


def _dedupe_news(news_items: list[dict]) -> list[dict]:
    seen = set()
    unique_items = []

    for item in news_items:
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        date = item.get("date", "").strip()
        if not title:
            continue

        key = url.lower() if url and url != "#" else f"{title.lower()}|{date}"
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)

    return unique_items


def _parse_start_date(start_date_str: str | None) -> datetime:
    if not start_date_str:
        return datetime.now()
    try:
        return datetime.strptime(start_date_str, "%Y-%m-%d")
    except Exception:
        return datetime.now()


def _fetch_yfinance_news(symbol: str, limit: int = NEWS_LIMIT) -> list[dict]:
    stock = yf.Ticker(symbol)
    raw_news = stock.news or []

    normalized_news = []
    for item in raw_news:
        news_item = _normalize_yfinance_news_item(item)
        if news_item:
            normalized_news.append(news_item)
        if len(normalized_news) >= limit:
            break

    return normalized_news


def _fetch_finmind_news(symbol: str, start_date_str: str | None = None, limit: int = NEWS_LIMIT) -> dict:
    stock_id = symbol.split(".")[0]
    news_list = []
    current_date = _parse_start_date(start_date_str)
    days_searched = 0

    while days_searched < FINMIND_LOOKBACK_DAYS and len(news_list) < limit:
        target_date_str = current_date.strftime("%Y-%m-%d")
        params = urllib.parse.urlencode({
            "dataset": "TaiwanStockNews",
            "data_id": stock_id,
            "start_date": target_date_str,
        })
        url = f"https://api.finmindtrade.com/api/v4/data?{params}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))

            if res_data.get("status") == 200:
                for item in res_data.get("data", []):
                    news_item = _normalize_finmind_news_item(item)
                    if news_item:
                        news_list.append(news_item)
        except Exception as e:
            print(f"[WARN] FinMind news fetch failed ({target_date_str}): {e}")

        current_date -= timedelta(days=1)
        days_searched += 1

    news_list = sorted(news_list, key=lambda item: item.get("date", ""), reverse=True)
    news_list = _dedupe_news(news_list)[:limit]

    oldest_allowed_date = datetime.now() - timedelta(days=MAX_FINMIND_HISTORY_DAYS)

    return {
        "news": news_list,
        "next_start_date": current_date.strftime("%Y-%m-%d"),
        "has_more": current_date >= oldest_allowed_date,
    }


def get_stock_news(symbol: str, start_date_str: str = None) -> dict:
    """
    Fetch real financial news.

    First page combines yfinance's built-in Yahoo Finance feed with FinMind's
    TaiwanStockNews dataset. Subsequent pages use FinMind because it supports
    date-based pagination.
    """
    if start_date_str:
        try:
            return _fetch_finmind_news(symbol, start_date_str, limit=NEWS_LIMIT)
        except Exception as e:
            print(f"[WARN] FinMind news fetch failed ({start_date_str}): {e}")
            return {"news": [], "next_start_date": start_date_str, "has_more": False}

    yfinance_news = []
    try:
        yfinance_news = _fetch_yfinance_news(symbol, limit=NEWS_LIMIT)
    except Exception as e:
        print(f"[WARN] yfinance news fetch failed ({symbol}): {e}")

    finmind_data = {"news": [], "next_start_date": None, "has_more": False}
    try:
        finmind_data = _fetch_finmind_news(symbol, None, limit=NEWS_LIMIT)
    except Exception as e:
        print(f"[WARN] FinMind news fetch failed ({symbol}): {e}")

    combined_news = _dedupe_news(yfinance_news + finmind_data["news"])[:NEWS_LIMIT]

    return {
        "news": combined_news,
        "next_start_date": finmind_data.get("next_start_date"),
        "has_more": finmind_data.get("has_more", False),
    }
