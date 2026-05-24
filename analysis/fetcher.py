"""
fetcher.py
負責從 yfinance 抓取台股資料。
"""
import urllib.request
import re
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np


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
        print(f"⚠️ 無法取得 {ticker_symbol} 中文名稱: {e}")
    return ticker_symbol


def get_stock_info(ticker_symbol: str) -> dict:
    """
    抓取股票基本面資訊（本益比、ROE、EPS 等）。
    """
    stock = yf.Ticker(ticker_symbol)
    info = stock.info

    # 取得中文名稱
    chinese_name = get_tw_stock_name(ticker_symbol)
    name = chinese_name if chinese_name != ticker_symbol else info.get("shortName", "N/A")

    return {
        "symbol": ticker_symbol,
        "name": name,
        "currency": info.get("currency", "N/A"),
        "current_price": info.get("currentPrice"),
        "pe_ratio": info.get("trailingPE"),
        "roe": info.get("returnOnEquity"),
        "eps": info.get("trailingEps"),
    }


def get_price_history(ticker_symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    抓取股票的歷史股價資料。
    """
    stock = yf.Ticker(ticker_symbol)
    history = stock.history(period=period)
    return history


def get_historical_pe(ticker_symbol: str, period: str = "3y") -> pd.DataFrame:
    """
    計算歷史本益比 (P/E Ratio)。
    優先使用 quarterly earnings 數據計算季度的 TTM EPS（滾動四季加總），
    並以階梯式 (step-wise) 對齊歷史股價，使估值河流在財報公布日產生階梯波動。
    """
    stock = yf.Ticker(ticker_symbol)
    history = stock.history(period=period)
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
        print(f"⚠️ 獲取季度 TTM EPS 失敗: {e}")

    if not used_quarterly:
        # 簡單 fallback
        info = stock.info
        current_eps = info.get("trailingEps") or 1.0
        daily_eps = daily_eps.fillna(current_eps)

    df = pd.DataFrame(index=history.index)
    df["Close"] = history["Close"]
    df["EPS"] = daily_eps
    df["PE"] = df["Close"] / df["EPS"]

    return df


def get_stock_news(symbol: str, start_date_str: str = None) -> dict:
    """
    透過 FinMind API 獲取台股相關新聞。
    限制一次只能抓一天，我們每次查詢最多往前推 7 天，直到抓滿 5 篇。
    回傳：{"news": list[dict], "next_start_date": str, "has_more": bool}
    """
    import urllib.request
    import json
    from datetime import datetime, timedelta

    # FinMind 股票代號不含 .TW
    stock_id = symbol.split('.')[0]
    
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        except Exception:
            start_date = datetime.now()
    else:
        start_date = datetime.now()
        
    news_list = []
    days_searched = 0
    max_days_to_search = 7
    
    current_search_date = start_date
    
    while days_searched < max_days_to_search:
        target_date_str = current_search_date.strftime("%Y-%m-%d")
        url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockNews&data_id={stock_id}&start_date={target_date_str}"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                
                if res_data.get("status") == 200:
                    daily_news = res_data.get("data", [])
                    for item in daily_news:
                        news_list.append({
                            "title": item.get("title", "無標題"),
                            "source": item.get("source", "財經新聞"),
                            "date": item.get("date", "").split(" ")[0],
                            "url": item.get("link", "#")
                        })
        except Exception as e:
            print(f"⚠️ FinMind 新聞獲取失敗 ({target_date_str}): {e}")
            
        current_search_date -= timedelta(days=1)
        days_searched += 1
        
        # 如果抓滿了 5 篇，就提早結束
        if len(news_list) >= 5:
            break
            
    # 計算是否還有更多新聞
    # 限制最多只往前追溯 30 天的歷史新聞，避免無限往前抓取
    delta = datetime.now() - current_search_date
    has_more = True
    if delta.days > 30:
        has_more = False
    elif len(news_list) == 0 and days_searched >= max_days_to_search:
        # 如果連續找了 7 天都沒半篇新聞，也宣告沒有更多新聞，避免讓前端無限載入
        has_more = False
        
    return {
        "news": news_list,
        "next_start_date": current_search_date.strftime("%Y-%m-%d"),
        "has_more": has_more
    }

