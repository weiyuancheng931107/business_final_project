"""
fetcher.py
負責從 yfinance 抓取台股資料。
"""
import yfinance as yf


def get_stock_info(ticker_symbol: str) -> dict:
    """
    抓取股票基本面資訊（本益比、ROE、EPS 等）。

    Args:
        ticker_symbol: 股票代號，台股格式為 "XXXX.TW"（上市）或 "XXXX.TWO"（上櫃）

    Returns:
        包含基本面數據的 dict
    """
    stock = yf.Ticker(ticker_symbol)
    info = stock.info

    return {
        "symbol": ticker_symbol,
        "name": info.get("shortName", "N/A"),
        "currency": info.get("currency", "N/A"),
        "current_price": info.get("currentPrice"),
        "pe_ratio": info.get("trailingPE"),
        "roe": info.get("returnOnEquity"),
        "eps": info.get("trailingEps"),
    }


def get_price_history(ticker_symbol: str, period: str = "1y") -> "pd.DataFrame":
    """
    抓取股票的歷史股價資料。

    Args:
        ticker_symbol: 股票代號
        period:        資料區間，例如 "1y"（一年）、"6mo"（六個月）

    Returns:
        包含 OHLCV 的 pandas DataFrame
    """
    import pandas as pd
    stock = yf.Ticker(ticker_symbol)
    history = stock.history(period=period)
    return history
