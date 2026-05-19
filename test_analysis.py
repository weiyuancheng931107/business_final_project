"""
test_analysis.py
以台積電 (2330.TW) 為例的測試腳本，用以驗證各模組是否正常運作。
"""
from analysis.fetcher import get_stock_info, get_price_history
from analysis.plotter import plot_price_history

TICKER = "2330.TW"

# --- 1. 基本面資訊 ---
info = get_stock_info(TICKER)
print("=" * 35)
print(f"股票代號：{info['symbol']}")
print(f"股票名稱：{info['name']}")
print(f"目前股價：{info['current_price']} {info['currency']}")
print(f"本益比  ：{info['pe_ratio']}")
print(f"ROE     ：{info['roe']}")
print(f"EPS     ：{info['eps']}")
print("=" * 35)

# --- 2. 歷史股價圖 ---
history = get_price_history(TICKER, period="1y")
plot_price_history(history, TICKER)
