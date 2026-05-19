"""
plotter.py
負責用 matplotlib 產生股票走勢圖。
"""
import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import pandas as pd

# 註冊並設定中文字型，使其跨平台通用
import platform
system = platform.system()

if system == "Windows":
    # Windows 常見中文字型：微軟正黑體、新細明體
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "MingLiU", "sans-serif"]
elif system == "Darwin":
    # macOS 常見中文字型：蘋果儷黑體、黑體-繁
    plt.rcParams["font.sans-serif"] = ["PingFang TC", "Apple LiGothic", "sans-serif"]
else:
    # Linux (如 Ubuntu) 嘗試手動載入 Noto Sans CJK TC
    _font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(_font_path):
        fm.fontManager.addfont(_font_path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_font_path).get_name()
    else:
        # Fallback
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Droid Sans Fallback", "sans-serif"]

plt.rcParams["axes.unicode_minus"] = False  # 避免負號顯示異常


def plot_price_history(history: pd.DataFrame, ticker_symbol: str, save_path: str = None):
    """
    畫出歷史收盤價折線圖並存為 PNG。

    Args:
        history:       由 fetcher.get_price_history() 取得的 DataFrame
        ticker_symbol: 股票代號（用於標題）
        save_path:     若指定路徑則將圖片存至該路徑；若為 None 則預設存為 output/<ticker>.png
    """
    if history.empty:
        print("⚠️  沒有可用的歷史股價資料。")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(history.index, history["Close"], linewidth=1.5, color="#2196F3", label="收盤價")

    # 標記最高與最低點
    max_idx = history["Close"].idxmax()
    min_idx = history["Close"].idxmin()
    ax.scatter(max_idx, history["Close"][max_idx], color="red",   zorder=5, label=f"最高 {history['Close'][max_idx]:.0f}")
    ax.scatter(min_idx, history["Close"][min_idx], color="green", zorder=5, label=f"最低 {history['Close'][min_idx]:.0f}")

    # 格式化 X 軸日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    ax.set_title(f"{ticker_symbol} 歷史股價走勢", fontsize=14)
    ax.set_ylabel("股價（元）")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path is None:
        os.makedirs("output", exist_ok=True)
        save_path = f"output/{ticker_symbol.replace('.', '_')}_price.png"

    plt.savefig(save_path, dpi=150)
    print(f"圖表已儲存至 {save_path}")
    plt.close(fig)
