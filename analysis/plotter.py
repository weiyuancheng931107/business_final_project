"""
plotter.py
負責用 matplotlib 產生股票走勢圖。
所有圖表預設輸出至 static/images/ 目錄供 Flask 前端讀取。
"""
import os
import time
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# 註冊並設定中文字型，使其跨平台通用
import platform
system = platform.system()

if system == "Windows":
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "MingLiU", "sans-serif"]
elif system == "Darwin":
    plt.rcParams["font.sans-serif"] = ["PingFang TC", "Apple LiGothic", "sans-serif"]
else:
    _font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(_font_path):
        fm.fontManager.addfont(_font_path)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_font_path).get_name()
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Droid Sans Fallback", "sans-serif"]

plt.rcParams["axes.unicode_minus"] = False

# 預設圖片輸出目錄
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(_BASE_DIR, "static", "images")
os.makedirs(IMAGE_DIR, exist_ok=True)


def _save_path_for(filename: str) -> str:
    """產生儲存圖片的完整路徑。"""
    return os.path.join(IMAGE_DIR, filename)


def _plot_placeholder_chart(filename: str, title: str, message: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#E7E5E4')
    ax.set_facecolor('#E7E5E4')
    ax.text(
        0.5,
        0.55,
        title,
        ha="center",
        va="center",
        fontsize=16,
        color="#1E2938",
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.43,
        message,
        ha="center",
        va="center",
        fontsize=11,
        color="#64748B",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(_save_path_for(filename), dpi=150)
    plt.close(fig)
    return filename


def plot_price_history(history: pd.DataFrame, ticker_symbol: str, period: str = "1y") -> str:
    """
    畫出歷史收盤價折線圖並存為 PNG。

    Returns:
        儲存的檔案名稱（相對於 static/images/）
    """
    filename = f"{ticker_symbol.replace('.', '_')}_{period}_price.png"
    save_path = _save_path_for(filename)
    
    if os.path.exists(save_path):
        if time.time() - os.path.getmtime(save_path) < 3600:
            return filename

    if history.empty or "Close" not in history.columns:
        return _plot_placeholder_chart(
            filename,
            f"{ticker_symbol} 歷史股價走勢",
            "暫時無法取得即時股價資料，請稍後再試。",
        )

    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#E7E5E4')
    ax.set_facecolor('#E7E5E4')

    ax.plot(history.index, history["Close"], linewidth=2, color="#006666", label="收盤價")

    max_idx = history["Close"].idxmax()
    min_idx = history["Close"].idxmin()
    ax.scatter(max_idx, history["Close"][max_idx], color="#FF2157", zorder=5, s=60, label=f"最高 {history['Close'][max_idx]:.0f}")
    ax.scatter(min_idx, history["Close"][min_idx], color="#00A63D", zorder=5, s=60, label=f"最低 {history['Close'][min_idx]:.0f}")

    # 決定時間軸刻度間隔
    num_days = len(history)
    if num_days > 1000:
        locator = mdates.YearLocator()
    elif num_days > 500:
        locator = mdates.MonthLocator(interval=6)
    elif num_days > 200:
        locator = mdates.MonthLocator(interval=2)
    else:
        locator = mdates.MonthLocator(interval=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(locator)
    fig.autofmt_xdate()

    ax.set_title(f"{ticker_symbol} 歷史股價走勢", fontsize=14, color="#1E2938", weight="bold")
    ax.set_ylabel("股價（元）", color="#1E2938")
    ax.tick_params(colors="#1E2938")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c4c2c1")
    
    # Legend style
    legend = ax.legend(facecolor='#E7E5E4', edgecolor='#c4c2c1')
    for text in legend.get_texts():
        text.set_color('#1E2938')

    ax.grid(True, linestyle="--", alpha=0.6, color="#c4c2c1")

    plt.tight_layout()

    # 加上隨機數或期間識別，避免快取問題
    plt.savefig(_save_path_for(filename), dpi=150)
    plt.close(fig)
    return filename


def plot_pe_river_chart(pe_df: pd.DataFrame, ticker_symbol: str, current_pe: float = None, period: str = "3y") -> str:
    """
    繪製本益比河流圖 (P/E Band Chart)。
    以隨 EPS 波動的顏色區塊（河流）標示「偏低、合理、偏高」的歷史本益比區間。

    Args:
        pe_df:         由 fetcher.get_historical_pe() 取得的 DataFrame (含 Close, EPS, PE)
        ticker_symbol: 股票代號
        current_pe:    目前的本益比（選填）

    Returns:
        儲存的檔案名稱
    """
    filename = f"{ticker_symbol.replace('.', '_')}_{period}_pe_river.png"
    save_path = _save_path_for(filename)
    
    if os.path.exists(save_path):
        if time.time() - os.path.getmtime(save_path) < 3600:
            return filename

    if pe_df.empty or "EPS" not in pe_df.columns:
        return _plot_placeholder_chart(
            filename,
            f"{ticker_symbol} 本益比河流圖",
            "暫時無法取得估值資料，請稍後再試。",
        )

    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#E7E5E4')
    ax.set_facecolor('#E7E5E4')

    pe_values = pe_df["PE"].dropna()
    if pe_values.empty:
        return _plot_placeholder_chart(
            filename,
            f"{ticker_symbol} 本益比河流圖",
            "暫時無法計算本益比河流圖，請稍後再試。",
        )

    # 計算歷史本益比倍數的百分位數
    pe_bands = {
        "極度低估": np.percentile(pe_values, 10),
        "偏低":     np.percentile(pe_values, 25),
        "合理":     np.percentile(pe_values, 50),
        "偏高":     np.percentile(pe_values, 75),
        "極度高估": np.percentile(pe_values, 90),
    }

    # 去除時區以利繪圖對齊（以防萬一）
    if pe_df.index.tz is not None:
        pe_df.index = pe_df.index.tz_localize(None)

    # 計算每一天對應本益比區間的價格線
    band_keys = list(pe_bands.keys())
    band_curves = {}
    for key in band_keys:
        band_curves[key] = pe_bands[key] * pe_df["EPS"]

    # 繪製歷史收盤價走勢
    ax.plot(pe_df.index, pe_df["Close"], linewidth=2.5, color="#006666", label="收盤價", zorder=4)

    # 繪製本益比區間邊界線
    colors = ["#00A63D", "#8BC34A", "#FE9900", "#FF9800", "#FF2157"]
    for i, key in enumerate(band_keys):
        ax.plot(pe_df.index, band_curves[key], label=f"{key} (PE={pe_bands[key]:.1f})",
                color=colors[i], alpha=0.8, linewidth=1, linestyle="--")
        
        # 在最右側標記區間名稱
        ax.text(pe_df.index[-1], band_curves[key].iloc[-1], f" {key}",
                fontsize=8, va="center", color=colors[i], weight="bold")

    # 填充河流色塊
    fill_colors = ["#00A63D12", "#8BC34A12", "#FE990012", "#FF980012"]
    for i in range(len(band_keys) - 1):
        ax.fill_between(pe_df.index,
                        band_curves[band_keys[i]],
                        band_curves[band_keys[i + 1]],
                        color=fill_colors[i], zorder=1)

    # 決定時間軸刻度間隔
    num_days = len(pe_df)
    if num_days > 1000:
        locator = mdates.YearLocator()
    elif num_days > 500:
        locator = mdates.MonthLocator(interval=6)
    elif num_days > 200:
        locator = mdates.MonthLocator(interval=3)
    else:
        locator = mdates.MonthLocator(interval=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(locator)
    fig.autofmt_xdate()

    ax.set_title(f"{ticker_symbol} 本益比河流圖 (動態估值)", fontsize=14, color="#1E2938", weight="bold")
    ax.set_ylabel("股價（元）", color="#1E2938")
    ax.tick_params(colors="#1E2938")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c4c2c1")

    # Legend style
    legend = ax.legend(loc="upper left", facecolor='#E7E5E4', edgecolor='#c4c2c1', framealpha=0.9)
    for text in legend.get_texts():
        text.set_color('#1E2938')

    ax.grid(True, linestyle="--", alpha=0.4, color="#c4c2c1", zorder=2)

    plt.tight_layout()

    plt.savefig(_save_path_for(filename), dpi=150)
    plt.close(fig)
    return filename
