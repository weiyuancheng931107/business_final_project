"""
app.py
Flask 後端伺服器 — 股票基本面分析工具。
啟動後在 http://localhost:2330 提供網頁介面。
"""
import json
import os
import re
import time
from datetime import datetime
import concurrent.futures
import pandas as pd
from flask import Flask, jsonify, request, render_template

import config

from analysis.fetcher import (
    get_stock_info,
    get_price_history,
    get_historical_pe,
    get_tw_stock_name,
    get_stock_news,
    fetch_article_content,
)
from analysis.plotter import (
    plot_price_history,
    plot_pe_river_chart,
)
from utils.twse_fetcher import fetch_stock_list

app = Flask(__name__)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "static", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ────────────────────────────────────────────
# 頁面路由
# ────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/get_stock_list")
def api_get_stock_list():
    """取得台股代號與名稱清單（用於前端搜尋）。"""
    stocks = fetch_stock_list()
    return jsonify(stocks)


# ────────────────────────────────────────────
# API: 股票資料查詢 (含健康評分與估值建議)
# ────────────────────────────────────────────
@app.route("/api/stock/<symbol>/info")
def api_stock_info(symbol):
    """取得單一股票的基本面資訊。"""
    try:
        info = get_stock_info(symbol)
        report_file = os.path.join(REPORTS_DIR, f"{symbol.upper()}_analysis.json")
        info["has_saved_analysis"] = os.path.exists(report_file)
        return jsonify(info)
    except Exception as e:
        print(f"[WARN] stock info endpoint fallback ({symbol}): {e}")
        report_file = os.path.join(REPORTS_DIR, f"{symbol.upper()}_analysis.json")
        return jsonify({
            "symbol": symbol,
            "name": symbol,
            "currency": "N/A",
            "current_price": None,
            "pe_ratio": None,
            "roe": None,
            "eps": None,
            "has_saved_analysis": os.path.exists(report_file),
            "warning": "資料來源暫時無法取得，已顯示降級資訊。",
        })


@app.route("/api/stock/<symbol>/price_chart")
def api_price_chart(symbol):
    """取得歷史股價走勢圖。"""
    period = request.args.get("period", "1y")
    try:
        history = get_price_history(symbol, period=period)
        filename = plot_price_history(history, symbol, period=period)
        return jsonify({"image": f"/static/images/{filename}"})
    except Exception as e:
        print(f"[WARN] price chart endpoint fallback ({symbol}, {period}): {e}")
        filename = plot_price_history(pd.DataFrame(), symbol, period=period)
        return jsonify({
            "image": f"/static/images/{filename}",
            "warning": "股價資料暫時無法取得，已顯示備援圖表。",
        })


@app.route("/api/stock/<symbol>/pe_river")
def api_pe_river(symbol):
    """取得本益比河流圖。"""
    period = request.args.get("period", "3y")
    try:
        pe_df = get_historical_pe(symbol, period=period)
        filename = plot_pe_river_chart(pe_df, symbol, period=period)
        return jsonify({"image": f"/static/images/{filename}"})
    except Exception as e:
        print(f"[WARN] PE river endpoint fallback ({symbol}, {period}): {e}")
        filename = plot_pe_river_chart(pd.DataFrame(), symbol, period=period)
        return jsonify({
            "image": f"/static/images/{filename}",
            "warning": "估值資料暫時無法取得，已顯示備援圖表。",
        })

# ────────────────────────────────────────────
# API: 相關新聞 
# ────────────────────────────────────────────
@app.route("/api/stock/<symbol>/news")
def api_stock_news(symbol):
    """取得股票相關新聞。支持分頁加載，並預先抓取內文摘要。"""
    start_date = request.args.get("start_date")
    try:
        news_data = get_stock_news(symbol, start_date)
        
        # 並行抓取內文摘要
        def enrich_news(item):
            content = fetch_article_content(item.get("url", ""))
            item["content_summary"] = content
            return item
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            enriched_news = list(executor.map(enrich_news, news_data["news"]))
            
        return jsonify({
            "symbol": symbol,
            "news": enriched_news,
            "next_start_date": news_data["next_start_date"],
            "has_more": news_data["has_more"],
            "note": ""
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────
# API: 新聞情緒分析 (真實 AI 版本)
# ────────────────────────────────────────────
@app.route("/api/stock/<symbol>/analyze_news", methods=["POST"])
def api_analyze_news(symbol):
    """
    接收前端傳來已包含內文摘要的新聞清單，交由 LLM 產生結構化 JSON 情緒分析報告。
    """
    data = request.json or {}
    news_list = data.get("news", [])
    
    if not news_list:
        return jsonify({"error": "No news provided"}), 400

    from analysis.ai_analyzer import analyze_stock_news
    
    try:
        report = analyze_stock_news(symbol, news_list)
        
        # Save analysis report to file
        try:
            report_file = os.path.join(REPORTS_DIR, f"{symbol.upper()}_analysis.json")
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as save_err:
            print(f"[WARN] Failed to save analysis report for {symbol}: {save_err}")
            
        return jsonify(report)
        
    except Exception as e:
        print(f"⚠️ AI 分析模組發生未知錯誤: {e}")
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────
# 啟動伺服器
# ────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2330, debug=True)
