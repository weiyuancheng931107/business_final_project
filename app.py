"""
app.py
Flask 後端伺服器 — 股票基本面分析工具。
啟動後在 http://localhost:2330 提供網頁介面。
"""
import json
import os
import pandas as pd
from flask import Flask, jsonify, request, render_template

from openai import OpenAI
import config

from analysis.fetcher import (
    get_stock_info,
    get_price_history,
    get_historical_pe,
    get_tw_stock_name,
    get_stock_news,
)
from analysis.plotter import (
    plot_price_history,
    plot_pe_river_chart,
)

# 初始化 OpenAI / OpenRouter 客戶端
client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_API_BASE
)

app = Flask(__name__)

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio.json")


# ────────────────────────────────────────────
# 持久化工具函式
# ────────────────────────────────────────────
def _load_portfolio() -> list[dict]:
    """從 JSON 檔案讀取自選股清單。若為舊版字串陣列，則自動升級中文化。"""
    if not os.path.exists(PORTFOLIO_FILE):
        return []
    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        stocks = data.get("stocks", [])

        # 舊版格式相容移轉 (Migration: list[str] -> list[dict])
        if isinstance(stocks, list) and len(stocks) > 0 and isinstance(stocks[0], str):
            migrated_stocks = []
            for s in stocks:
                name = get_tw_stock_name(s)
                migrated_stocks.append({"symbol": s, "name": name})
            _save_portfolio(migrated_stocks)
            return migrated_stocks

        return stocks
    except Exception as e:
        print(f"⚠️ 載入 portfolio 發生錯誤: {e}")
        return []


def _save_portfolio(stocks: list[dict]):
    """將自選股清單寫入 JSON 檔案。"""
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump({"stocks": stocks}, f, ensure_ascii=False, indent=2)


# ────────────────────────────────────────────
# 頁面路由
# ────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ────────────────────────────────────────────
# API: 自選股管理
# ────────────────────────────────────────────
@app.route("/api/portfolio", methods=["GET"])
def api_get_portfolio():
    """取得目前的自選股清單。"""
    stocks = _load_portfolio()
    return jsonify({"stocks": stocks})


@app.route("/api/portfolio", methods=["POST"])
def api_add_stock():
    """新增一檔股票到清單，並自動進行中文名稱解析。"""
    data = request.get_json()
    symbol = data.get("symbol", "").strip().upper()

    if not symbol:
        return jsonify({"error": "請提供股票代號"}), 400

    # 自動補上 .TW 後綴（如果使用者只輸入數字）
    if symbol.isdigit():
        symbol = symbol + ".TW"

    stocks = _load_portfolio()
    if any(s["symbol"] == symbol for s in stocks):
        return jsonify({"error": f"{symbol} 已在清單中"}), 409

    try:
        # 自動抓取中文簡稱
        name = get_tw_stock_name(symbol)
        
        # 測試是否能正常查詢
        info = get_stock_info(symbol)
        if not info.get("current_price"):
            return jsonify({"error": f"找不到 {symbol} 的股價資料，請檢查代號是否正確"}), 404

        new_stock = {"symbol": symbol, "name": name}
        stocks.append(new_stock)
        _save_portfolio(stocks)
        return jsonify({"stocks": stocks, "added": symbol, "name": name})
    except Exception as e:
        return jsonify({"error": f"新增失敗: {str(e)}"}), 500


@app.route("/api/portfolio/<symbol>", methods=["DELETE"])
def api_remove_stock(symbol):
    """從清單移除一檔股票。"""
    symbol = symbol.upper()
    stocks = _load_portfolio()
    
    target = None
    for s in stocks:
        if s["symbol"] == symbol:
            target = s
            break

    if not target:
        return jsonify({"error": f"{symbol} 不在清單中"}), 404

    stocks.remove(target)
    _save_portfolio(stocks)
    return jsonify({"stocks": stocks, "removed": symbol})


# ────────────────────────────────────────────
# API: 股票資料查詢 (含健康評分與估值建議)
# ────────────────────────────────────────────
@app.route("/api/stock/<symbol>/info")
def api_stock_info(symbol):
    """取得單一股票的基本面資訊。"""
    try:
        info = get_stock_info(symbol)
        return jsonify(info)
    except Exception as e:
        print(f"[WARN] stock info endpoint fallback ({symbol}): {e}")
        return jsonify({
            "symbol": symbol,
            "name": symbol,
            "currency": "N/A",
            "current_price": None,
            "pe_ratio": None,
            "roe": None,
            "eps": None,
            "warning": "資料來源暫時無法取得，已顯示降級資訊。",
        })


@app.route("/api/stock/<symbol>/price_chart")
def api_price_chart(symbol):
    """取得歷史股價走勢圖。"""
    period = request.args.get("period", "1y")
    try:
        history = get_price_history(symbol, period=period)
        filename = plot_price_history(history, symbol)
        return jsonify({"image": f"/static/images/{filename}"})
    except Exception as e:
        print(f"[WARN] price chart endpoint fallback ({symbol}, {period}): {e}")
        filename = plot_price_history(pd.DataFrame(), symbol)
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
        filename = plot_pe_river_chart(pe_df, symbol)
        return jsonify({"image": f"/static/images/{filename}"})
    except Exception as e:
        print(f"[WARN] PE river endpoint fallback ({symbol}, {period}): {e}")
        filename = plot_pe_river_chart(pd.DataFrame(), symbol)
        return jsonify({
            "image": f"/static/images/{filename}",
            "warning": "估值資料暫時無法取得，已顯示備援圖表。",
        })

# ────────────────────────────────────────────
# API: 相關新聞 
# ────────────────────────────────────────────
@app.route("/api/stock/<symbol>/news")
def api_stock_news(symbol):
    """取得股票相關新聞。支持分頁加載。"""
    start_date = request.args.get("start_date")
    try:
        news_data = get_stock_news(symbol, start_date)
        return jsonify({
            "symbol": symbol,
            "news": news_data["news"],
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
    接收前端送來的新聞清單，使用 Gemini 進行真實情緒分析。
    """
    data = request.json or {}
    news_list = data.get("news", [])
    
    analyzed_news = []
    
    # 針對每一篇新聞，呼叫 AI 判斷情緒
    for item in news_list:
        title = item.get("title", "未命名標題")
        url = item.get("url", "#")
        source = item.get("source", "未知")
        date = item.get("date", "")
        
        # 預設值（萬一 AI 判斷失敗或異常，至少保持中立，不要讓網頁當掉）
        sentiment_label = "中立"
        sentiment_type = "neutral"
        
        if title != "未命名標題":
            prompt = f"""
            請以財經分析師的角度，判斷這則新聞標題的情緒。
            你只能回答「正向」、「負向」或「中立」其中一個詞，不要任何其他解釋。
            新聞標題：{title}
            """
            try:
                # 呼叫 OpenAI / OpenRouter 相容的 API 模型
                response = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10,
                    temperature=0.0
                )
                ai_result = response.choices[0].message.content.strip()
                
                # 將 AI 的中文回答對應到前端需要的 CSS 顏色樣式 (type)
                if "正向" in ai_result:
                    sentiment_label = "正向"
                    sentiment_type = "positive"
                elif "負向" in ai_result:
                    sentiment_label = "負向"
                    sentiment_type = "negative"
                else:
                    sentiment_label = "中立"
                    sentiment_type = "neutral"
                    
            except Exception as e:
                print(f"⚠️ AI 分析 {title} 時發生錯誤: {e}")
                pass 
                
        analyzed_news.append({
            "title": title,
            "url": url,
            "source": source,
            "date": date,
            "sentiment_label": sentiment_label,
            "sentiment_type": sentiment_type
        })

    summary = f"已完成 {symbol} 最新 {len(news_list)} 篇新聞的真實 AI 情緒分析。"
    
    report = {
        "summary": summary,
        "details": analyzed_news
    }
    
    return jsonify(report)


# ────────────────────────────────────────────
# 啟動伺服器
# ────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=2330, debug=True)
