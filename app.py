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

from openai import OpenAI
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

# 初始化 OpenAI / OpenRouter 客戶端
client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_API_BASE
)

app = Flask(__name__)

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio.json")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "static", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


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
    start_time = time.time()
    data = request.json or {}
    news_list = data.get("news", [])
    
    if not news_list:
        return jsonify({"error": "No news provided"}), 400

    # 整理前端傳來的新聞資料
    enriched_news = []
    for i, item in enumerate(news_list):
        enriched_news.append({
            "index": i,
            "title": item.get("title", "未命名標題"),
            "url": item.get("url", "#"),
            "source": item.get("source", "未知"),
            "date": item.get("date", ""),
            "content": item.get("content_summary", "")
        })

    # 2. 準備給 LLM 的 Prompt
    news_text_blocks = []
    for item in enriched_news:
        news_text_blocks.append(
            f"[{item['index']}] 標題: {item['title']} | 來源: {item['source']} | 內文摘錄: {item['content'][:300] if item['content'] else '無'}"
        )
    
    prompt = f"""
請以資深財經分析師的角度，對以下 {symbol} 相關的新聞進行情緒與影響力分析。
請輸出嚴格的 JSON 格式（不要加上 markdown 的 ```json 標記，直接輸出 JSON 物件），結構如下：

{{
  "overall": {{
    "sentiment_label": "整體情緒標籤（例如：偏向樂觀、中立、強烈悲觀等）",
    "sentiment_type": "positive 或 neutral 或 negative",
    "score": 數值 (範圍 -5.0 到 5.0，表示整體情緒，保留一位小數),
    "summary": "以財經專家口吻撰寫的一段整體分析與後市看法，約 50-80 字",
    "keywords": ["關鍵字1", "關鍵字2", "關鍵字3"]
  }},
  "details": [
    {{
      "index": 數字 (對應下方新聞的索引 [0], [1]...),
      "sentiment_label": "單則情緒（例如：極度利多、輕微偏空等）",
      "sentiment_type": "positive 或 neutral 或 negative",
      "impact_score": 數字 (單則影響力分數，範圍 -5 到 5 的整數),
      "reason": "一兩句話說明這則新聞對該公司或股價的具體影響與原因"
    }}
  ]
}}

以下是 {symbol} 的近期新聞：
""" + "\n\n".join(news_text_blocks)

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant that outputs raw, valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        if not response or not hasattr(response, 'choices') or not response.choices:
            raise ValueError("LLM 沒有回傳有效的 choices，可能為 API 限制或服務異常。")
            
        ai_result = response.choices[0].message.content
        if not ai_result:
            raise ValueError("LLM 回傳內容為空。")
            
        # 3. 處理並防呆 JSON 解析
        # 移除可能的 markdown json codeblock
        ai_result = ai_result.strip()
        ai_result = re.sub(r"^```(?:json)?\s*", "", ai_result)
        ai_result = re.sub(r"\s*```$", "", ai_result).strip()
        
        try:
            parsed_result = json.loads(ai_result)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM 回傳非有效 JSON 格式: {str(e)}")
            
        if not isinstance(parsed_result, dict):
            if isinstance(parsed_result, list) and len(parsed_result) > 0 and isinstance(parsed_result[0], dict):
                parsed_result = {"details": parsed_result, "overall": {}}
            else:
                parsed_result = {}
        
        # 4. 把原始 URL 和 Meta data 合併回 details
        details_list = parsed_result.get("details", [])
        if not isinstance(details_list, list):
            details_list = []
            
        details_map = {}
        for d in details_list:
            if isinstance(d, dict) and "index" in d:
                details_map[d["index"]] = d
        
        final_details = []
        for item in enriched_news:
            idx = item["index"]
            d = details_map.get(idx, {})
            if not isinstance(d, dict):
                d = {}
                
            final_details.append({
                "title": item.get("title", "未命名"),
                "url": item.get("url", "#"),
                "source": item.get("source", "未知"),
                "date": item.get("date", ""),
                "content_summary": item.get("content", ""),
                "sentiment_label": d.get("sentiment_label", "中立"),
                "sentiment_type": d.get("sentiment_type", "neutral"),
                "impact_score": d.get("impact_score", 0),
                "reason": d.get("reason", "無特定分析")
            })
            
        end_time = time.time()
        
        overall_data = parsed_result.get("overall", {})
        if not isinstance(overall_data, dict):
            overall_data = {}
            
        report = {
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_duration_seconds": round(end_time - start_time, 2),
            "overall": {
                "sentiment_label": overall_data.get("sentiment_label", "中立"),
                "sentiment_type": overall_data.get("sentiment_type", "neutral"),
                "score": overall_data.get("score", 0.0),
                "summary": overall_data.get("summary", "分析完成。"),
                "keywords": overall_data.get("keywords", [])
            },
            "details": final_details
        }
        
        # Save analysis report to file
        try:
            report_file = os.path.join(REPORTS_DIR, f"{symbol.upper()}_analysis.json")
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as save_err:
            print(f"[WARN] Failed to save analysis report for {symbol}: {save_err}")
            
        return jsonify(report)
        
    except Exception as e:
        print(f"⚠️ AI 分析時發生錯誤: {e}")
        end_time = time.time()
        # Fallback 回應
        fallback_details = []
        for item in enriched_news:
            fallback_details.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": item.get("source", ""),
                "date": item.get("date", ""),
                "content_summary": item.get("content", ""),
                "sentiment_label": "分析失敗",
                "sentiment_type": "neutral",
                "impact_score": 0,
                "reason": f"無法分析，錯誤: {str(e)[:50]}"
            })
        return jsonify({
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_duration_seconds": round(end_time - start_time, 2),
            "overall": {
                "sentiment_label": "系統錯誤",
                "sentiment_type": "negative",
                "score": 0.0,
                "summary": "AI 分析服務暫時無法使用，或回傳格式異常。",
                "keywords": ["錯誤", "服務異常"]
            },
            "details": fallback_details
        })


# ────────────────────────────────────────────
# 啟動伺服器
# ────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=2330, debug=True)
