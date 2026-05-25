"""
app.py
Flask 後端伺服器 — 股票基本面分析工具 + 比較模式。
啟動後在 http://localhost:2330 提供網頁介面。
"""
import hashlib
import json
import os
import re
import pandas as pd
from flask import Flask, jsonify, request, render_template

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
    plot_normalized_comparison,
    plot_correlation_heatmap,
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
        name = get_tw_stock_name(symbol)
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
# API: 股票資料查詢
# ────────────────────────────────────────────
@app.route("/api/stock/<symbol>/info")
def api_stock_info(symbol):
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
# API: 新聞情緒分析（Claude AI）
# ────────────────────────────────────────────
@app.route("/api/stock/<symbol>/analyze_news", methods=["POST"])
def api_analyze_news(symbol):
    """接收前端送來的新聞清單，使用 Claude AI 進行情緒分析。"""
    import anthropic
    import json as _json

    data = request.json or {}
    news_list = data.get("news", [])

    if not news_list:
        return jsonify({"error": "目前沒有新聞可以分析"}), 400

    news_text = "\n".join(
        f"- [{item.get('date', '')}] {item.get('title', '')} （{item.get('source', '')}）"
        for item in news_list
    )

    prompt = f"""你是一位台股分析師。以下是 {symbol} 的最新相關新聞，請逐條判斷情緒並給出整體摘要。

新聞清單：
{news_text}

請以繁體中文回覆，並嚴格以 JSON 格式輸出，結構如下（不要加 markdown code block）：
{{
  "summary": "整體情緒摘要（2-3句，說明目前市場對此股的看法趨勢）",
  "details": [
    {{
      "title": "完整新聞標題",
      "sentiment_label": "正向 或 中立 或 負向",
      "sentiment_type": "positive 或 neutral 或 negative",
      "reason": "判斷理由（1句話）"
    }}
  ]
}}"""

    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text
        cleaned = re.sub(r"```json|```", "", raw).strip()
        result = _json.loads(cleaned)

        # 把原始 url/date/source 補回（Claude 只需判斷，不需重複輸出）
        url_map = {item["title"]: item for item in news_list}
        for detail in result.get("details", []):
            title = detail.get("title", "")
            if title in url_map:
                detail["url"] = url_map[title].get("url", "#")
                detail["date"] = url_map[title].get("date", "")
                detail["source"] = url_map[title].get("source", "")

        return jsonify(result)

    except Exception as e:
        print(f"[WARN] Claude API analyze_news failed ({symbol}): {e}")
        return jsonify({"error": f"AI 分析失敗：{str(e)}"}), 500


def _corr_strength(corr_val: float) -> str:
    """根據相關係數絕對值判斷強度。"""
    abs_corr = abs(corr_val)
    if abs_corr >= 0.7:
        return "極強"
    elif abs_corr >= 0.5:
        return "強"
    elif abs_corr >= 0.3:
        return "中等"
    else:
        return "弱"


# ────────────────────────────────────────────
# API: 多股比較分析
# ────────────────────────────────────────────
@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    多股比較分析 API。

    Request body:
        symbols:     股票代號清單（必須 2~6 支）
        period:      分析期間 ('6mo'、'1y'、'3y'、'5y')
        names:       symbol -> 中文名稱（optional）

    Returns:
        symbols:        成功分析的代號清單
        failed:         失敗的代號清單
        period:         分析期間
        data_days:      用於計算的交易日數
        correlation_matrix:  相關係數矩陣
        beta_matrix:    Beta 矩陣
        pairs:          配對詳細資料（上三角）
        performance:    各股期間績效（%）
        volatility:     各股年化波動率（%）
        normalized_chart:    標準化走勢圖 URL
        heatmap_chart:  相關係數熱力圖 URL
    """
    data = request.get_json() or {}
    raw_symbols = [s.strip().upper() for s in data.get("symbols", []) if s.strip()]
    period = data.get("period", "1y")
    names: dict[str, str] = data.get("names", {})

    if len(raw_symbols) < 2:
        return jsonify({"error": "請至少選擇 2 支股票進行比較"}), 400
    if len(raw_symbols) > 6:
        return jsonify({"error": "最多同時比較 6 支股票"}), 400

    # ── 抓取各股歷史收盤價 ──────────────────────
    price_series: dict[str, pd.Series] = {}
    failed: list[str] = []

    for sym in raw_symbols:
        try:
            hist = get_price_history(sym, period=period)
            if not hist.empty and "Close" in hist.columns:
                s = hist["Close"].dropna()
                if not s.empty:
                    price_series[sym] = s
                else:
                    failed.append(sym)
            else:
                failed.append(sym)
        except Exception as e:
            print(f"[WARN] compare fetch failed ({sym}): {e}")
            failed.append(sym)

    if len(price_series) < 2:
        return jsonify({"error": "無法取得足夠的股價資料（至少需要 2 支），請稍後再試"}), 500

    available_symbols = list(price_series.keys())

    # ── 對齊日期序列（內部交集，去除各股停市日的 NaN）──
    price_df = pd.DataFrame(price_series)
    price_df = price_df.dropna(how="any")   # 保留所有股票都有資料的交易日

    if len(price_df) < 10:
        # 交集太少改用 ffill 補值
        price_df = pd.DataFrame(price_series).fillna(method="ffill").dropna()

    # ── 計算日報酬率 ───────────────────────────
    returns_df = price_df.pct_change().dropna()

    # ── 相關係數矩陣 ──────────────────────────
    corr_matrix = returns_df.corr()

    # ── Beta 矩陣 ─────────────────────────────
    # beta(A, B) = cov(rA, rB) / var(rB)
    # 含義：當 B 上漲 1%，A 預計漲/跌 beta%
    cov_matrix = returns_df.cov()
    beta_matrix: dict[str, dict[str, float | None]] = {}
    for sym_a in available_symbols:
        beta_matrix[sym_a] = {}
        for sym_b in available_symbols:
            if sym_a == sym_b:
                beta_matrix[sym_a][sym_b] = 1.0
            else:
                var_b = cov_matrix.loc[sym_b, sym_b]
                if var_b and var_b != 0:
                    beta_matrix[sym_a][sym_b] = round(
                        float(cov_matrix.loc[sym_a, sym_b] / var_b), 4
                    )
                else:
                    beta_matrix[sym_a][sym_b] = None

    # ── 各股績效與波動率 ─────────────────────
    performance: dict[str, float | None] = {}
    volatility: dict[str, float | None] = {}

    for sym in available_symbols:
        series = price_df[sym]
        if len(series) >= 2:
            perf = (series.iloc[-1] / series.iloc[0] - 1) * 100
            performance[sym] = round(float(perf), 2)
        else:
            performance[sym] = None

        ret_std = returns_df[sym].std()
        if not pd.isna(ret_std):
            # 年化（假設 250 個交易日）
            volatility[sym] = round(float(ret_std * (250 ** 0.5) * 100), 2)
        else:
            volatility[sym] = None

    # ── 配對詳細資料（上三角）───────────────
    pairs: list[dict] = []
    for i, sym_a in enumerate(available_symbols):
        for j, sym_b in enumerate(available_symbols):
            if i >= j:
                continue
            corr_val = float(corr_matrix.loc[sym_a, sym_b])
            direction = "正相關" if corr_val > 0 else "負相關"
            pairs.append({
                "pair": [sym_a, sym_b],
                "name_a": names.get(sym_a, sym_a),
                "name_b": names.get(sym_b, sym_b),
                "correlation": round(corr_val, 4),
                "strength": _corr_strength(corr_val),
                "direction": direction,
                # Beta(A→B)：B 漲 1% 時 A 的預期漲幅
                "beta_a_on_b": beta_matrix[sym_a].get(sym_b),
                # Beta(B→A)：A 漲 1% 時 B 的預期漲幅
                "beta_b_on_a": beta_matrix[sym_b].get(sym_a),
            })

    # 依相關係數絕對值降序排列（最強關聯放最前面）
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

    # ── 產生圖表 ─────────────────────────────
    # 以 sorted 代號清單的 MD5 前 8 碼作為唯一 key，避免不同組合互相覆寫
    chart_key = hashlib.md5(
        ("_".join(sorted(available_symbols)) + period).encode()
    ).hexdigest()[:8]

    try:
        norm_filename = plot_normalized_comparison(
            price_df, available_symbols, names=names, chart_key=chart_key
        )
        norm_chart_url = f"/static/images/{norm_filename}"
    except Exception as e:
        print(f"[WARN] plot_normalized_comparison failed: {e}")
        norm_chart_url = None

    try:
        heatmap_filename = plot_correlation_heatmap(
            corr_matrix, names=names, chart_key=chart_key
        )
        heatmap_chart_url = f"/static/images/{heatmap_filename}"
    except Exception as e:
        print(f"[WARN] plot_correlation_heatmap failed: {e}")
        heatmap_chart_url = None

    return jsonify({
        "symbols": available_symbols,
        "failed": failed,
        "period": period,
        "data_days": len(price_df),
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "beta_matrix": beta_matrix,
        "pairs": pairs,
        "performance": performance,
        "volatility": volatility,
        "normalized_chart": norm_chart_url,
        "heatmap_chart": heatmap_chart_url,
    })


@app.route("/api/compare/interpret", methods=["POST"])
def api_compare_interpret():
    """
    接收比較分析結果，使用 Claude AI 產生自然語言解讀報告。

    Request body:
        symbols:     成功分析的代號清單
        names:       symbol -> 中文名稱
        pairs:       /api/compare 回傳的 pairs 陣列
        performance: 各股績效
        volatility:  各股波動率
        period:      分析期間
    """
    import anthropic
    import json as _json

    data = request.get_json() or {}
    symbols: list[str] = data.get("symbols", [])
    names: dict[str, str] = data.get("names", {})
    pairs: list[dict] = data.get("pairs", [])
    performance: dict = data.get("performance", {})
    volatility: dict = data.get("volatility", {})
    period: str = data.get("period", "1y")

    if not pairs:
        return jsonify({"error": "沒有相關性資料可供分析"}), 400

    # ── 組裝給 Claude 的結構化描述 ──────────
    period_label = {
        "6mo": "6 個月", "1y": "1 年", "3y": "3 年", "5y": "5 年"
    }.get(period, period)

    symbol_names = "、".join(f"{names.get(s, s)}（{s}）" for s in symbols)

    stock_stats = "\n".join(
        f"  - {names.get(s, s)}（{s}）：期間績效 {performance.get(s, 'N/A')}%，"
        f"年化波動率 {volatility.get(s, 'N/A')}%"
        for s in symbols
    )

    pairs_desc = "\n".join(
        f"  - {p['name_a']}（{p['pair'][0]}）× {p['name_b']}（{p['pair'][1]}）：\n"
        f"    相關係數 {p['correlation']:.3f}（{p['strength']}{p['direction']}），\n"
        f"    Beta({p['pair'][0]}對{p['pair'][1]}) = {p['beta_a_on_b']}，"
        f"Beta({p['pair'][1]}對{p['pair'][0]}) = {p['beta_b_on_a']}"
        for p in pairs
    )

    prompt = f"""你是一位台股資深分析師，擅長從量化數據中找出股票之間的產業關聯性與投資意涵。

以下是 {symbol_names} 在近 {period_label} 的量化分析結果：

【各股表現】
{stock_stats}

【配對相關性與 Beta】
{pairs_desc}

請以繁體中文撰寫一份分析報告，嚴格以 JSON 格式輸出（不要加 markdown code block），結構如下：
{{
  "summary": "整體摘要：這組股票的連動性概況與投資分散化效果（3-4 句）",
  "pair_analyses": [
    {{
      "pair_label": "股票A × 股票B",
      "interpretation": "相關係數與 Beta 的白話解讀，說明兩股的連動方向與強度",
      "reason": "從產業鏈、市場定位或總體經濟角度解釋為什麼會有這樣的關聯",
      "beta_insight": "Beta 的具體意涵，例如：當 B 上漲 1%，A 預計會如何反應"
    }}
  ],
  "portfolio_advice": "基於以上分析，對這組股票組合的風險分散化建議（2-3 句，含具體說明哪些組合可以互補）",
  "risk_warning": "主要風險提示（1-2 句，說明相關性在市場極端情況下可能失效的原因）"
}}"""

    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text
        cleaned = re.sub(r"```json|```", "", raw).strip()
        result = _json.loads(cleaned)
        return jsonify(result)

    except Exception as e:
        print(f"[WARN] Claude API compare_interpret failed: {e}")
        return jsonify({"error": f"AI 解讀失敗：{str(e)}"}), 500


# ────────────────────────────────────────────
# API: 多股新聞情緒分析
# ────────────────────────────────────────────
@app.route("/api/compare/analyze_news", methods=["POST"])
def api_compare_analyze_news():
    """
    對多支股票的新聞進行跨股票情緒分析。
    
    Request body:
        symbols:     股票代號清單
        names:       symbol -> 中文名稱（optional）
        news:        新聞清單 [{title, date, source, url}]
    
    Returns:
        [{
            "news_title": "新聞標題",
            "date": "日期",
            "source": "來源",
            "url": "連結",
            "stocks": [
                {
                    "symbol": "2330",
                    "name": "台積電",
                    "sentiment": "positive",
                    "score": 0.82,
                    "reason": "..."
                }
            ]
        }]
    """
    import anthropic
    import json as _json

    data = request.get_json() or {}
    symbols: list[str] = [s.upper() for s in data.get("symbols", [])]
    names: dict[str, str] = data.get("names", {}) or {}
    news_list: list[dict] = data.get("news", [])

    if not symbols or len(symbols) < 2:
        return jsonify({"error": "請至少選擇 2 支股票"}), 400
    
    if not news_list:
        return jsonify({"error": "沒有新聞可分析"}), 400

    # ── 組織符號與名稱 ──────────────────────
    symbol_names = "、".join(f"{names.get(s, s)}（{s}）" for s in symbols)
    
    # ── 組織新聞列表 ────────────────────────
    news_text = "\n".join(
        f"- [{item.get('date', '')}] {item.get('title', '')} （{item.get('source', '')}）"
        for item in news_list
    )

    prompt = f"""你是一位台股資深分析師。以下是一份新聞，以及我們要分析的股票清單：{symbol_names}。

請逐條評估此新聞對於清單內各支股票的影響程度與方向（正面、中立、負面）。
考慮：供應鏈關聯、競爭態勢、產業政策、技術變化、市場風險等因素。

新聞內容：
{news_text}

請以繁體中文回覆，並嚴格以 JSON 格式輸出（不要加 markdown code block），結構如下：
{{
  "analyses": [
    {{
      "symbol": "2330",
      "name": "台積電",
      "sentiment": "positive",
      "score": 0.82,
      "reason": "AI GPU 晶片代工需求增加"
    }}
  ]
}}

sentiment 只能是 "positive"、"neutral" 或 "negative"。
score 是 0~1 之間的浮點數，表示信心度。
"""

    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = message.content[0].text
        cleaned = re.sub(r"```json|```", "", raw).strip()
        result = _json.loads(cleaned)
        
        # 補充原始新聞資料
        analyses_by_symbol = {
            a["symbol"]: a for a in result.get("analyses", [])
        }
        
        response_news = []
        for news_item in news_list:
            response_item = {
                "news_title": news_item.get("title", ""),
                "date": news_item.get("date", ""),
                "source": news_item.get("source", ""),
                "url": news_item.get("url", "#"),
                "stocks": []
            }
            
            for sym in symbols:
                if sym in analyses_by_symbol:
                    analysis = analyses_by_symbol[sym]
                    response_item["stocks"].append({
                        "symbol": sym,
                        "name": names.get(sym, sym),
                        "sentiment": analysis.get("sentiment", "neutral"),
                        "score": analysis.get("score", 0.5),
                        "reason": analysis.get("reason", "")
                    })
                else:
                    response_item["stocks"].append({
                        "symbol": sym,
                        "name": names.get(sym, sym),
                        "sentiment": "neutral",
                        "score": 0.5,
                        "reason": "暫無評估"
                    })
            
            response_news.append(response_item)
        
        return jsonify(response_news)

    except Exception as e:
        print(f"[WARN] Claude API compare_analyze_news failed: {e}")
        return jsonify({"error": f"AI 分析失敗：{str(e)}"}), 500


# ────────────────────────────────────────────
# API: 健康檢查
# ────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    from datetime import datetime
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


# ────────────────────────────────────────────
# 啟動伺服器
# ────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=2330, debug=True)
