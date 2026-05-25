import hashlib
import json
import os
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

from config import client, GPT_MODEL

app = Flask(__name__)

PORTFOLIO_FILE = os.path.join(
    os.path.dirname(__file__),
    "portfolio.json"
)

# =========================================================
# Portfolio
# =========================================================

def _load_portfolio():

    if not os.path.exists(PORTFOLIO_FILE):
        return []

    try:

        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:

            data = json.load(f)

        stocks = data.get("stocks", [])

        # 舊格式相容
        if (
            isinstance(stocks, list)
            and len(stocks) > 0
            and isinstance(stocks[0], str)
        ):

            migrated = []

            for s in stocks:

                migrated.append({
                    "symbol": s,
                    "name": get_tw_stock_name(s)
                })

            _save_portfolio(migrated)

            return migrated

        return stocks

    except Exception as e:

        print(e)

        return []


def _save_portfolio(stocks):

    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:

        json.dump(
            {"stocks": stocks},
            f,
            ensure_ascii=False,
            indent=2
        )

# =========================================================
# Main Page
# =========================================================

@app.route("/")
def index():

    return render_template("index.html")

# =========================================================
# Portfolio API
# =========================================================

@app.route("/api/portfolio")
def api_get_portfolio():

    return jsonify({
        "stocks": _load_portfolio()
    })


@app.route("/api/portfolio", methods=["POST"])
def api_add_stock():

    data = request.get_json()

    symbol = data.get("symbol", "").strip().upper()

    if not symbol:

        return jsonify({
            "error": "請輸入股票代號"
        }), 400

    if symbol.isdigit():

        symbol += ".TW"

    stocks = _load_portfolio()

    if any(s["symbol"] == symbol for s in stocks):

        return jsonify({
            "error": "股票已存在"
        }), 400

    try:

        name = get_tw_stock_name(symbol)

        info = get_stock_info(symbol)

        if not info.get("current_price"):

            return jsonify({
                "error": "找不到股票資料"
            }), 404

        new_stock = {
            "symbol": symbol,
            "name": name
        }

        stocks.append(new_stock)

        _save_portfolio(stocks)

        return jsonify({
            "added": symbol,
            "name": name,
            "stocks": stocks
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/portfolio/<symbol>", methods=["DELETE"])
def api_remove_stock(symbol):

    symbol = symbol.upper()

    stocks = _load_portfolio()

    stocks = [
        s for s in stocks
        if s["symbol"] != symbol
    ]

    _save_portfolio(stocks)

    return jsonify({
        "removed": symbol
    })

# =========================================================
# Stock Info
# =========================================================

@app.route("/api/stock/<symbol>/info")
def api_stock_info(symbol):

    try:

        info = get_stock_info(symbol)

        return jsonify(info)

    except Exception as e:

        print(e)

        return jsonify({
            "symbol": symbol,
            "name": symbol,
            "current_price": None,
            "pe_ratio": None,
            "roe": None,
            "eps": None,
            "warning": "資料取得失敗"
        })

# =========================================================
# Price Chart
# =========================================================

@app.route("/api/stock/<symbol>/price_chart")
def api_price_chart(symbol):

    period = request.args.get("period", "1y")

    try:

        history = get_price_history(symbol, period)

        filename = plot_price_history(
            history,
            symbol
        )

        return jsonify({
            "image": f"/static/images/{filename}"
        })

    except Exception as e:

        print(e)

        return jsonify({
            "error": str(e)
        }), 500

# =========================================================
# PE River
# =========================================================

@app.route("/api/stock/<symbol>/pe_river")
def api_pe_river(symbol):

    period = request.args.get("period", "3y")

    try:

        pe_df = get_historical_pe(
            symbol,
            period
        )

        filename = plot_pe_river_chart(
            pe_df,
            symbol
        )

        return jsonify({
            "image": f"/static/images/{filename}"
        })

    except Exception as e:

        print(e)

        return jsonify({
            "error": str(e)
        }), 500

# =========================================================
# News
# =========================================================

@app.route("/api/stock/<symbol>/news")
def api_stock_news(symbol):

    start_date = request.args.get("start_date")

    try:

        news_data = get_stock_news(
            symbol,
            start_date
        )

        return jsonify({
            "symbol": symbol,
            "news": news_data["news"],
            "next_start_date": news_data["next_start_date"],
            "has_more": news_data["has_more"]
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# =========================================================
# Single Stock AI Analysis
# =========================================================

@app.route("/api/stock/<symbol>/analyze_news", methods=["POST"])
def api_analyze_news(symbol):

    data = request.json or {}

    news_list = data.get("news", [])

    if not news_list:

        return jsonify({
            "error": "沒有新聞"
        }), 400

    news_text = "\n".join([
        f"- {n['title']}"
        for n in news_list
    ])

    prompt = f"""
你是一位台股分析師。

以下是 {symbol} 的新聞：

{news_text}

請分析每篇新聞情緒。

請只輸出 JSON：

{{
  "summary":"...",
  "details":[
    {{
      "title":"...",
      "sentiment_label":"正向",
      "sentiment_type":"positive",
      "reason":"..."
    }}
  ]
}}
"""

    try:

        response = client.chat.completions.create(

            model=GPT_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3
        )

        raw = response.choices[0].message.content

        raw = raw.replace("```json", "").replace("```", "")

        result = json.loads(raw)

        url_map = {
            n["title"]: n
            for n in news_list
        }

        for d in result["details"]:

            title = d["title"]

            if title in url_map:

                d["url"] = url_map[title].get("url", "#")
                d["date"] = url_map[title].get("date", "")
                d["source"] = url_map[title].get("source", "")

        return jsonify(result)

    except Exception as e:

        print(e)

        return jsonify({
            "error": str(e)
        }), 500

# =========================================================
# Compare API
# =========================================================

@app.route("/api/compare", methods=["POST"])
def api_compare():

    data = request.get_json()

    symbols = data.get("symbols", [])

    period = data.get("period", "1y")

    names = data.get("names", {})

    if len(symbols) < 2:

        return jsonify({
            "error": "至少需要兩支股票"
        }), 400

    price_series = {}

    for sym in symbols:

        try:

            hist = get_price_history(
                sym,
                period
            )

            if not hist.empty:

                price_series[sym] = hist["Close"]

        except Exception as e:

            print(e)

    if len(price_series) < 2:

        return jsonify({
            "error": "無法取得足夠資料"
        }), 500

    price_df = pd.DataFrame(price_series)

    price_df = price_df.dropna()

    returns_df = price_df.pct_change().dropna()

    corr_matrix = returns_df.corr()

    performance = {}

    volatility = {}

    for sym in symbols:

        if sym not in price_df.columns:
            continue

        series = price_df[sym]

        performance[sym] = round(
            (
                series.iloc[-1]
                / series.iloc[0]
                - 1
            ) * 100,
            2
        )

        volatility[sym] = round(
            returns_df[sym].std()
            * (250 ** 0.5)
            * 100,
            2
        )

    chart_key = hashlib.md5(
        (
            "_".join(symbols)
            + period
        ).encode()
    ).hexdigest()[:8]

    norm_chart = plot_normalized_comparison(
        price_df,
        symbols,
        names=names,
        chart_key=chart_key
    )

    heatmap_chart = plot_correlation_heatmap(
        corr_matrix,
        names=names,
        chart_key=chart_key
    )

    return jsonify({
        "symbols": symbols,
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "performance": performance,
        "volatility": volatility,
        "normalized_chart": f"/static/images/{norm_chart}",
        "heatmap_chart": f"/static/images/{heatmap_chart}"
    })

# =========================================================
# Multi-stock News Analysis
# =========================================================

@app.route("/api/compare/analyze_news", methods=["POST"])
def api_compare_analyze_news():

    data = request.get_json()

    symbols = data.get("symbols", [])

    news = data.get("news", [])

    names = data.get("names", {})

    if not symbols or not news:

        return jsonify({
            "error": "缺少資料"
        }), 400

    news_text = "\n".join([
        f"- {n['title']}"
        for n in news
    ])

    prompt = f"""
你是一位台股分析師。

以下新聞：

{news_text}

請分析這些新聞對：

{",".join(symbols)}

各自的影響。

請輸出 JSON：

[
  {{
    "news_title":"...",
    "stocks":[
      {{
        "symbol":"2330",
        "sentiment":"positive",
        "score":0.82,
        "reason":"..."
      }}
    ]
  }}
]
"""

    try:

        response = client.chat.completions.create(

            model=GPT_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3
        )

        raw = response.choices[0].message.content

        raw = raw.replace("```json", "").replace("```", "")

        result = json.loads(raw)

        return jsonify(result)

    except Exception as e:

        print(e)

        return jsonify({
            "error": str(e)
        }), 500

# =========================================================
# Health
# =========================================================

@app.route("/api/health")
def api_health():

    return jsonify({
        "status": "ok"
    })

# =========================================================
# Run
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=2330,
        debug=True
    )
