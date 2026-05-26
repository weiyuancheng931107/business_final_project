import time
import json
import re
from datetime import datetime
from openai import OpenAI
import config

# 初始化 OpenAI / OpenRouter 客戶端
client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_API_BASE
)

def analyze_stock_news(symbol: str, news_list: list) -> dict:
    """
    接收前端傳來已包含內文摘要的新聞清單，交由 LLM 產生結構化 JSON 情緒分析報告。
    """
    start_time = time.time()
    
    if not news_list:
        raise ValueError("No news provided")

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
            f"[{item['index']}] 標題: {item['title']} | 來源: {item['source']} | 內文摘錄: {item['content'][:2500] if item['content'] else '無'}"
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
            
        return {
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
        return {
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
        }
