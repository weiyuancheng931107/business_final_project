import os
import json
import time
import requests
import pandas as pd
from datetime import datetime

CACHE_FILE = "stock_list_cache.json"
CACHE_EXPIRE_SECONDS = 24 * 60 * 60  # 24 hours

def fetch_stock_list():
    """
    從 TWSE 爬取股票代號與名稱，並實作 24 小時緩存機制。
    """
    # 檢查緩存是否存在且未過期
    if os.path.exists(CACHE_FILE):
        file_time = os.path.getmtime(CACHE_FILE)
        if (time.time() - file_time) < CACHE_EXPIRE_SECONDS:
            print("讀取本地股票清單緩存...")
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)

    print("從 TWSE 爬取最新股票清單...")
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    
    try:
        # 使用 pandas 直接讀取表格
        response = requests.get(url)
        response.encoding = 'big5' # TWSE 網頁通常是 big5
        
        # read_html 回傳的是 list of DataFrames
        dfs = pd.read_html(response.text)
        df = dfs[0]
        
        # 設定欄位名稱（第一列通常是標題）
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        
        stock_list = []
        
        # 我們只需要「股票」這一個區段，通常在「有價證券代號及名稱」欄位
        # 格式為 "2330  台積電"
        for index, row in df.iterrows():
            item = str(row[0])
            # 股票區段通常在「備註」欄位為空白且前面是四位數代號
            # 這裡簡單判斷：只要能被拆分成 "代號 名稱" 且代號是數字
            parts = item.split('\u3000') # 全形空格
            if len(parts) < 2:
                parts = item.split(' ') # 半形空格
                
            if len(parts) >= 2:
                symbol = parts[0].strip()
                name = parts[1].strip()
                
                # 過濾出純數字的股票代號 (主要為一般股票)
                if symbol.isdigit() and len(symbol) == 4:
                    stock_list.append({
                        "symbol": symbol,
                        "name": name,
                        "display": f"{symbol} {name}"
                    })
            
            # 碰到「債券」或「認購(售)權證」就停止，這可以縮小搜尋範圍
            if "債券" in item or "權證" in item:
                break
                
        # 存入緩存
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(stock_list, f, ensure_ascii=False, indent=2)
            
        return stock_list
        
    except Exception as e:
        print(f"爬取股票清單失敗: {e}")
        # 如果爬取失敗但有舊緩存，就勉其量回傳舊的
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

if __name__ == "__main__":
    # 測試執行
    stocks = fetch_stock_list()
    print(f"成功抓取 {len(stocks)} 檔股票。")
    if stocks:
        print(f"範例: {stocks[0]}")
