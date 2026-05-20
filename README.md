# 商管程式設計期末專案 - 股票基本面分析

這是一個以 Python 實作的台股基本面分析工具。
透過網頁介面管理自選股、查看歷史股價走勢與本益比河流圖。

## 專案結構

```
business_final_project/
├── analysis/           # 核心功能模組
│   ├── fetcher.py      # 負責從 yfinance 抓取股價與基本面資料
│   └── plotter.py      # 負責用 matplotlib 繪製走勢圖與河流圖
├── templates/
│   └── index.html      # 網頁前端模板
├── static/
│   ├── index.css       # 前端樣式 (Neumorphism 風格)
│   └── images/         # 動態產生的圖表暫存
├── docs/
│   └── SKILL.md        # 前端設計規範參考 (Neumorphism Design System)
├── app.py              # Flask 後端伺服器
├── portfolio.json      # 自選股清單 (持久化)
├── test_analysis.py    # 針對台積電 (2330.TW) 的測試腳本
└── requirements.txt
```

## 開發環境設置

本專案使用 Python 的虛擬環境 (venv) 進行開發，以確保套件版本獨立。

1. **建立虛擬環境**（若尚未建立）
   ```bash
   python3 -m venv .venv
   ```

2. **啟動虛擬環境**
   - Linux/macOS: `source .venv/bin/activate`
   - Windows: `.\.venv\Scripts\activate`

3. **安裝所需套件**
   ```bash
   pip install -r requirements.txt
   ```

## 執行程式

啟動 Flask 網頁伺服器：
```bash
python3 app.py
```

開啟後在瀏覽器中訪問：**[http://localhost:2330](http://localhost:2330)**

或執行後端測試腳本（不需啟動伺服器）：
```bash
python3 test_analysis.py
```
