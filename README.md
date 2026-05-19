# 商管程式設計期末專案 - 股票基本面分析

這是一個以 Python 實作的簡單股票基本面分析工具，主要聚焦於**台股**市場。
目前提供最基礎的資料抓取功能，未來將與小組討論後再進行擴充。

## 專案結構

```
business_final_project/
├── analysis/           # 核心功能模組
│   ├── fetcher.py      # 負責從 yfinance 抓取股價與基本面資料
│   └── plotter.py      # 負責用 matplotlib 畫圖
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

執行測試腳本，會印出台積電的基本面數據並顯示歷史股價圖：
```bash
python3 test_analysis.py
```
