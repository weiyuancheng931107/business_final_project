# 商管程式設計期末專案 - 股票基本面分析

這是一個以 Python 實作的台股基本面分析工具。
透過網頁介面管理自選股、查看歷史股價走勢、本益比河流圖，以及讀取真實財經新聞與基本面資料。

## 主要功能

- 自選股管理：新增、移除並持久化台股清單。
- 基本面資訊：顯示目前股價、本益比 P/E、每股盈餘 EPS、ROE。
- 圖表分析：產生歷史股價走勢圖與本益比河流圖。
- 相關新聞：從真實新聞來源載入個股相關新聞，支援載入更多。
- 穩定 fallback：當 yfinance 因 SSL、網路或資料缺漏失敗時，改用 FinMind 或備援圖表，避免前端 500/404。

## 資料來源與穩定性更新

本專案已將原本容易失敗或模擬的資料流程改為真實資料來源：

- 新聞資料：
  - 優先讀取 yfinance 內建 Yahoo Finance 新聞。
  - 同時整合 FinMind `TaiwanStockNews`。
  - 回傳資料會正規化成前端固定格式：`title`、`source`、`date`、`url`。
  - 若 yfinance 因本機 SSL 憑證問題失敗，會自動 fallback 到 FinMind，不讓新聞 API 中斷。

- 股價資料：
  - 優先使用 yfinance 歷史股價。
  - 若 yfinance 回傳空資料或發生錯誤，改用 FinMind `TaiwanStockPrice`。
  - 圖表 API 即使暫時無法取得資料，也會回傳 placeholder 圖表，不再讓前端收到 404。

- 基本面數據：
  - 本益比 P/E 取 FinMind `TaiwanStockPER` 最近一筆 `PER`。
  - 每股盈餘 EPS 取 FinMind `TaiwanStockFinancialStatements` 最近一期 `EPS`。
  - ROE 以最近四期 `IncomeAfterTaxes` 加總除以最新權益計算，回傳 decimal，前端顯示為百分比。
  - 若 yfinance fundamentals 無法使用，P/E、EPS、ROE 仍可由 FinMind 補值。

- 快取與產物：
  - yfinance cache 改放在專案內 `.yfinance_cache/`。
  - `.yfinance_cache/` 與測試快取不進 Git。
  - `static/images/` 內的圖表是執行時產物，若只是本機測試重新產生，不需要納入 commit。

## 專案結構

```
business_final_project/
├── analysis/           # 核心功能模組
│   ├── fetcher.py      # yfinance + FinMind 資料抓取、fallback 與資料正規化
│   └── plotter.py      # matplotlib 圖表與 placeholder 圖表產生
├── templates/
│   └── index.html      # 網頁前端模板
├── static/
│   ├── index.css       # 前端樣式 (Neumorphism 風格)
│   └── images/         # 動態產生的圖表暫存
├── docs/
│   └── SKILL.md        # 前端設計規範參考 (Neumorphism Design System)
├── tests/              # 新聞、股價、基本面 fallback 的回歸測試
├── app.py              # Flask 後端伺服器
├── portfolio.json      # 自選股清單 (持久化)
└── requirements.txt
```

## 開發環境設置

本專案使用 Python 的虛擬環境 (venv) 進行開發，以確保套件版本獨立。

1. **建立虛擬環境**（若尚未建立）
   - Linux/macOS: `python3 -m venv .venv`
   - Windows: `python -m venv .venv`

2. **啟動虛擬環境**
   - Linux/macOS: `source .venv/bin/activate`
   - Windows: `.\.venv\Scripts\activate`

3. **安裝所需套件**
   ```bash
   pip install -r requirements.txt
   ```

## 執行程式

啟動 Flask 網頁伺服器：
- Linux/macOS: `python3 app.py`
- Windows: `python app.py`

開啟後在瀏覽器中訪問：**[http://localhost:2330](http://localhost:2330)**

## 測試

執行所有回歸測試：

```bash
python -m unittest discover -s tests
```

目前測試覆蓋：

- yfinance 新聞格式正規化。
- yfinance 與 FinMind 新聞合併、去重。
- FinMind 新聞單日查詢格式。
- yfinance 股價失敗時改用 FinMind 股價。
- yfinance 基本面失敗時改用 FinMind P/E、EPS、ROE。
- 圖表與 API 在資料來源失敗時仍回 200，不回 500/404。
