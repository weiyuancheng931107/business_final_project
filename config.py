import os
from dotenv import load_dotenv

# 載入當前目錄下的 .env 檔案
load_dotenv()

# API 設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# 預設為 OpenAI 官方 API 網址，之後可於 .env 中覆寫為 OpenRouter 網址
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("⚠️ 警告：未設定 OPENAI_API_KEY 環境變數。請在專案根目錄的 .env 檔案中填入您的 API 金鑰。")
