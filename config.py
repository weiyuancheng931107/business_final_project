# config.py

from openai import OpenAI

OPENAI_API_KEY = "API-key"

client = OpenAI(
    api_key=OPENAI_API_KEY
)

GPT_MODEL = "gpt-4.1-mini"
