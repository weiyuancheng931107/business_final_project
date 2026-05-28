FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Node.js 20 (給 Gemini CLI 使用)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


CMD ["/bin/bash"]