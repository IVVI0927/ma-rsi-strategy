# ✅ A-Share Quantitative Trading System
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 默认运行API服务器
CMD ["python", "api_server.py"]