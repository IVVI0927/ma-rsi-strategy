# ✅ 后端容器（Python 回测引擎）
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 默认运行回测主逻辑（你可以换成 FastAPI 等）
CMD ["python", "backtest_engine.py"]