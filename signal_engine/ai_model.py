# ai_model.py
import requests

def call_ai_model(features: dict) -> float:
    """
    调用本地或远程 AI 模型接口，根据输入特征返回预测分数。
    参数：features 是一个 dict，包含如下字段：
      - pe
      - pb
      - rsi
      - macd
      - ma_diff
      - market_cap
    返回：score（0~1 之间的浮点数）
    """
    url = "http://127.0.0.1:8000/predict"  # 模拟模型服务地址
    try:
        res = requests.post(url, json=features, timeout=3)
        if res.status_code == 200:
            return res.json().get("score", 0.5)
        else:
            print(f"⚠️ AI模型返回错误码：{res.status_code} → {res.text}")
            return 0.5
    except Exception as e:
        print("❌ 无法连接 AI 模型接口：", e)
        return 0.5
