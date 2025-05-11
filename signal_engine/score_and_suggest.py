print("▶▶▶ Using 6‑factor scoring ▶▶▶")
import pandas as pd
import os
from datetime import datetime
from signal_engine.fundamentals import get_fundamentals
from signal_engine.stock_signal import (
    get_latest_rsi_signal,
    get_ma_signal,
    get_macd_signal
)
import logging

# 创建日志目录
os.makedirs("logs", exist_ok=True)
log_path = f"logs/recommend_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

def score_stock(code: str, use_ai_model: bool = False) -> dict:
    """评分主函数"""
    try:
        # 读取本地数据
        df = pd.read_csv(f"data/{code}.csv")
        df.columns = [col.capitalize() for col in df.columns]

        # 技术指标
        rsi_signal = get_latest_rsi_signal(df)
        ma_signal = get_ma_signal(df)
        macd_signal = get_macd_signal(df)

        # 基本面
        fundamentals = get_fundamentals(code)
        pe = fundamentals.get("pe_ttm")
        pb = fundamentals.get("pb")
        market_cap = fundamentals.get("market_cap")

        # 计算得分
        score = 0
        max_score = 2.0
        
        # 技术指标得分
        if rsi_signal == "buy":
            score += 0.3
        if ma_signal == "golden":
            score += 0.3
        if macd_signal == "bullish":
            score += 0.4
            
        # 基本面得分（只在有数据时计算）
        if pe is not None and pe < 20:
            score += 0.3
        if pb is not None and pb < 2:
            score += 0.3
        if market_cap is not None and market_cap < 1000:
            score += 0.4

        score = round(score / max_score, 2)
        suggest = "✅ BUY" if score >= 0.8 else "HOLD"

        result = {
            "code": code,
            "pe": pe,
            "pb": pb,
            "market_cap": market_cap,
            "score": round(score, 3),
            "suggest": suggest,
            "reason": "Rule-based logic"
        }
        
        logging.info(f"Scored {code}: {result}")
        return result
        
    except Exception as e:
        logging.error(f"Error scoring {code}: {e}")
        return {}

def get_today_scores():
    """提供 FastAPI 接口调用今日打分推荐"""
    if not os.path.exists("today_recommendations.csv"):
        return []

    df = pd.read_csv("today_recommendations.csv")
    df = df.sort_values(by="score", ascending=False)
    results = df.to_dict(orient="records")
    return results

if __name__ == "__main__":
    all_codes = [
        f.replace(".csv", "")
        for f in os.listdir("data")
        if f.endswith(".csv") and (".SH" in f or ".SZ" in f)
    ]

    results = []
    for code in all_codes:
        result = score_stock(code)
        if result:  # 只添加非空结果
            results.append(result)

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="score", ascending=False)
        df.to_csv("today_recommendations.csv", index=False)
        print("✅ 今日推荐已保存至 today_recommendations.csv")
