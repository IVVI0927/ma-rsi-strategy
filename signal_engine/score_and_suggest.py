print("▶▶▶ Using 6‑factor scoring ▶▶▶")
import pandas as pd
import os
from signal_engine.fundamentals import get_pe_pb
from signal_engine.signal import (
    get_latest_rsi_signal,
    get_ma_signal,
    get_macd_signal
)
from signal_engine.ai_score_stock import ai_score_stock
from signal_engine.portfolio_builder import build_portfolio
from signal_engine.ai_model import call_ai_model

# 是否使用 AI 模型
use_ai_model = True

def score_stock(code, use_ai_model=False):
    df = pd.read_csv(f"data/{code}.csv")
    df.columns = [col.capitalize() for col in df.columns]

    # === 技术指标 ===
    rsi_signal = get_latest_rsi_signal(df)
    ma_signal = get_ma_signal(df)
    macd_signal = get_macd_signal(df)

    # 提取特征
    rsi_value = df["Rsi"].iloc[-1] if "Rsi" in df.columns else 50
    ma_short_val = df["Close"].rolling(5).mean().iloc[-1]
    ma_long_val = df["Close"].rolling(20).mean().iloc[-1]
    ma_diff = round(ma_short_val - ma_long_val, 2)

    # 基本面
    fundamentals = get_pe_pb(code)
    pe = fundamentals.get("pe_ttm", 10)
    pb = fundamentals.get("pb", 1)
    market_cap = fundamentals.get("market_cap", 1000)

    if use_ai_model:
        features = {
            "rsi": rsi_value,
            "ma_diff": ma_diff,
            "pe": pe,
            "pb": pb,
            "volatility": 0.02
        }
        score = ai_score_stock(features)
        suggest = "✅ BUY" if score >= 0.8 else "HOLD"
        reason = "AI agent decision"
    else:
        score = 0
        max_score = 2.0
        if rsi_signal == "buy":
            score += 0.3
        if ma_signal == "golden":
            score += 0.3
        if macd_signal == "bullish":
            score += 0.4
        if pe < 20:
            score += 0.3
        if pb < 2:
            score += 0.3
        if market_cap < 1000:
            score += 0.4
        score = round(score / max_score, 2)
        suggest = "✅ BUY" if score >= 0.8 else "HOLD"
        reason = "Rule-based logic"

    return {
        "code": code,
        "pe": pe,
        "pb": pb,
        "market_cap": market_cap,
        "score": round(score, 3),
        "suggest": suggest,
        "reason": reason
    }

if __name__ == "__main__":
    all_codes = [
        f.replace(".csv", "")
        for f in os.listdir("data")
        if f.endswith(".csv") and (".SH" in f or ".SZ" in f)
    ]

    results = []
    for code in all_codes:
        result = score_stock(code, use_ai_model=use_ai_model)
        results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values(by="score", ascending=False)
    df.to_csv("today_recommendations.csv", index=False)

# 可选：在推荐完成后直接生成组合建议
portfolio = build_portfolio(df, capital=50000, max_stocks=5)
portfolio.to_csv("today_portfolio.csv", index=False)
print("✅ 今日推荐已保存至 today_recommendations.csv")
