print("▶▶▶ Using 6‑factor scoring ▶▶▶")
import pandas as pd
import os
from signal_engine.fundamentals import get_pe_pb
from signal_engine.stock_signal import (
    get_latest_rsi_signal,
    get_ma_signal,
    get_macd_signal
)
from signal_engine.ai_score_stock import ai_score_stock
from signal_engine.portfolio_builder import build_portfolio
from signal_engine.ai_model import call_ai_model
from signal_engine.news_sentiment import get_sentiment
import datetime
os.makedirs("logs", exist_ok=True)  # 如果 logs 不存在，自动创建这个文件夹
log_path = f"logs/recommend_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_path, "w", encoding="utf-8")


# 是否使用 AI 模型
use_ai_model = True

def score_stock(code, use_ai_model=False):
    df = pd.read_csv(f"data/{code}.csv")
    df.columns = [col.capitalize() for col in df.columns]

    # 技术指标
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
        volatility = df["Close"].pct_change().rolling(10).std().iloc[-1]
        from signal_engine.fundamentals import (
            get_roe, get_net_profit_margin, get_debt_ratio, get_profit_growth,
            get_ps, get_turnover_rate, get_volume_change
        )
        from signal_engine.stock_signal import get_bollinger_width, get_momentum_10, get_macd_value

        roe = get_roe(code)
        net_margin = get_net_profit_margin(code)
        debt_ratio = get_debt_ratio(code)
        profit_growth = get_profit_growth(code)
        ps_ratio = get_ps(code)
        turnover_rate = get_turnover_rate(code)
        volume_change = get_volume_change(df)
        boll_width = get_bollinger_width(df)
        momentum_10 = get_momentum_10(df)
        macd_value = get_macd_value(df)

        features = {
            "rsi": rsi_value,
            "ma_diff": ma_diff,
            "pe": pe,
            "pb": pb,
            "volatility": round(volatility, 4),
            "news_sentiment": get_sentiment(code),
            "macd": macd_value,
            "momentum_10": momentum_10,
            "roe": roe,
            "turnover_rate": turnover_rate,
            "volume_change": volume_change,
            "bollinger_width": boll_width,
            "profit_growth": profit_growth,
            "ps": ps_ratio,
            "debt_ratio": debt_ratio,
            "net_profit_margin": net_margin
        }
        ai_result = call_ai_model(features)
        print("📈 推荐股票:", code)
        print("🔢 得分:", ai_result.get("score", "?"))
        print("🧠 理由:", ai_result.get("reason", "").strip())
        print("📊 特征因子:", features)
        print("-" * 50)
        log_file.write(f"📈 推荐股票: {code}\n")
        log_file.write(f"🔢 得分: {ai_result.get('score', '?')}\n")
        log_file.write(f"🧠 理由: {ai_result.get('reason', '').strip()}\n")
        log_file.write(f"📊 特征因子: {features}\n")
        log_file.write("-" * 50 + "\n")
        score = ai_result.get("score", 0.5)
        suggest = "✅ BUY" if score >= 0.8 else "HOLD"
        reason = ai_result.get("reason", "AI agent decision")
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

log_file.close()
