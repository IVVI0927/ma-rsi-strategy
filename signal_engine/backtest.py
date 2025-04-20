import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from signal_engine.score_and_suggest import score_stock
from jqdatasdk import auth

def run_backtest(start_date="2025-01-10", end_date="2025-01-16",
                 top_n=5, hold_days=1):
    # === 读取基准日期列表（使用任意一只股票）
    sample = pd.read_csv("data/600519.SH.csv", index_col=0, parse_dates=True)
    dates = sample.index
    dates = dates[(dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))]
    dates = sorted(dates.strftime("%Y-%m-%d").tolist())

    nav = 1_000_000
    portfolio_values = []

    for today in dates:
        print(f"\n📅 Processing date: {today}")
        scores = []
        for f in os.listdir("data"):
            if not f.endswith(".csv"):
                continue
            code = f.replace(".csv", "")
            print(f"  📊 Scoring stock: {code}")
            try:
                s = score_stock(code)
                scores.append((code, s["score"]))
            except:
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        picks = [code for code, _ in scores[:top_n]]

        returns = []
        for code in picks:
            df = pd.read_csv(f"data/{code}.csv", index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)
            try:
                open_price = df.at[pd.to_datetime(today), "close"]
                sell_date = pd.to_datetime(today) + timedelta(days=hold_days)
                sell_price = df.at[sell_date, "close"]
                ret = (sell_price - open_price) / open_price
            except:
                ret = 0
            returns.append(ret)

        avg_return = np.mean(returns) if returns else 0
        nav *= (1 + avg_return)
        portfolio_values.append(nav)

    pnl = pd.Series(portfolio_values, index=pd.to_datetime(dates))
    stats = {
        "total_return": round((nav / 1_000_000 - 1) * 100, 2),
        "max_drawdown": round(((pnl.cummax() - pnl).max() / pnl.cummax().max()) * 100, 2)
    }
    return pnl, stats


if __name__ == "__main__":
    curve, stat = run_backtest(top_n=5, hold_days=1)
    print("✅ 回测完成")
    print("📈 总收益率：", stat["total_return"], "%")
    print("📉 最大回撤：", stat["max_drawdown"], "%")
    curve.plot(title="Backtest Net Value Curve")