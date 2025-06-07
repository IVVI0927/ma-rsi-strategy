import pandas as pd
import numpy as np

def analyze_backtest(path: str = "backtest_result.csv") -> dict:
    df = pd.read_csv(path, parse_dates=["date"])

    df["returns"] = df["total_asset"].pct_change()
    df["cum_returns"] = (1 + df["returns"]).cumprod()

    total_days = len(df)
    annualized_return = (df["total_asset"].iloc[-1] / df["total_asset"].iloc[0]) ** (252 / total_days) - 1

    rolling_max = df["total_asset"].cummax()
    drawdown = df["total_asset"] / rolling_max - 1
    max_drawdown = drawdown.min()

    sharpe = df["returns"].mean() / df["returns"].std() * np.sqrt(252)

    avg_holdings = df["holdings"].apply(lambda x: len(eval(x))).mean()

    print("📊 回测指标报告")
    print(f"年化收益率: {annualized_return:.2%}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"平均持仓数: {avg_holdings:.2f}")

if __name__ == "__main__":
    analyze_backtest()