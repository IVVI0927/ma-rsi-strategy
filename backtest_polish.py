import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from signal_engine.backtest import run_backtest

def analyze_backtest(curve: pd.Series, stats: dict, output_prefix: str = "backtest") -> dict:
    # 1. 计算每日收益
    daily_ret = curve.pct_change().dropna()
    days = len(daily_ret)
    
    # 2. 年化收益率
    total_return = curve.iloc[-1] / curve.iloc[0] - 1
    ann_return = (1 + total_return) ** (252 / days) - 1

    # 3. 年化波动率
    ann_vol = daily_ret.std() * np.sqrt(252)

    # 4. 夏普比（无风险收益率为0）
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    # 5. Calmar 比率
    max_dd = stats["max_drawdown"] / 100
    calmar = ann_return / max_dd if max_dd != 0 else np.nan

    # 6. 胜率
    win_rate = (daily_ret > 0).sum() / days

    # 汇总到 DataFrame
    df = pd.DataFrame([{
        "Total Return (%)": round(total_return * 100, 2),
        "Annualized Return (%)": round(ann_return * 100, 2),
        "Annualized Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Max Drawdown (%)": stats["max_drawdown"],
        "Calmar Ratio": round(calmar, 3),
        "Win Rate (%)": round(win_rate * 100, 2),
    }])

    # 保存指标
    metrics_file = f"{output_prefix}_metrics.csv"
    df.to_csv(metrics_file, index=False)
    print(f"📊 指标已保存：{metrics_file}")

    # 绘制资金曲线与回撤图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    curve.plot(ax=ax1, title="Net Asset Value") 
    ax1.set_ylabel("NAV")

    # 计算并画回撤
    drawdown = (curve.cummax() - curve) / curve.cummax()
    drawdown.plot(ax=ax2, title="Drawdown", color=None)
    ax2.set_ylabel("Drawdown")

    plt.tight_layout()
    img_file = f"{output_prefix}_report.png"
    plt.savefig(img_file)
    print(f"🖼️ 图表已保存：{img_file}")
    plt.close(fig)

if __name__ == "__main__":
    # 直接调用 run_backtest
    curve, stats = run_backtest(top_n=5, hold_days=1)
    analyze_backtest(curve, stats)