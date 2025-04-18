import pandas as pd
import matplotlib.pyplot as plt
from signal_engine.signal import generate_signal
from signal_engine.backtest import run_backtest

# 1. 读取数据
df = pd.read_csv("data/hs300_daily_2023_2025.csv", index_col=0, parse_dates=True)

#2. 定义参数组合
ma_short_list   = [3, 5, 10]
ma_long_list    = [30, 50, 60]
rsi_thresh_list = [45, 50]

#3. 批量回测参数组合，收集结果
results = []
for ma_short in ma_short_list:
    for ma_long in ma_long_list:
        if ma_short >= ma_long:
            continue
        for rsi_thresh in rsi_thresh_list:
            # 生成信号
            df_copy = generate_signal(df.copy(), ma_short, ma_long, rsi_thresh)
            # 回测
            profit, drawdown, curve, index = run_backtest(df_copy)
            results.append({
                "ma_short": ma_short,
                "ma_long":  ma_long,
                "rsi":      rsi_thresh,
                "profit":   profit,
                "drawdown": drawdown,
                "final":    curve[-1],
                "curve":    curve,
                "index":    index
            })
            # 打印每组结果
            print(f"MA{ma_short}/{ma_long}, RSI>{rsi_thresh} → "
                  f"Profit: {profit:.2f}%, Drawdown: {drawdown:.2f}%")

#4. 输出所有组合结果到CSV file
df_results = pd.DataFrame(results)
df_results.to_csv("strategy_combinations.csv", index=False)
print("\n✅ All parameter results saved to strategy_combinations.csv")

#5. 筛选最优组合：优先回撤<10%，否则放宽至<15%，再退回所有？为什么总是报错
filtered = [r for r in results if r["drawdown"] < 10]
print(f"▶ Combos with drawdown<10%: {len(filtered)}")

if not filtered:
    print("⚠️ No combos with drawdown<10%, trying drawdown<15%")
    filtered = [r for r in results if r["drawdown"] < 15]
    print(f"▶ Combos with drawdown<15%: {len(filtered)}")

if not filtered:
    print("⚠️ Still none match drawdown criteria, falling back to all combos")
    filtered = results

#6.筛选收益最高
best = max(filtered, key=lambda r: r["profit"])
print("\n=== 🎯 Optimal Strategy ===")
print(f"Parameters: MA{best['ma_short']}/{best['ma_long']}, RSI>{best['rsi']}")
print(f"Profit: {best['profit']:.2f}%   Drawdown: {best['drawdown']:.2f}%")

#7. 绘制、保存
plt.figure(figsize=(14, 6))
plt.plot(best["index"], best["curve"],
         label=f"MA{best['ma_short']}/{best['ma_long']}, RSI>{best['rsi']}",
         linewidth=2)
plt.title("Optimal Strategy Equity Curve")
plt.xlabel("Time")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存图像不弹窗
output_plot = "optimal_equity_curve.png"
plt.savefig(output_plot, dpi=150)
plt.close()
print(f"✅ Optimal equity curve saved to {output_plot}")