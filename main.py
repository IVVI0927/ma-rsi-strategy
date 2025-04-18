
import pandas as pd
from signal_engine.signal import generate_signal
from signal_engine.backtest import run_backtest
from signal_engine.plot_signals import plot_signals
import matplotlib.pyplot as plt

df = pd.read_csv("data/hs300_daily_2023_2025.csv", index_col=0, parse_dates=True)

ma_short = 5
ma_long = 60
rsi_threshold = 45

df = generate_signal(df, ma_short, ma_long, rsi_threshold)
# 执行回测
profit, drawdown, curve, index = run_backtest(df)

print(f"\n✅ 策略参数：MA{ma_short}/{ma_long}, RSI>{rsi_threshold}")
print(f"📈 策略收益率：{profit:.2f}%")
print(f"📉 最大回撤：{drawdown:.2f}%")

plot_signals(df, ma_short, ma_long, rsi_threshold)

# 可视化 买卖点图，自动保存
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Close Price', linewidth=1)
plt.plot(df.index, df['MA_short'], linestyle='--', label=f'MA{ma_short}')
plt.plot(df.index, df['MA_long'],  linestyle=':',  label=f'MA{ma_long}')
buy = df[df['Signal']==1]
sell= df[df['Signal']==-1]
plt.scatter(buy.index, buy['Close'], marker='^', color='green', label='Buy',  s=80)
plt.scatter(sell.index,sell['Close'], marker='v', color='red',   label='Sell', s=80)
plt.title(f"Signals: MA{ma_short}/{ma_long}, RSI>{rsi_threshold}")
plt.xlabel("Time"); plt.ylabel("Price"); plt.legend(); plt.grid(True)
plt.tight_layout()

# 自动保存到本地，不要弹窗
plt.savefig("signal_plot.png", dpi=150)  
plt.close()
print("✅ Signal plot saved to signal_plot.png")
