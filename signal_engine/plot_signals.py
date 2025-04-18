import matplotlib.pyplot as plt

def plot_signals(df, ma_short, ma_long, rsi_threshold):
    df = df.copy()
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Close Price', linewidth=1)
    plt.plot(df.index, df['MA_short'], label=f'MA{ma_short}', linestyle='--')
    plt.plot(df.index, df['MA_long'], label=f'MA{ma_long}', linestyle=':')

    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy', s=100)
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell', s=100)

    plt.title(f"Signal Plot: MA{ma_short}/{ma_long}, RSI>{rsi_threshold}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
