def run_backtest(df, take_profit=0.1, stop_loss=0.03):
    df = df.copy()
    capital = 100000
    cash = capital
    position = 0
    entry_price = None
    equity_curve = []

    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i]

        if signal == 1 and position == 0:
            entry_price = price
            position = (cash * 0.5) / price
            cash = cash * 0.5
        elif signal == -1 and position > 0:
            cash += position * price
            position = 0
            entry_price = None
        elif position > 0 and entry_price:
            gain_pct = (price - entry_price) / entry_price
            if gain_pct >= take_profit or gain_pct <= -stop_loss:
                cash += position * price
                position = 0
                entry_price = None

        total_value = cash + position * price
        equity_curve.append(total_value)

    final_value = equity_curve[-1]
    profit_pct = (final_value - capital) / capital * 100

    peak = equity_curve[0]
    max_drawdown = 0
    for val in equity_curve:
        if val > peak:
            peak = val
        drawdown = (peak - val) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return profit_pct, max_drawdown * 100, equity_curve, df.index[-len(equity_curve):]
