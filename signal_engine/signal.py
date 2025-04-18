from .indicators import calculate_rsi

def generate_signal(df, ma_short, ma_long, rsi_threshold):
    df['MA_short'] = df['Close'].rolling(window=ma_short).mean()
    df['MA_long'] = df['Close'].rolling(window=ma_long).mean()
    df['RSI'] = calculate_rsi(df['Close'])

    df['Signal'] = 0
    df.loc[(df['MA_short'] > df['MA_long']) & (df['RSI'] > rsi_threshold), 'Signal'] = 1
    df.loc[(df['MA_short'] < df['MA_long']) & (df['RSI'] < 50), 'Signal'] = -1
    return df
