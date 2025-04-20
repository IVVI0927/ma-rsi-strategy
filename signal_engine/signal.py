import pandas as pd
import ta


def get_latest_rsi_signal(df, window=14, threshold=45):
    df.columns = [col.capitalize() for col in df.columns]
    rsi = ta.momentum.RSIIndicator(df["Close"], window=window).rsi()
    latest = rsi.iloc[-1]
    if latest < threshold:
        return "buy"
    elif latest > 100 - threshold:
        return "sell"
    else:
        return "hold"


def get_ma_signal(df, short_window=5, long_window=60):
    df.columns = [col.capitalize() for col in df.columns]
    ma_short = df["Close"].rolling(window=short_window).mean()
    ma_long = df["Close"].rolling(window=long_window).mean()
    if ma_short.iloc[-1] > ma_long.iloc[-1] and ma_short.iloc[-2] <= ma_long.iloc[-2]:
        return "golden"  # 金叉
    elif ma_short.iloc[-1] < ma_long.iloc[-1] and ma_short.iloc[-2] >= ma_long.iloc[-2]:
        return "death"   # 死叉
    else:
        return "neutral"


def get_macd_signal(df, fast=12, slow=26, signal=9):
    df.columns = [col.capitalize() for col in df.columns]
    macd = ta.trend.MACD(df["Close"], window_slow=slow, window_fast=fast, window_sign=signal)
    dif = macd.macd_diff()
    latest = dif.iloc[-1]
    prev = dif.iloc[-2]
    if latest > 0 and prev <= 0:
        return "bullish"  # DIF上穿DEA
    elif latest < 0 and prev >= 0:
        return "bearish"  # DIF下穿DEA
    else:
        return "neutral"
