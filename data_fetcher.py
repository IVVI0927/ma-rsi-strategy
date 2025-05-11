import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    使用 yfinance 获取美股数据
    symbol: 股票代码，如 'AAPL', 'GOOGL'
    period: 时间范围，如 '1d', '5d', '1mo', '3mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    """
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        df.columns = [col.capitalize() for col in df.columns]
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame() 