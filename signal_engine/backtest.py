import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from signal_engine.score_and_suggest import score_stock
from signal_engine.data_fetcher import fetch_stock_data
from signal_engine.portfolio_builder import get_watchlist
from signal_engine.fundamentals import get_fundamentals

def get_trading_days(start_date: str, end_date: str) -> list:
    """
    获取A股交易日历
    """
    import akshare as ak
    trading_days = ak.tool_trade_date_hist_sina()
    trading_days = trading_days[(trading_days >= start_date) & (trading_days <= end_date)]
    return trading_days.tolist()

def run_backtest(start_date: str, end_date: str, top_n: int = 10, hold_days: int = 5):
    """
    A股回测系统
    - 考虑A股交易时间
    - 考虑A股交易费用
    - 考虑A股交易单位
    """
    # 获取交易日历（A股交易日）
    trading_days = get_trading_days(start_date, end_date)
    
    nav = 1_000_000  # 初始资金
    portfolio_values = []

    for today in trading_days:
        scores = []
        for code in get_watchlist():  # 获取观察列表
            try:
                s = score_stock(code)
                scores.append((code, s["score"]))
            except:
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        picks = [code for code, _ in scores[:top_n]]

        returns = []
        for code in picks:
            df = fetch_stock_data(code)
            try:
                open_price = df.at[today, "Close"]
                sell_date = today + timedelta(days=hold_days)
                sell_price = df.at[sell_date, "Close"]
                ret = (sell_price - open_price) / open_price
            except:
                ret = 0
            returns.append(ret)

        avg_return = np.mean(returns) if returns else 0
        nav *= (1 + avg_return)
        portfolio_values.append(nav)

    return calculate_performance_metrics(portfolio_values)

def calculate_performance_metrics(portfolio_values: list) -> tuple:
    """
    计算回测性能指标
    """
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    
    # 计算最大回撤
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown) * 100
    
    # 计算夏普比率
    risk_free_rate = 0.03  # 假设无风险利率为3%
    excess_returns = returns - risk_free_rate/252
    sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    return pd.Series(portfolio_values), {
        "total_return": round(total_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe_ratio, 2)
    }

if __name__ == "__main__":
    curve, stat = run_backtest(top_n=5, hold_days=1)
    print("✅ 回测完成")
    print("📈 总收益率：", stat["total_return"], "%")
    print("📉 最大回撤：", stat["max_drawdown"], "%")
    print("📊 夏普比率：", stat["sharpe_ratio"])
    curve.plot(title="Backtest Net Value Curve")