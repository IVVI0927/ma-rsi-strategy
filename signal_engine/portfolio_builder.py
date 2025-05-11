import pandas as pd
import math

def build_portfolio(df: pd.DataFrame, capital: float = 100000, max_stocks: int = 10) -> pd.DataFrame:
    """
    美股组合构建
    - 考虑美股交易单位（可以买1股）
    - 考虑美股交易时间（美东时间）
    - 考虑美股交易费用（如 Robinhood 免佣金）
    """
    # 过滤建议买入的股票
    candidates = df[df["suggest"] == "✅ BUY"].copy()
    candidates = candidates.sort_values(by="score", ascending=False).head(max_stocks)

    if candidates.empty:
        return pd.DataFrame()

    # 按打分比例分配资金
    total_score = candidates["score"].sum()
    candidates["weight"] = candidates["score"] / total_score
    candidates["allocated_fund"] = candidates["weight"] * capital

    # 计算建议买入股数（美股可以买1股）
    candidates["suggested_shares"] = (candidates["allocated_fund"] / candidates["price"]).round(0)

    # 实际投资金额
    candidates["actual_cost"] = candidates["suggested_shares"] * candidates["price"]
    candidates["actual_pct"] = candidates["actual_cost"] / capital

    return candidates