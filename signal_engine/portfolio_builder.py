import pandas as pd
import math

def build_portfolio(df, capital=50000, max_stocks=5):
    """
    构建组合建议，考虑A股涨停限制、100股最小单位、评分权重分配资金。
    参数：
        df: 推荐结果DataFrame，需包含 ['code', 'score', 'suggest', 'reason']
        capital: 用户初始资金（如50000元）
        max_stocks: 最多选几只股票
    返回：
        portfolio_df: 包含建议买入股数、资金分配、涨停状态的组合建议表
    """

    # 模拟当前价格和昨日收盘价（未来可接入真实接口）
    price_list = [39.2, 45.1, 168, 12.3, 8.5, 17.6, 29.9, 23.5, 55.0, 41.2]
    close_list = [38.5, 44.9, 152.7, 11.8, 8.0, 17.2, 29.8, 23.0, 53.0, 40.0]
    df = df.copy()
    df["price"] = price_list[:len(df)]
    df["yesterday_close"] = close_list[:len(df)]

    # 计算涨停价，并判断是否涨停
    df["limit_up_price"] = (df["yesterday_close"] * 1.10).round(2)
    df["is_limit_up"] = df["price"] >= df["limit_up_price"]

    # 过滤建议买入 + 非涨停股票
    candidates = df[(df["suggest"] == "✅ BUY") & (~df["is_limit_up"])].copy()
    candidates = candidates.sort_values(by="score", ascending=False).head(max_stocks)

    if candidates.empty:
        return pd.DataFrame(columns=[
            "code", "price", "score", "suggested_shares",
            "actual_cost", "actual_pct", "is_limit_up", "reason"
        ])

    # 按打分比例分配资金
    total_score = candidates["score"].sum()
    candidates["weight"] = candidates["score"] / total_score
    candidates["allocated_fund"] = candidates["weight"] * capital

    # 计算建议买入股数（100股为单位）
    candidates["suggested_shares"] = candidates["allocated_fund"] / candidates["price"]
    candidates["suggested_shares"] = candidates["suggested_shares"].apply(lambda x: math.floor(x / 100) * 100)

    # 实际投资金额与资金占比
    candidates["actual_cost"] = candidates["suggested_shares"] * candidates["price"]
    candidates["actual_pct"] = candidates["actual_cost"] / capital

    portfolio = candidates[[
        "code", "price", "score", "suggested_shares",
        "actual_cost", "actual_pct", "is_limit_up", "reason"
    ]].reset_index(drop=True)

    return portfolio