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
    valid_codes = []
    valid_prices = []
    valid_closes = []

    for i, code in enumerate(df["code"].tolist()):
        if i < len(price_list) and i < len(close_list):
            valid_codes.append(code)
            valid_prices.append(price_list[i])
            valid_closes.append(close_list[i])

    df = df[df["code"].isin(valid_codes)].reset_index(drop=True)
    df["price"] = valid_prices
    df["yesterday_close"] = valid_closes

    # 计算涨停价，判断是否涨停
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

    # 计算建议买入股数（单位：100股）
    candidates["suggested_shares"] = candidates["allocated_fund"] / candidates["price"]
    candidates["suggested_shares"] = candidates["suggested_shares"].apply(lambda x: math.floor(x / 100) * 100)

    # 实际投资金额与资金占比
    candidates["actual_cost"] = candidates["suggested_shares"] * candidates["price"]
    candidates["actual_pct"] = candidates["actual_cost"] / capital

    # 加入买入手续费（0.05%，可调）
    candidates["buy_fee"] = (candidates["actual_cost"] * 0.0005).round(2)
    candidates["total_cost"] = candidates["actual_cost"] + candidates["buy_fee"]

    # 止盈止损比例（可调）
    take_profit_pct = 0.10 
    stop_loss_pct = 0.05  

    # 计算止盈止损价
    candidates["target_profit_price"] = (candidates["price"] * (1 + take_profit_pct)).round(2)
    candidates["stop_loss_price"] = (candidates["price"] * (1 - stop_loss_pct)).round(2)

    # 卖出手续费（0.15%，包含印花税等）
    sell_fee_rate = 0.0015

    # 预估卖出手续费 + 净利润（触发止盈时）
    candidates["sell_fee_at_tp"] = (candidates["target_profit_price"] * candidates["suggested_shares"] * sell_fee_rate).round(2)
    gross_sell = candidates["target_profit_price"] * candidates["suggested_shares"]
    total_cost_all = candidates["total_cost"] + candidates["sell_fee_at_tp"]
    candidates["profit_after_tp"] = (gross_sell - total_cost_all).round(2)
    candidates["return_ratio"] = (candidates["profit_after_tp"] / candidates["total_cost"]).round(4)
    
    # 止损时的卖出手续费与亏损估算
    candidates["sell_fee_at_sl"] = (candidates["stop_loss_price"] * candidates["suggested_shares"] * sell_fee_rate).round(2)
    gross_sell_sl = candidates["stop_loss_price"] * candidates["suggested_shares"]
    total_cost_all_sl = candidates["total_cost"] + candidates["sell_fee_at_sl"]
    candidates["loss_after_sl"] = (gross_sell_sl - total_cost_all_sl).round(2)
    candidates["loss_ratio"] = (candidates["loss_after_sl"] / candidates["total_cost"]).round(4)

    portfolio = candidates[[
        "code", "price", "score", "suggested_shares",
        "actual_cost", "actual_pct", "is_limit_up", "reason",
        "target_profit_price", "stop_loss_price", "buy_fee", "total_cost",
        "sell_fee_at_tp", "profit_after_tp", "return_ratio",
        "sell_fee_at_sl", "loss_after_sl", "loss_ratio"
    ]].reset_index(drop=True)

    return portfolio