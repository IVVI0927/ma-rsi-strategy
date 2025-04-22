import pandas as pd
import math
import datetime

# 初始化参数
initial_capital = 100000
transaction_fee_rate = 0.0015  # 买入 + 卖出手续费估算
take_profit = 0.10  # 止盈阈值
stop_loss = 0.05    # 止损阈值

# 加载历史行情数据（示例）
data = pd.read_csv("data/hs300_daily_2025_01.csv", parse_dates=["date"])
data = data.sort_values(by=["date", "code"]).reset_index(drop=True)

# 模拟每日推荐结果（替换成实际推荐输出）
def get_daily_recommendations(date):
    try:
        df = pd.read_csv("today_recommendations.csv")
        df = df[df["date"] == date]
        return df.sort_values(by="score", ascending=False)["code"].unique()[:5]
    except:
        return []

# 获取个股某日价格
def get_price(code, date):
    row = data[(data["code"] == code) & (data["date"] == date)]
    return row["close"].values[0] if not row.empty else None

# 主回测函数
def run_backtest():
    trading_days = sorted(data["date"].unique())
    current_cash = initial_capital
    positions = {}  # {code: {"shares": int, "buy_price": float}}
    logs = []

    for date in trading_days:
        print(f"\n📅 当前日期: {date.strftime('%Y-%m-%d')}")
        # 更新当前持仓的持有天数
        for code in positions:
            positions[code]["hold_days"] += 1

        # 止盈止损 + 持仓满3天才允许卖出
        to_sell = []
        for code, pos in positions.items():
            current_price = get_price(code, date)
            if current_price is None:
                continue
            entry_price = pos["buy_price"]
            if pos["hold_days"] >= 3:
                if current_price >= entry_price * (1 + take_profit) or \
                   current_price <= entry_price * (1 - stop_loss):
                    proceeds = pos["shares"] * current_price * (1 - transaction_fee_rate)
                    current_cash += proceeds
                    to_sell.append(code)
        for code in to_sell:
            del positions[code]

        # 获取推荐列表 + 分数 + 排除已持仓
        try:
            rec_df = pd.read_csv("today_recommendations.csv")
            rec_df["date"] = pd.to_datetime(rec_df["date"])  # ensure correct format
            print("🔍 原始推荐记录（前5行）：", rec_df.head())
            rec_df = rec_df[rec_df["date"] == date]
            print("✅ 当前日期推荐股票：", rec_df[["code", "score"]].values.tolist())
            rec_df = rec_df.sort_values(by="score", ascending=False)
        except Exception as e:
            print("❌ 加载推荐数据失败:", e)
            rec_df = pd.DataFrame(columns=["code", "score"])

        buy_df = rec_df[~rec_df["code"].isin(positions)]
        buy_pool = []

        if not buy_df.empty:
            topN = min(5, len(buy_df))
            buy_df = buy_df.head(topN)
            total_score = buy_df["score"].sum()
            buy_df["weight"] = buy_df["score"] / total_score if total_score > 0 else 1.0 / topN
            buy_pool = buy_df[["code", "weight"]].values.tolist()
            print("📊 推荐买入池：", buy_pool)

        # 加权买入
        for code, weight in buy_pool:
            price = get_price(code, date)
            if price is None:
                continue
            budget = current_cash * weight
            shares = int((budget / price) // 100 * 100)
            cost = shares * price * (1 + transaction_fee_rate)
            if shares > 0 and cost <= current_cash:
                positions[code] = {"shares": shares, "buy_price": price, "hold_days": 0}
                current_cash -= cost
                print(f"✅ 买入 {code}：{shares} 股，价格：{price}，成本：{round(cost,2)}")

        print(f"💰 当前现金：{round(current_cash,2)}，持仓：{positions}")
        # 计算总资产
        market_value = sum(get_price(c, date) * p["shares"] for c, p in positions.items() if get_price(c, date))
        total_asset = current_cash + market_value
        logs.append({
            "date": date,
            "cash": round(current_cash, 2),
            "market_value": round(market_value, 2),
            "total_asset": round(total_asset, 2),
            "holdings": list(positions.keys())
        })

    return pd.DataFrame(logs)

if __name__ == "__main__":
    result = run_backtest()
    result.to_csv("backtest_result.csv", index=False)
    print("✅ 回测完成，结果保存至 backtest_result.csv")