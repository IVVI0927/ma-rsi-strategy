from signal_engine.backtest import run_backtest
from signal_engine.recommend import get_top_stocks
import pandas as pd

def update_and_save() -> None:
    print("▶▶▶ Running update_daily.py")
    
    # — 可选回测 & 保存 —
    curve, stats = run_backtest()
    curve.to_csv("backtest_curve.csv")
    with open("backtest_stats.txt", "w") as f:
        f.write(f"Total Return: {stats['total_return']}%\n")
        f.write(f"Max Drawdown: {stats['max_drawdown']}%\n")

    # — 获取推荐列表（list of dict） —
    top_stocks = get_top_stocks(top_n=10)
    # 确认一下格式（调试时可打开注释）
    # print(top_stocks[:2], type(top_stocks))

    # — 转成 DataFrame —
    df = pd.DataFrame(top_stocks)

    # — 如果有 reason 字段，把 list 转成字符串 —
    if "reason" in df.columns:
        df["reason"] = df["reason"].apply(
            lambda x: "; ".join(x) if isinstance(x, list) else str(x)
        )

    # — 最终只保存前端需要的三列 —
    df[["code", "score", "suggest"]].to_csv(
        "today_recommendations.csv", index=False
    )

    print("✅ 推荐列表已保存：today_recommendations.csv")

if __name__ == "__main__":
    update_and_save()