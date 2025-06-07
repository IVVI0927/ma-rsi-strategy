import pandas as pd
from signal_engine.backtest import run_backtest

def grid_search(top_n_list: list, hold_days_list: list, start_date: str = "2025-01-10", end_date: str = "2025-01-16") -> None:
    records = []
    for top_n in top_n_list:
        for hold_days in hold_days_list:
            # 调用 run_backtest，只传它支持的参数
            curve, stats = run_backtest(
                start_date=start_date,
                end_date=end_date,
                top_n=top_n,
                hold_days=hold_days
            )
            records.append({
                "top_n": top_n,
                "hold_days": hold_days,
                "total_return_pct": stats["total_return"],
                "max_drawdown_pct": stats["max_drawdown"]
            })
            print(f"Tested top_n={top_n}, hold_days={hold_days} → "
                  f"Return {stats['total_return']:.2f}%, "
                  f"Drawdown {stats['max_drawdown']:.2f}%")
    return pd.DataFrame(records)

if __name__ == "__main__":
    # 调整这两个列表来试不同组合
    top_n_list    = [3, 5, 10]
    hold_days_list = [1, 2, 5]

    df = grid_search(top_n_list, hold_days_list)
    df.to_csv("param_grid_search.csv", index=False)
    print("✅ Grid search complete: param_grid_search.csv")