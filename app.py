# FastAPI 后端服务接口
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from signal_engine.signal import generate_signal
from signal_engine.backtest import run_backtest

app = FastAPI()

class StrategyParams(BaseModel):
    ma_short: int
    ma_long: int
    rsi_threshold: float
    take_profit: float = 0.1
    stop_loss: float = 0.03

@app.post("/run_strategy")
def run_strategy(params: StrategyParams) -> dict:
    df = pd.read_csv("data/hs300_daily_2023_2025.csv", index_col=0, parse_dates=True)
    df = generate_signal(df, params.ma_short, params.ma_long, params.rsi_threshold)
    profit, drawdown, _, _ = run_backtest(df, take_profit=params.take_profit, stop_loss=params.stop_loss)
    return {
        "profit_pct": profit,
        "max_drawdown_pct": drawdown
    }