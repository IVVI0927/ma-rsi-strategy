from fastapi import FastAPI, Request
from pydantic import BaseModel
import random

app = FastAPI()

class StockFeatures(BaseModel):
    pe: float
    pb: float
    rsi: float
    macd: float
    ma_diff: float
    market_cap: float

@app.post("/predict")
async def predict(features: StockFeatures):
    # 模拟AI模型打分逻辑（这里用随机数代替）
    score = round(random.uniform(0.3, 0.9), 4)  # 假设AI输出的是上涨概率
    return {"score": score}
