from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from signal_engine.portfolio_builder import build_portfolio

app = FastAPI()

@app.get("/")
def root() -> dict:
    return {"message": "Quant Portfolio API is running."}

@app.get("/recommend")
def get_recommendations() -> dict:
    path = "today_recommendations.csv"
    if not os.path.exists(path):
        return {"status": "error", "message": "No recommendation file found."}
    try:
        df = pd.read_csv(path)
        result = df[["code", "score", "suggest"]].to_dict(orient="records")
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class PortfolioRequest(BaseModel):
    capital: float = 50000
    max_stocks: int = 5

@app.post("/recommend_portfolio/")
def recommend_portfolio(request: PortfolioRequest) -> dict:
    try:
        df = pd.read_csv("today_recommendations.csv")
        portfolio = build_portfolio(df, capital=request.capital, max_stocks=request.max_stocks)
        return {
            "success": True,
            "portfolio": portfolio.to_dict(orient="records")
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }