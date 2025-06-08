from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
from signal_engine.portfolio_builder import build_portfolio
from logging_config import app_logger, access_logger
import time
from typing import Dict, Any

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Log access in JSON format
    access_logger.info(
        f"Request processed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": f"{process_time:.3f}s",
            "client_ip": request.client.host if request.client else None
        }
    )
    return response

@app.get("/")
def root() -> dict:
    app_logger.info("Root endpoint accessed")
    return {"message": "Quant Portfolio API is running."}

@app.get("/recommend")
def get_recommendations() -> dict:
    app_logger.info("Getting recommendations")
    path = "today_recommendations.csv"
    if not os.path.exists(path):
        app_logger.error(f"Recommendation file not found: {path}")
        return {"status": "error", "message": "No recommendation file found."}
    try:
        df = pd.read_csv(path)
        result = df[["code", "score", "suggest"]].to_dict(orient="records")
        app_logger.info(f"Successfully retrieved {len(result)} recommendations")
        return {"status": "success", "data": result}
    except Exception as e:
        app_logger.error(f"Error getting recommendations: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}

class PortfolioRequest(BaseModel):
    capital: float = 50000
    max_stocks: int = 5

@app.post("/recommend_portfolio/")
def recommend_portfolio(request: PortfolioRequest) -> dict:
    app_logger.info(f"Portfolio recommendation requested with capital={request.capital}, max_stocks={request.max_stocks}")
    try:
        df = pd.read_csv("today_recommendations.csv")
        portfolio = build_portfolio(df, capital=request.capital, max_stocks=request.max_stocks)
        app_logger.info(f"Successfully built portfolio with {len(portfolio)} stocks")
        return {
            "success": True,
            "portfolio": portfolio.to_dict(orient="records")
        }
    except Exception as e:
        app_logger.error(f"Error building portfolio: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

@app.on_event("startup")
async def startup_event():
    app_logger.info("API server starting up")

@app.on_event("shutdown")
async def shutdown_event():
    app_logger.info("API server shutting down")