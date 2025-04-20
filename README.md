# 📊 QuantStock AI Agent System (Demo Project)

A lightweight AI-assisted stock scoring and recommendation system for A-shares, featuring multi-factor evaluation, backtesting, portfolio suggestion, and web API exposure.

---

<pre><code>## 📁 Project Structure

```
ma_agent_project/
├── signal_engine/
│   ├── fundamentals.py           # Basic financial factors (PE, PB, market cap)
│   ├── signal.py                 # Technical indicators (RSI, MA, MACD)
│   ├── ai_model.py               # AI model calling interface (mock API)
│   ├── score_and_suggest.py     # Core scoring logic with AI model
│   ├── portfolio_builder.py     # Portfolio construction based on user capital
│   └── backtest.py              # Backtesting engine for strategy evaluation
├── app_server/
│   └── api_server.py            # FastAPI interface (recommendation endpoints)
├── data/
│   └── hs300_daily_2023_2025.csv   # Daily A-share data (local sample)
├── update_daily.py              # Daily run script to refresh today_recommendations.csv
├── app_frontend.py              # Streamlit / Web UI demo (optional)
└── requirements.txt             # Dependencies
```
</code></pre>

---

## ✅ Key Features

### 1. Daily Stock Scoring System
- Combine technical (MA, RSI, MACD) and fundamental (PE, PB, MKT_CAP) factors
- Call external or mock AI model to assign scores and generate suggestions/reasons
- Recommend top stocks daily based on user input (capital, stock limit)

### 2. AI Model Integration
- Mock API built with FastAPI (`mock_ai_server.py`)
- Real model-ready `ai_model.py` for swapping in pretrained models or APIs
- Modular `call_ai_model()` used throughout scoring logic

### 3. Strategy Backtesting
- Evaluate strategy returns and drawdown using historical data
- Visualize net value curve with matplotlib
- Optimize parameters (e.g., MA/RSI grid search)

### 4. Web API (FastAPI)
- `/recommend`: returns all scores and suggestions
- `/recommend_portfolio`: accepts capital, max_stocks → returns filtered buy list

### 5. (Optional) Frontend Demo
- Streamlit interface or custom HTML to visualize scores and charts
- Supports testing JSON outputs live

---

## 🚀 How to Run

**Install dependencies**
```bash
pip install -r requirements.txt
