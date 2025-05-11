# 📊 A-Share Quantitative Stock Selection System

A lightweight quantitative stock selection system for Chinese A-shares, featuring multi-factor evaluation, backtesting, and portfolio suggestions.

## 📁 Project Structure

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

## ✅ Key Features

### 1. Multi-Factor Scoring System
- Technical Factors: MA, RSI, MACD
- Fundamental Factors: PE, PB, Market Cap
- Customizable factor weights
- Daily automatic scoring and recommendations

### 2. Data Sources
- JoinQuant data (jqdatasdk)
- Local CSV data backup
- Automatic data updates

### 3. Portfolio Suggestions
- Automatic stock selection based on scores
- A-share trading rules (100-share minimum)
- Capital allocation and position control
- Price limit considerations

### 4. Backtesting System
- Historical data backtesting
- Performance metrics calculation
- Visualization of results

### 5. Web API
- FastAPI interface
- Real-time score queries
- Portfolio suggestion endpoints

## 🚀 Quick Start

### Requirements
- Python 3.8+
- JoinQuant account

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
1. Configure JoinQuant credentials in `signal_engine/fundamentals.py`:
```python
auth("your_username", "your_password")
```

2. Place A-share data files in the `data` directory:
- Format: `600519.SH.csv` (Shanghai) or `000001.SZ.csv` (Shenzhen)
- Required fields: Date, Open, High, Low, Close, Volume

### Running
```bash
# Run main program
python main.py

# Start API server
uvicorn app_server.api_server:app --reload
```

## 📊 Scoring Logic

### Technical Indicators (60% weight)
- RSI (Relative Strength Index): 30%
- MA (Moving Average): 30%
- MACD (Trend Indicator): 40%

### Fundamental Factors (40% weight)
- PE (Price-to-Earnings): 30%
- PB (Price-to-Book): 30%
- Market Cap: 40%

## 🌐 API Endpoints

### Get Today's Recommendations
```
GET /api/get_today_scores
```

### Get Portfolio Suggestions
```
GET /api/recommend_portfolio?capital=100000&max_stocks=5
```

## 📈 Output Files

- `today_recommendations.csv`: Daily scoring results
- `today_portfolio.csv`: Portfolio suggestions
- `logs/recommend_log_*.txt`: Operation logs

## 🔧 Development Guide

### Adding New Factors
1. Create new factor file in `signal_engine` directory
2. Integrate new factor in `score_and_suggest.py`
3. Adjust weight configuration

### Customizing Scoring Rules
Modify the `calculate_score` function in `score_and_suggest.py`

## 📝 Important Notes

1. Data Updates
   - Ensure daily local data updates
   - Check JoinQuant data connection status

2. Risk Control
   - Implement stop-loss
   - Position sizing
   - Market risk consideration

3. System Maintenance
   - Regular log checks
   - Performance monitoring
   - Data backup

## 🤝 Contributing

1. Fork the project
2. Create feature branch
3. Commit changes
4. Submit Pull Request

## 📄 License

MIT License