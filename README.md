# ma-rsi-strategy

## 📈 MA + RSI Quant Strategy API

A simple quantitative strategy based on Moving Average (MA) and Relative Strength Index (RSI), supporting backtesting and FastAPI deployment.

---

## 🔧 Project Structure

ma_agent_project/
├── app.py                   # FastAPI API entry (POST /run_strategy)
├── main.py                 # Run locally with plots (single strategy)
├── optimize_and_plot.py    # Batch parameter testing & best selection
├── requirements.txt        # Dependency list
├── data/
│   └── hs300_daily_2023_2025.csv
├── signal_engine/
│   ├── signal.py           # Signal generation (MA/RSI)
│   ├── backtest.py         # Backtesting engine with stop-loss/take-profit
│   └── plot_signals.py     # Plotting signal indicators
---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py
