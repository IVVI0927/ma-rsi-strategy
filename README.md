# ma-rsi-strategy
# MA+RSI Quant Strategy API

📈 一个基于 MA+RSI 的 A 股量化策略，支持 FastAPI 部署与参数调优。

## 项目结构
- `main.py`: 本地运行，绘图评估策略
- `app.py`: FastAPI 接口，运行策略接口 /run_strategy
- `optimize_and_plot.py`: 多组参数自动回测并保存最优结果
- `signal_engine/`: 策略核心逻辑模块
- `data/`: 数据源（可换成自己的股票数据）

## 本地运行

```bash
pip install -r requirements.txt
python main.py
