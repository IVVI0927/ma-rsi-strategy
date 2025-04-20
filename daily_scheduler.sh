#!/bin/bash

# 激活虚拟环境
source /Volumes/T7\ Shield/quant_project/ma_agent_project/venv/bin/activate

# 进入项目目录
cd /Volumes/T7\ Shield/quant_project/ma_agent_project

# 运行策略更新脚本
python update_daily.py

# 可选：记录运行日志
echo "[$(date)] update_daily.py executed." >> daily_run.log