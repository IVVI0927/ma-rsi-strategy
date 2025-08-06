"""FastAPI-based real-time monitoring dashboard"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import asdict
import uvicorn

from src.risk.risk_manager import RiskManager
from src.execution.broker_interface import PaperTradingBroker
from src.data.pipeline.data_fetcher import data_pipeline
from signal_engine.score_and_suggest import get_today_scores

logger = logging.getLogger(__name__)

app = FastAPI(title="A-Share Quant Trading Monitor", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected: {len(self.active_connections)} total connections")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected: {len(self.active_connections)} total connections")
    
    async def broadcast_message(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message, default=str)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

class TradingMonitor:
    """Main trading system monitor"""
    
    def __init__(self):
        # Initialize components (in production, these would be injected)
        self.risk_manager = RiskManager({
            'initial_capital': 1000000,
            'max_position_pct': 0.1,
            'max_daily_loss': 0.05,
            'max_drawdown': 0.20
        })
        
        self.broker = PaperTradingBroker({
            'initial_capital': 1000000,
            'commission_rate': 0.0003,
            'slippage_rate': 0.0005
        })
        
        # Monitoring state
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.portfolio_value = 1000000
        self.active_strategies = ["multi_factor", "mean_reversion"]
        self.system_status = "healthy"
        self.last_update = datetime.now()
        
        # Performance tracking
        self.performance_history = []
        self.trade_history = []
        
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        
        # Portfolio overview
        account_info = self.broker.get_account_info() if hasattr(self.broker, 'get_account_info') else {}
        
        # Risk metrics
        risk_summary = self.risk_manager.get_risk_summary()
        
        # Today's recommendations
        try:
            recommendations = get_today_scores()[:10]  # Top 10
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            recommendations = []
        
        # Market data status
        pipeline_status = data_pipeline.get_pipeline_status()
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # System health
        system_health = self._get_system_health()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio': {
                'total_value': self.portfolio_value,
                'cash': account_info.get('cash', 0),
                'daily_pnl': self.daily_pnl,
                'daily_pnl_pct': (self.daily_pnl / self.portfolio_value) * 100 if self.portfolio_value > 0 else 0,
                'positions_count': len(self.current_positions),
                'active_strategies': self.active_strategies
            },
            'risk': {
                'metrics': risk_summary.get('risk_metrics', {}),
                'limits': risk_summary.get('risk_limits', {}),
                'alerts': risk_summary.get('active_alerts', 0),
                'exposure': risk_summary.get('total_exposure', 0)
            },
            'recommendations': recommendations,
            'data_pipeline': pipeline_status,
            'performance': performance_metrics,
            'system': system_health
        }
    
    async def get_positions_data(self) -> List[Dict]:
        """Get current positions data"""
        positions_data = []
        
        for symbol, quantity in self.current_positions.items():
            try:
                # Get current price (mock for now)
                current_price = np.random.uniform(10, 100)  # Mock price
                market_value = quantity * current_price
                
                positions_data.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'current_price': round(current_price, 2),
                    'market_value': round(market_value, 2),
                    'weight': round((market_value / self.portfolio_value) * 100, 2) if self.portfolio_value > 0 else 0,
                    'unrealized_pnl': round(np.random.uniform(-1000, 1000), 2),  # Mock PnL
                    'unrealized_pnl_pct': round(np.random.uniform(-5, 5), 2)     # Mock PnL%
                })
            except Exception as e:
                logger.error(f"Error getting position data for {symbol}: {e}")
        
        return positions_data
    
    async def get_trades_data(self, limit: int = 50) -> List[Dict]:
        """Get recent trades data"""
        # Mock trade data for demo
        trades = []
        for i in range(min(limit, len(self.trade_history))):
            trade = {
                'id': f'T{i:06d}',
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                'symbol': f'00000{i % 10}.SZ',
                'side': 'buy' if i % 2 == 0 else 'sell',
                'quantity': np.random.randint(100, 1000) * 100,
                'price': round(np.random.uniform(10, 100), 2),
                'commission': round(np.random.uniform(5, 50), 2),
                'status': 'filled'
            }
            trades.append(trade)
        
        return trades
    
    async def get_performance_chart_data(self, days: int = 30) -> Dict[str, List]:
        """Get performance chart data"""
        # Generate mock performance data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Simulate portfolio performance
        returns = np.random.normal(0.001, 0.02, days)  # Daily returns
        portfolio_values = [1000000]  # Starting value
        
        for ret in returns:
            new_value = portfolio_values[-1] * (1 + ret)
            portfolio_values.append(new_value)
        
        portfolio_values = portfolio_values[1:]  # Remove initial value
        
        # Simulate benchmark (CSI 300)
        benchmark_returns = np.random.normal(0.0005, 0.018, days)
        benchmark_values = [1000000]
        
        for ret in benchmark_returns:
            new_value = benchmark_values[-1] * (1 + ret)
            benchmark_values.append(new_value)
        
        benchmark_values = benchmark_values[1:]
        
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'portfolio': [round(v, 0) for v in portfolio_values],
            'benchmark': [round(v, 0) for v in benchmark_values],
            'drawdown': [round((max(portfolio_values[:i+1]) - v) / max(portfolio_values[:i+1]) * 100, 2) 
                        for i, v in enumerate(portfolio_values)]
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate current performance metrics"""
        # Mock metrics for demo
        return {
            'total_return_pct': round(np.random.uniform(-5, 15), 2),
            'annual_return_pct': round(np.random.uniform(-10, 25), 2),
            'max_drawdown_pct': round(np.random.uniform(0, 15), 2),
            'sharpe_ratio': round(np.random.uniform(0.5, 2.5), 2),
            'win_rate_pct': round(np.random.uniform(45, 65), 2),
            'profit_factor': round(np.random.uniform(1.1, 2.5), 2),
            'volatility_pct': round(np.random.uniform(10, 25), 2)
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        return {
            'status': self.system_status,
            'uptime_hours': round((datetime.now() - self.last_update).total_seconds() / 3600, 1),
            'memory_usage_pct': np.random.uniform(30, 70),
            'cpu_usage_pct': np.random.uniform(10, 50),
            'data_freshness_minutes': np.random.randint(1, 30),
            'api_latency_ms': np.random.randint(50, 200),
            'error_rate_pct': round(np.random.uniform(0, 2), 2)
        }

# Initialize global monitor
monitor = TradingMonitor()
manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the main dashboard HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>A-Share Quant Trading Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f5f5f5; 
            }
            .header { 
                background: white; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                margin-bottom: 20px; 
            }
            .metrics-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin-bottom: 20px; 
            }
            .metric-card { 
                background: white; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            }
            .metric-title { 
                font-size: 14px; 
                color: #666; 
                margin-bottom: 5px; 
            }
            .metric-value { 
                font-size: 24px; 
                font-weight: bold; 
                margin-bottom: 10px; 
            }
            .status-indicator { 
                display: inline-block; 
                width: 10px; 
                height: 10px; 
                border-radius: 50%; 
                margin-right: 8px; 
            }
            .status-healthy { background-color: #4CAF50; }
            .status-warning { background-color: #FF9800; }
            .status-error { background-color: #F44336; }
            .recommendations { 
                background: white; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            }
            .recommendation-item { 
                display: flex; 
                justify-content: space-between; 
                padding: 10px 0; 
                border-bottom: 1px solid #eee; 
            }
            .last-update { 
                color: #666; 
                font-size: 12px; 
                text-align: right; 
                margin-top: 10px; 
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏦 A-Share Quantitative Trading System</h1>
            <p><span class="status-indicator status-healthy"></span>System Status: <span id="system-status">Healthy</span></p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Portfolio Value</div>
                <div class="metric-value" id="portfolio-value">¥1,000,000</div>
                <div style="color: #4CAF50;">Daily P&L: <span id="daily-pnl">+¥0</span> (<span id="daily-pnl-pct">0.00%</span>)</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Active Positions</div>
                <div class="metric-value" id="positions-count">0</div>
                <div style="color: #666;">Cash: <span id="cash">¥1,000,000</span></div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Risk Metrics</div>
                <div>Max Drawdown: <span id="max-drawdown">0.00%</span></div>
                <div>Sharpe Ratio: <span id="sharpe-ratio">0.00</span></div>
                <div>VaR (95%): <span id="var-95">0.00%</span></div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">System Health</div>
                <div>Data Pipeline: <span id="data-status">Healthy</span></div>
                <div>API Latency: <span id="api-latency">0ms</span></div>
                <div>Uptime: <span id="uptime">0h</span></div>
            </div>
        </div>
        
        <div class="recommendations">
            <h3>📈 Today's Top Recommendations</h3>
            <div id="recommendations-list">
                <div style="text-align: center; color: #666; padding: 20px;">
                    Loading recommendations...
                </div>
            </div>
            <div class="last-update">
                Last updated: <span id="last-update">--</span>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket("ws://localhost:8000/ws/monitor");
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                document.getElementById('system-status').textContent = 'Connection Error';
                document.querySelector('.status-indicator').className = 'status-indicator status-error';
            };
            
            function updateDashboard(data) {
                // Update portfolio metrics
                document.getElementById('portfolio-value').textContent = 
                    '¥' + data.portfolio.total_value.toLocaleString();
                document.getElementById('daily-pnl').textContent = 
                    (data.portfolio.daily_pnl >= 0 ? '+' : '') + '¥' + data.portfolio.daily_pnl.toLocaleString();
                document.getElementById('daily-pnl-pct').textContent = 
                    data.portfolio.daily_pnl_pct.toFixed(2) + '%';
                document.getElementById('positions-count').textContent = 
                    data.portfolio.positions_count;
                document.getElementById('cash').textContent = 
                    '¥' + data.portfolio.cash.toLocaleString();
                
                // Update risk metrics
                const risk = data.risk.metrics;
                document.getElementById('max-drawdown').textContent = 
                    (risk.max_drawdown * 100).toFixed(2) + '%';
                document.getElementById('sharpe-ratio').textContent = 
                    risk.sharpe_ratio.toFixed(2);
                document.getElementById('var-95').textContent = 
                    (risk.var_95 * 100).toFixed(2) + '%';
                
                // Update system health
                const system = data.system;
                document.getElementById('data-status').textContent = 
                    system.status === 'healthy' ? 'Healthy' : 'Warning';
                document.getElementById('api-latency').textContent = 
                    system.api_latency_ms + 'ms';
                document.getElementById('uptime').textContent = 
                    system.uptime_hours.toFixed(1) + 'h';
                
                // Update recommendations
                updateRecommendations(data.recommendations);
                
                // Update timestamp
                document.getElementById('last-update').textContent = 
                    new Date(data.timestamp).toLocaleString();
            }
            
            function updateRecommendations(recommendations) {
                const container = document.getElementById('recommendations-list');
                if (!recommendations || recommendations.length === 0) {
                    container.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">No recommendations available</div>';
                    return;
                }
                
                container.innerHTML = recommendations.slice(0, 5).map(rec => 
                    `<div class="recommendation-item">
                        <div>
                            <strong>${rec.code}</strong>
                            <div style="font-size: 12px; color: #666;">Score: ${rec.score}</div>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: ${rec.suggest === '✅ BUY' ? '#4CAF50' : '#666'}">${rec.suggest}</span>
                        </div>
                    </div>`
                ).join('');
            }
            
            // Refresh dashboard every 5 seconds if WebSocket fails
            setInterval(() => {
                if (ws.readyState !== WebSocket.OPEN) {
                    location.reload();
                }
            }, 5000);
        </script>
    </body>
    </html>
    """

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Send dashboard data every second
            dashboard_data = await monitor.get_dashboard_data()
            await websocket.send_text(json.dumps(dashboard_data, default=str))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/dashboard")
async def get_dashboard_api():
    """REST API endpoint for dashboard data"""
    return await monitor.get_dashboard_data()

@app.get("/api/positions")
async def get_positions():
    """Get current positions"""
    return await monitor.get_positions_data()

@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get recent trades"""
    return await monitor.get_trades_data(limit)

@app.get("/api/performance")
async def get_performance(days: int = 30):
    """Get performance chart data"""
    return await monitor.get_performance_chart_data(days)

@app.get("/api/recommendations")
async def get_recommendations_api():
    """Get today's stock recommendations"""
    try:
        recommendations = get_today_scores()
        return {"recommendations": recommendations[:20], "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Error fetching recommendations")

@app.get("/api/risk-summary")
async def get_risk_summary():
    """Get risk management summary"""
    return monitor.risk_manager.get_risk_summary()

@app.get("/api/system-status")  
async def get_system_status():
    """Get system health status"""
    pipeline_status = data_pipeline.get_pipeline_status()
    system_health = monitor._get_system_health()
    
    return {
        "system": system_health,
        "data_pipeline": pipeline_status,
        "timestamp": datetime.now()
    }

# Background task to broadcast updates
async def broadcast_updates():
    """Background task to broadcast dashboard updates"""
    while True:
        try:
            dashboard_data = await monitor.get_dashboard_data()
            await manager.broadcast_message(dashboard_data)
        except Exception as e:
            logger.error(f"Error broadcasting updates: {e}")
        
        await asyncio.sleep(2)  # Update every 2 seconds

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(broadcast_updates())
    logger.info("Monitoring dashboard started")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)