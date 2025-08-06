"""Main A-Share Quantitative Trading System Integration"""

import os
import sys
import asyncio
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import all system components
from src.data.pipeline.data_fetcher import DataPipeline
from src.risk.risk_manager import RiskManager, Order, OrderType
from src.execution.broker_interface import PaperTradingBroker, ExecutionEngine, OrderRequest
from src.strategies.indicators.technical import TechnicalIndicators
from src.strategies.indicators.sentiment import SentimentAggregator, DeepSeekSentimentAnalyzer
from src.backtesting.engine import BacktestEngine, BacktestConfig, Strategy
from src.security.security_manager import SecurityManager
from src.data.storage.market_data_store import MarketDataStore
from src.monitoring.dashboard import TradingMonitor

logger = logging.getLogger(__name__)

class ComprehensiveStrategy(Strategy):
    """Comprehensive multi-factor strategy combining technical, fundamental, and sentiment analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.technical_indicators = TechnicalIndicators()
        
        # Initialize sentiment analyzer if available
        try:
            self.sentiment_analyzer = DeepSeekSentimentAnalyzer()
            self.sentiment_aggregator = SentimentAggregator(self.sentiment_analyzer)
        except Exception as e:
            logger.warning(f"Sentiment analysis not available: {e}")
            self.sentiment_aggregator = None
        
        # Strategy parameters
        self.lookback_period = config.get('lookback_period', 20)
        self.technical_weight = config.get('technical_weight', 0.4)
        self.fundamental_weight = config.get('fundamental_weight', 0.4)
        self.sentiment_weight = config.get('sentiment_weight', 0.2)
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, float]:
        """Generate comprehensive trading signals"""
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < self.lookback_period or date not in df.index:
                continue
            
            try:
                # Get data up to current date
                historical_data = df[df.index <= date].tail(self.lookback_period * 2)
                
                if len(historical_data) < self.lookback_period:
                    continue
                
                # Technical analysis signals
                tech_signal = self._generate_technical_signal(historical_data, date)
                
                # Fundamental analysis (simplified)
                fundamental_signal = self._generate_fundamental_signal(symbol, historical_data)
                
                # Sentiment analysis
                sentiment_signal = self._generate_sentiment_signal(symbol)
                
                # Combine signals
                combined_signal = (
                    tech_signal * self.technical_weight +
                    fundamental_signal * self.fundamental_weight +
                    sentiment_signal * self.sentiment_weight
                )
                
                signals[symbol] = np.clip(combined_signal, -1.0, 1.0)
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _generate_technical_signal(self, df: pd.DataFrame, date: datetime) -> float:
        """Generate technical analysis signal"""
        try:
            # RSI signal
            rsi = self.technical_indicators.rsi(df['Close'])
            current_rsi = rsi.loc[date] if date in rsi.index else 50
            
            rsi_signal = 0
            if current_rsi < 30:
                rsi_signal = 0.5  # Oversold - buy signal
            elif current_rsi > 70:
                rsi_signal = -0.5  # Overbought - sell signal
            
            # MACD signal
            macd, signal_line, histogram = self.technical_indicators.macd(df['Close'])
            if date in macd.index and date in signal_line.index:
                macd_signal = 0.3 if macd.loc[date] > signal_line.loc[date] else -0.3
            else:
                macd_signal = 0
            
            # Bollinger Bands signal
            bb_upper, bb_middle, bb_lower = self.technical_indicators.bollinger_bands(df['Close'])
            current_price = df['Close'].loc[date] if date in df.index else 0
            
            bb_signal = 0
            if date in bb_lower.index and current_price < bb_lower.loc[date]:
                bb_signal = 0.4  # Below lower band - buy signal
            elif date in bb_upper.index and current_price > bb_upper.loc[date]:
                bb_signal = -0.4  # Above upper band - sell signal
            
            # Moving average signal
            sma_20 = df['Close'].rolling(20).mean()
            sma_50 = df['Close'].rolling(50).mean()
            
            ma_signal = 0
            if date in sma_20.index and date in sma_50.index:
                if sma_20.loc[date] > sma_50.loc[date]:
                    ma_signal = 0.2  # Golden cross
                elif sma_20.loc[date] < sma_50.loc[date]:
                    ma_signal = -0.2  # Death cross
            
            # Combine technical signals
            total_signal = rsi_signal + macd_signal + bb_signal + ma_signal
            return np.clip(total_signal, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Technical signal error: {e}")
            return 0.0
    
    def _generate_fundamental_signal(self, symbol: str, df: pd.DataFrame) -> float:
        """Generate fundamental analysis signal (simplified)"""
        try:
            # Price-based fundamental signals
            current_price = df['Close'].iloc[-1]
            avg_price_20 = df['Close'].rolling(20).mean().iloc[-1]
            avg_price_60 = df['Close'].rolling(60).mean().iloc[-1]
            
            # Price momentum
            price_momentum = (current_price - avg_price_60) / avg_price_60
            
            # Volume analysis
            current_volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1] if 'Volume' in df.columns else 1
            
            volume_signal = 0.2 if current_volume > avg_volume * 1.5 else 0
            
            # Combine fundamental signals
            fundamental_signal = np.clip(price_momentum * 0.5 + volume_signal, -1.0, 1.0)
            
            return fundamental_signal
            
        except Exception as e:
            logger.error(f"Fundamental signal error for {symbol}: {e}")
            return 0.0
    
    def _generate_sentiment_signal(self, symbol: str) -> float:
        """Generate sentiment analysis signal"""
        if self.sentiment_aggregator is None:
            return 0.0
        
        try:
            sentiment_data = self.sentiment_aggregator.get_stock_sentiment(symbol)
            sentiment_score = sentiment_data.get('sentiment_score', 0.0)
            confidence = sentiment_data.get('confidence', 0.0)
            
            # Weight sentiment by confidence
            return sentiment_score * confidence
            
        except Exception as e:
            logger.error(f"Sentiment signal error for {symbol}: {e}")
            return 0.0
    
    def get_position_sizes(self, signals: Dict[str, float], portfolio_value: float) -> Dict[str, int]:
        """Convert signals to position sizes"""
        target_positions = {}
        
        # Filter signals above threshold
        min_signal_strength = 0.3
        strong_signals = {k: v for k, v in signals.items() if abs(v) >= min_signal_strength}
        
        if not strong_signals:
            return target_positions
        
        # Sort signals by strength
        sorted_signals = sorted(strong_signals.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Take top N positions
        max_positions = min(20, len(sorted_signals))
        top_signals = dict(sorted_signals[:max_positions])
        
        # Calculate position sizes
        total_weight = 0.9  # Use 90% of portfolio
        position_weight = total_weight / len(top_signals)
        
        for symbol, signal in top_signals.items():
            if signal > 0:  # Only long positions for now
                position_value = portfolio_value * position_weight * abs(signal)
                # Assume average price of 50 for position sizing
                estimated_price = 50
                shares = int(position_value / estimated_price)
                # Round to nearest 100 (lot size)
                shares = (shares // 100) * 100
                if shares > 0:
                    target_positions[symbol] = shares
        
        return target_positions

class AShareTradingSystem:
    """Main A-Share Quantitative Trading System"""
    
    def __init__(self, config_path: str = "config/trading_config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Initialize all components
        self.data_pipeline = DataPipeline()
        self.market_data_store = MarketDataStore()
        
        self.risk_manager = RiskManager(self.config.get('risk', {}))
        
        self.broker = PaperTradingBroker(self.config.get('execution', {}))
        self.execution_engine = ExecutionEngine(self.broker, self.risk_manager)
        
        self.security_manager = SecurityManager()
        
        self.strategy = ComprehensiveStrategy(self.config.get('strategy', {}))
        
        # System state
        self.is_running = False
        self.trading_mode = False
        self.current_positions = {}
        
        logger.info("A-Share Trading System initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default system configuration"""
        return {
            'trading': {
                'initial_capital': 1000000,
                'commission': {'rate': 0.0003, 'minimum': 5.0},
                'slippage': {'rate': 0.0005}
            },
            'risk': {
                'max_position_pct': 0.1,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.20,
                'stop_loss_pct': 0.03
            },
            'strategy': {
                'lookback_period': 20,
                'technical_weight': 0.4,
                'fundamental_weight': 0.4,
                'sentiment_weight': 0.2
            },
            'execution': {
                'initial_capital': 1000000,
                'commission_rate': 0.0003,
                'slippage_rate': 0.0005,
                'min_commission': 5.0
            }
        }
    
    def _setup_logging(self):
        """Setup system logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading_system.log'),
                logging.StreamHandler()
            ]
        )
    
    async def start_system(self):
        """Start the trading system"""
        try:
            logger.info("Starting A-Share Trading System")
            
            # Connect to data sources
            await self._initialize_data_pipeline()
            
            # Connect broker
            self.broker.connect()
            
            # Start monitoring
            self.is_running = True
            
            # Start main trading loop
            await self._main_trading_loop()
            
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            await self.stop_system()
    
    async def _initialize_data_pipeline(self):
        """Initialize data pipeline and load initial data"""
        try:
            # Check data sources health
            pipeline_status = self.data_pipeline.get_pipeline_status()
            
            if not (pipeline_status['primary_source_healthy'] or 
                   pipeline_status['backup_source_healthy']):
                raise Exception("No healthy data sources available")
            
            logger.info("Data pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Data pipeline initialization failed: {e}")
            raise
    
    async def _main_trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                # Check if we're in trading hours
                if not self._is_trading_hours():
                    await asyncio.sleep(60)  # Check every minute
                    continue
                
                # Generate trading signals
                signals = await self._generate_trading_signals()
                
                if signals:
                    # Execute trading strategy
                    await self._execute_strategy(signals)
                
                # Risk monitoring
                self._monitor_risk()
                
                # Wait before next iteration
                await asyncio.sleep(30)  # 30 second intervals
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        now = datetime.now()
        weekday = now.weekday()
        
        # Skip weekends
        if weekday >= 5:
            return False
        
        # A-Share trading hours
        time = now.time()
        morning_start = datetime.strptime("09:30", "%H:%M").time()
        morning_end = datetime.strptime("11:30", "%H:%M").time()
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("15:00", "%H:%M").time()
        
        return (morning_start <= time <= morning_end) or (afternoon_start <= time <= afternoon_end)
    
    async def _generate_trading_signals(self) -> Dict[str, float]:
        """Generate trading signals using the strategy"""
        try:
            # Get available symbols
            symbols = self.market_data_store.get_available_symbols()[:50]  # Limit for demo
            
            if not symbols:
                logger.warning("No symbols available for trading")
                return {}
            
            # Load recent data for symbols
            current_data = {}
            for symbol in symbols:
                df = self.market_data_store.load_symbol_data(
                    symbol,
                    start_date=datetime.now() - timedelta(days=100)
                )
                if df is not None and len(df) > 20:
                    current_data[symbol] = df
            
            if not current_data:
                logger.warning("No data available for signal generation")
                return {}
            
            # Generate signals
            signals = self.strategy.generate_signals(current_data, datetime.now())
            
            logger.info(f"Generated {len(signals)} trading signals")
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return {}
    
    async def _execute_strategy(self, signals: Dict[str, float]):
        """Execute trading strategy based on signals"""
        try:
            # Get current portfolio value
            account_info = self.broker.get_account_info()
            portfolio_value = account_info.get('total_value', 0)
            
            # Convert signals to position sizes
            target_positions = self.strategy.get_position_sizes(signals, portfolio_value)
            
            if not target_positions:
                logger.info("No positions to execute")
                return
            
            # Generate orders
            orders = []
            current_positions = self.broker.get_positions()
            
            for symbol, target_qty in target_positions.items():
                current_qty = current_positions.get(symbol, 0)
                qty_diff = target_qty - current_qty
                
                if abs(qty_diff) >= 100:  # Minimum lot size
                    order = OrderRequest(
                        symbol=symbol,
                        quantity=abs(qty_diff),
                        side="buy" if qty_diff > 0 else "sell",
                        price=None  # Market order
                    )
                    orders.append(order)
            
            # Execute orders
            if orders:
                execution_reports = self.execution_engine.execute_orders(orders)
                
                successful_orders = [r for r in execution_reports if r.status.value == "filled"]
                logger.info(f"Executed {len(successful_orders)}/{len(orders)} orders")
            
        except Exception as e:
            logger.error(f"Strategy execution error: {e}")
    
    def _monitor_risk(self):
        """Monitor risk metrics and alerts"""
        try:
            # Update risk manager with current state
            current_positions = self.broker.get_positions()
            account_info = self.broker.get_account_info()
            
            # Get risk summary
            risk_summary = self.risk_manager.get_risk_summary()
            
            # Check for risk alerts
            if risk_summary.get('active_alerts', 0) > 0:
                logger.warning(f"Risk alerts active: {risk_summary['active_alerts']}")
            
            # Log daily P&L
            daily_pnl_pct = risk_summary.get('daily_pnl_pct', 0)
            if abs(daily_pnl_pct) > 0.02:  # 2% threshold
                logger.info(f"Daily P&L: {daily_pnl_pct:.2%}")
            
        except Exception as e:
            logger.error(f"Risk monitoring error: {e}")
    
    def enable_trading_mode(self):
        """Enable secure air-gapped trading mode"""
        try:
            success = self.security_manager.enable_trading_mode()
            if success:
                self.trading_mode = True
                logger.info("Trading mode enabled - system is air-gapped")
            else:
                logger.error("Failed to enable trading mode")
        except Exception as e:
            logger.error(f"Error enabling trading mode: {e}")
    
    def disable_trading_mode(self):
        """Disable trading mode"""
        try:
            success = self.security_manager.disable_trading_mode()
            if success:
                self.trading_mode = False
                logger.info("Trading mode disabled - normal operations restored")
        except Exception as e:
            logger.error(f"Error disabling trading mode: {e}")
    
    def run_backtest(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run comprehensive backtest"""
        try:
            config = BacktestConfig(
                initial_capital=self.config['trading']['initial_capital'],
                start_date=start_date,
                end_date=end_date,
                commission_rate=self.config['trading']['commission']['rate']
            )
            
            backtest_engine = BacktestEngine(config)
            
            # Load data for backtest
            symbols = self.market_data_store.get_available_symbols()[:20]  # Limit for demo
            data = {}
            
            for symbol in symbols:
                df = self.market_data_store.load_symbol_data(
                    symbol,
                    start_date=datetime.strptime(start_date, '%Y-%m-%d'),
                    end_date=datetime.strptime(end_date, '%Y-%m-%d')
                )
                if df is not None and len(df) > 50:
                    data[symbol] = df
            
            if not data:
                raise Exception("No data available for backtest")
            
            backtest_engine.load_data(data)
            result = backtest_engine.run_backtest(self.strategy)
            
            logger.info(f"Backtest completed: {result.total_return:.2%} total return")
            
            return {
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'win_rate': result.win_rate,
                'total_trades': len(result.trades)
            }
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return {'error': str(e)}
    
    async def stop_system(self):
        """Stop the trading system"""
        try:
            logger.info("Stopping A-Share Trading System")
            
            self.is_running = False
            
            # Disconnect broker
            if hasattr(self.broker, 'disconnect'):
                self.broker.disconnect()
            
            # Disable trading mode if active
            if self.trading_mode:
                self.disable_trading_mode()
            
            logger.info("System stopped successfully")
            
        except Exception as e:
            logger.error(f"System shutdown error: {e}")

# CLI interface
async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='A-Share Quantitative Trading System')
    parser.add_argument('--config', default='config/trading_config.yaml', help='Config file path')
    parser.add_argument('--backtest', action='store_true', help='Run backtest mode')
    parser.add_argument('--start-date', default='2023-01-01', help='Backtest start date')
    parser.add_argument('--end-date', default='2023-12-31', help='Backtest end date')
    parser.add_argument('--trading-mode', action='store_true', help='Enable secure trading mode')
    
    args = parser.parse_args()
    
    # Create trading system
    system = AShareTradingSystem(args.config)
    
    try:
        if args.backtest:
            # Run backtest
            result = system.run_backtest(args.start_date, args.end_date)
            print("Backtest Results:")
            for key, value in result.items():
                if isinstance(value, float):
                    if 'return' in key or 'drawdown' in key:
                        print(f"  {key}: {value:.2%}")
                    else:
                        print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        else:
            # Enable trading mode if requested
            if args.trading_mode:
                system.enable_trading_mode()
            
            # Start live trading system
            await system.start_system()
    
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, stopping system...")
        await system.stop_system()
    except Exception as e:
        print(f"System error: {e}")
        await system.stop_system()

if __name__ == "__main__":
    asyncio.run(main())