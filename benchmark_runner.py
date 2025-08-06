#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner for A-Share Quantitative Trading System
Tests multiple strategies and generates performance reports
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import our modules
from src.backtesting.engine import BacktestEngine, BacktestConfig, Strategy, BacktestResult
from src.strategies.indicators.technical import TechnicalIndicators
from src.strategies.indicators.sentiment import get_stock_sentiment_score
from signal_engine.fundamentals import get_fundamentals
from signal_engine.backtest import run_backtest as legacy_backtest
from signal_engine.score_and_suggest import score_stock

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RSIMACDBBStrategy(Strategy):
    """RSI + MACD + Bollinger Bands Strategy"""
    
    def __init__(self):
        self.name = "RSI_MACD_BB"
        self.lookback_period = 50
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, float]:
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < self.lookback_period:
                continue
                
            # Calculate indicators
            current_data = df[df.index <= date].tail(self.lookback_period)
            if len(current_data) < 20:
                continue
                
            try:
                # RSI Signal
                rsi = TechnicalIndicators.rsi(current_data['Close'])
                current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
                
                # MACD Signal
                macd_line, signal_line, histogram = TechnicalIndicators.macd(current_data['Close'])
                macd_signal = 1 if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0 else 0
                
                # Bollinger Bands Signal
                bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(current_data['Close'])
                current_price = current_data['Close'].iloc[-1]
                bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                
                # Combine signals
                signal_strength = 0.0
                
                # RSI component (buy when oversold, sell when overbought)
                if current_rsi < 30:
                    signal_strength += 0.4
                elif current_rsi > 70:
                    signal_strength -= 0.4
                else:
                    signal_strength += (50 - current_rsi) / 100  # Normalize to -0.2 to 0.2
                
                # MACD component
                if macd_signal > 0:
                    signal_strength += 0.3
                else:
                    signal_strength -= 0.1
                    
                # Bollinger Bands component
                if bb_position < 0.2:  # Near lower band - oversold
                    signal_strength += 0.3
                elif bb_position > 0.8:  # Near upper band - overbought  
                    signal_strength -= 0.3
                
                signals[symbol] = np.clip(signal_strength, -1.0, 1.0)
                
            except Exception as e:
                logger.warning(f"Error calculating signals for {symbol}: {e}")
                continue
                
        return signals
    
    def get_position_sizes(self, signals: Dict[str, float], portfolio_value: float) -> Dict[str, int]:
        # Filter for strong buy signals
        buy_signals = {k: v for k, v in signals.items() if v > 0.3}
        
        if not buy_signals:
            return {}
        
        # Equal weight allocation
        position_value_per_stock = (portfolio_value * 0.8) / len(buy_signals)  # 80% invested
        
        target_positions = {}
        for symbol, signal in buy_signals.items():
            # Assume average price of 50 for position sizing
            shares = int(position_value_per_stock / 50)
            shares = (shares // 100) * 100  # Round to lots of 100
            if shares > 0:
                target_positions[symbol] = shares
                
        return target_positions

class FundamentalStrategy(Strategy):
    """Fundamental Analysis Strategy (P/E, P/B, Market Cap)"""
    
    def __init__(self):
        self.name = "Fundamental"
        self.fundamental_cache = {}
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, float]:
        signals = {}
        
        for symbol in data.keys():
            try:
                # Get or cache fundamental data
                if symbol not in self.fundamental_cache:
                    fundamentals = get_fundamentals(symbol)
                    self.fundamental_cache[symbol] = fundamentals
                else:
                    fundamentals = self.fundamental_cache[symbol]
                
                pe = fundamentals.get("pe_ttm")
                pb = fundamentals.get("pb") 
                market_cap = fundamentals.get("market_cap")
                
                signal_strength = 0.0
                
                # P/E ratio scoring
                if pe is not None:
                    if pe < 10:
                        signal_strength += 0.4
                    elif pe < 20:
                        signal_strength += 0.2
                    elif pe > 30:
                        signal_strength -= 0.2
                
                # P/B ratio scoring
                if pb is not None:
                    if pb < 1:
                        signal_strength += 0.3
                    elif pb < 2:
                        signal_strength += 0.1
                    elif pb > 3:
                        signal_strength -= 0.2
                
                # Market cap scoring (favor mid-cap)
                if market_cap is not None:
                    if 100 <= market_cap <= 1000:  # Mid-cap sweet spot
                        signal_strength += 0.3
                    elif market_cap < 50:  # Small cap - higher risk
                        signal_strength += 0.1
                    elif market_cap > 5000:  # Large cap - lower growth
                        signal_strength -= 0.1
                
                signals[symbol] = np.clip(signal_strength, -1.0, 1.0)
                
            except Exception as e:
                logger.warning(f"Error getting fundamentals for {symbol}: {e}")
                continue
                
        return signals
    
    def get_position_sizes(self, signals: Dict[str, float], portfolio_value: float) -> Dict[str, int]:
        # Filter for positive signals
        buy_signals = {k: v for k, v in signals.items() if v > 0.1}
        
        if not buy_signals:
            return {}
        
        # Weight by signal strength
        total_signal = sum(buy_signals.values())
        target_positions = {}
        
        for symbol, signal in buy_signals.items():
            weight = signal / total_signal
            position_value = portfolio_value * 0.8 * weight  # 80% invested
            shares = int(position_value / 50)  # Assume average price
            shares = (shares // 100) * 100  # Round to lots
            if shares > 0:
                target_positions[symbol] = shares
                
        return target_positions

class SentimentStrategy(Strategy):
    """Sentiment Analysis Strategy using DeepSeek LLM"""
    
    def __init__(self):
        self.name = "Sentiment"
        self.sentiment_cache = {}
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, float]:
        signals = {}
        
        for symbol in data.keys():
            try:
                # Get sentiment score (cached to avoid API limits)
                if symbol not in self.sentiment_cache:
                    sentiment_score = get_stock_sentiment_score(symbol)
                    self.sentiment_cache[symbol] = sentiment_score
                else:
                    sentiment_score = self.sentiment_cache[symbol]
                
                # Convert sentiment to signal (-1 to 1 range)
                signals[symbol] = np.clip(sentiment_score, -1.0, 1.0)
                
            except Exception as e:
                logger.warning(f"Error getting sentiment for {symbol}: {e}")
                signals[symbol] = 0.0  # Neutral if error
                
        return signals
    
    def get_position_sizes(self, signals: Dict[str, float], portfolio_value: float) -> Dict[str, int]:
        # Filter for positive sentiment
        buy_signals = {k: v for k, v in signals.items() if v > 0.2}
        
        if not buy_signals:
            return {}
            
        # Equal weight for simplicity
        position_value_per_stock = (portfolio_value * 0.8) / len(buy_signals)
        target_positions = {}
        
        for symbol, signal in buy_signals.items():
            shares = int(position_value_per_stock / 50)
            shares = (shares // 100) * 100
            if shares > 0:
                target_positions[symbol] = shares
                
        return target_positions

class BenchmarkRunner:
    """Main benchmark runner class"""
    
    def __init__(self):
        self.strategies = {
            'RSI_MACD_BB': RSIMACDBBStrategy(),
            'Fundamental': FundamentalStrategy(), 
            'Sentiment': SentimentStrategy()
        }
        
        self.config = BacktestConfig(
            initial_capital=1000000,
            start_date="2023-01-01",
            end_date="2024-12-31",
            commission_rate=0.0003,
            slippage_rate=0.0005
        )
        
        self.results = {}
        
    def load_stock_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available stock data"""
        logger.info("Loading stock data...")
        
        data = {}
        data_dir = "data"
        
        if not os.path.exists(data_dir):
            logger.error("Data directory not found")
            return {}
            
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files")
        
        for file in csv_files[:50]:  # Limit for testing - remove in production
            symbol = file.replace('.csv', '')
            try:
                df = pd.read_csv(os.path.join(data_dir, file))
                # Standardize column names
                df.columns = [col.capitalize() for col in df.columns]
                
                # Ensure required columns exist
                required_cols = ['High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols):
                    # Set date index if Date column exists
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                    elif df.index.dtype != 'datetime64[ns]':
                        # Try to convert index to datetime
                        try:
                            df.index = pd.to_datetime(df.index)
                        except:
                            logger.warning(f"Could not set date index for {symbol}")
                            continue
                    
                    data[symbol] = df
                    
            except Exception as e:
                logger.warning(f"Error loading {symbol}: {e}")
                
        logger.info(f"Successfully loaded {len(data)} stocks")
        return data
    
    def run_strategy_backtest(self, strategy_name: str, strategy: Strategy, 
                            data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """Run backtest for a single strategy"""
        logger.info(f"Running backtest for {strategy_name}...")
        
        engine = BacktestEngine(self.config)
        engine.load_data(data)
        
        start_time = time.time()
        result = engine.run_backtest(strategy)
        end_time = time.time()
        
        logger.info(f"{strategy_name} backtest completed in {end_time - start_time:.2f} seconds")
        return result
    
    def calculate_additional_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate additional performance metrics"""
        returns = result.daily_returns
        
        if len(returns) == 0:
            return {}
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
        sortino_ratio = (result.annual_return - 0.03) / downside_deviation if downside_deviation > 0 else 0
        
        # Win Rate
        winning_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # Daily PnL Volatility
        daily_pnl_vol = returns.std() * np.sqrt(252)
        
        # Maximum consecutive losing days
        consecutive_losses = 0
        max_consecutive_losses = 0
        for ret in returns:
            if ret < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return {
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'daily_pnl_volatility': daily_pnl_vol,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_daily_return': returns.mean(),
            'skewness': returns.skew() if len(returns) > 3 else 0,
            'kurtosis': returns.kurtosis() if len(returns) > 3 else 0
        }
    
    def run_all_backtests(self) -> Dict[str, Dict]:
        """Run backtests for all strategies"""
        data = self.load_stock_data()
        
        if not data:
            logger.error("No data loaded - cannot run backtests")
            return {}
        
        results = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                backtest_result = self.run_strategy_backtest(strategy_name, strategy, data)
                additional_metrics = self.calculate_additional_metrics(backtest_result)
                
                # Combine all metrics
                strategy_results = {
                    'strategy_name': strategy_name,
                    'total_return': backtest_result.total_return,
                    'annual_return': backtest_result.annual_return,
                    'max_drawdown': backtest_result.max_drawdown,
                    'sharpe_ratio': backtest_result.sharpe_ratio,
                    'calmar_ratio': backtest_result.calmar_ratio,
                    'volatility': backtest_result.volatility,
                    'beta': backtest_result.beta,
                    'alpha': backtest_result.alpha,
                    'information_ratio': backtest_result.information_ratio,
                    'profit_factor': backtest_result.profit_factor,
                    'total_trades': len(backtest_result.trades),
                    **additional_metrics
                }
                
                results[strategy_name] = strategy_results
                
            except Exception as e:
                logger.error(f"Error running backtest for {strategy_name}: {e}")
                results[strategy_name] = {'error': str(e)}
        
        return results
    
    def generate_performance_report(self, results: Dict[str, Dict]) -> str:
        """Generate markdown performance report"""
        report = []
        report.append("# A-Share Quantitative Trading System - Backtest Results\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Period:** {self.config.start_date} to {self.config.end_date}")
        report.append(f"**Initial Capital:** ¥{self.config.initial_capital:,.0f}\n")
        
        # Summary Table
        report.append("## Performance Summary\n")
        report.append("| Strategy | Total Return | Annual Return | Sharpe Ratio | Max Drawdown | Win Rate | Sortino Ratio |")
        report.append("|----------|-------------|--------------|-------------|-------------|----------|--------------|")
        
        for strategy_name, metrics in results.items():
            if 'error' not in metrics:
                report.append(f"| {strategy_name} | {metrics.get('total_return', 0):.2%} | "
                            f"{metrics.get('annual_return', 0):.2%} | "
                            f"{metrics.get('sharpe_ratio', 0):.2f} | "
                            f"{metrics.get('max_drawdown', 0):.2%} | "
                            f"{metrics.get('win_rate', 0):.2%} | "
                            f"{metrics.get('sortino_ratio', 0):.2f} |")
        
        report.append("\n## Detailed Metrics\n")
        
        for strategy_name, metrics in results.items():
            if 'error' in metrics:
                report.append(f"### {strategy_name} (Error)\n")
                report.append(f"**Error:** {metrics['error']}\n")
                continue
                
            report.append(f"### {strategy_name}\n")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            
            metric_items = [
                ("Total Return", f"{metrics.get('total_return', 0):.2%}"),
                ("Annualized Return", f"{metrics.get('annual_return', 0):.2%}"),
                ("Volatility", f"{metrics.get('volatility', 0):.2%}"),
                ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"),
                ("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}"),
                ("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}"),
                ("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}"),
                ("Win Rate", f"{metrics.get('win_rate', 0):.2%}"),
                ("Daily P&L Volatility", f"{metrics.get('daily_pnl_volatility', 0):.2%}"),
                ("Beta", f"{metrics.get('beta', 0):.2f}"),
                ("Alpha", f"{metrics.get('alpha', 0):.4f}"),
                ("Information Ratio", f"{metrics.get('information_ratio', 0):.2f}"),
                ("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"),
                ("Total Trades", f"{metrics.get('total_trades', 0):,}"),
                ("Max Consecutive Losses", f"{metrics.get('max_consecutive_losses', 0)}"),
                ("Average Daily Return", f"{metrics.get('avg_daily_return', 0):.4%}"),
                ("Skewness", f"{metrics.get('skewness', 0):.2f}"),
                ("Kurtosis", f"{metrics.get('kurtosis', 0):.2f}")
            ]
            
            for metric_name, value in metric_items:
                report.append(f"| {metric_name} | {value} |")
            
            report.append("")
        
        # Strategy Rankings
        report.append("## Strategy Rankings\n")
        
        # Rank by Sharpe Ratio
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if valid_results:
            sharpe_ranking = sorted(valid_results.items(), 
                                  key=lambda x: x[1].get('sharpe_ratio', 0), reverse=True)
            
            report.append("### By Sharpe Ratio")
            for i, (strategy, metrics) in enumerate(sharpe_ranking, 1):
                report.append(f"{i}. **{strategy}**: {metrics.get('sharpe_ratio', 0):.2f}")
            
            # Rank by Total Return
            return_ranking = sorted(valid_results.items(),
                                  key=lambda x: x[1].get('total_return', 0), reverse=True)
            
            report.append("\n### By Total Return")
            for i, (strategy, metrics) in enumerate(return_ranking, 1):
                report.append(f"{i}. **{strategy}**: {metrics.get('total_return', 0):.2%}")
        
        # Risk Analysis
        report.append("\n## Risk Analysis\n")
        report.append("| Strategy | Max Drawdown | Daily Volatility | Beta | VaR (95%) |")
        report.append("|----------|-------------|-----------------|------|-----------|")
        
        for strategy_name, metrics in valid_results.items():
            var_95 = metrics.get('daily_pnl_volatility', 0) * -1.645  # Approximate VaR
            report.append(f"| {strategy_name} | "
                        f"{metrics.get('max_drawdown', 0):.2%} | "
                        f"{metrics.get('daily_pnl_volatility', 0):.2%} | "
                        f"{metrics.get('beta', 0):.2f} | "
                        f"{var_95:.2%} |")
        
        # Next Optimization Recommendations
        report.append("\n## Next Optimization Recommendations\n")
        recommendations = [
            "**Portfolio Optimization**: Implement Modern Portfolio Theory for optimal asset allocation",
            "**Dynamic Position Sizing**: Use Kelly Criterion or risk parity for position sizing",
            "**Risk Management**: Add stop-loss, take-profit, and position limits",
            "**Alternative Data**: Incorporate satellite imagery, social media sentiment, supply chain data",
            "**Machine Learning**: Implement ensemble methods, neural networks, reinforcement learning",
            "**High-Frequency Features**: Add intraday momentum, microstructure indicators",
            "**Regime Detection**: Identify market regimes and adapt strategies accordingly",
            "**Transaction Cost Analysis**: Model realistic execution costs and market impact",
            "**Multi-Asset**: Expand to bonds, commodities, forex for diversification",
            "**Real-Time Execution**: Implement streaming data processing and automated execution"
        ]
        
        for rec in recommendations:
            report.append(f"- {rec}")
        
        report.append(f"\n---\n*Report generated by A-Share Quantitative Trading System*")
        
        return "\n".join(report)

def main():
    """Main function to run all benchmarks"""
    logger.info("Starting comprehensive benchmark run...")
    
    runner = BenchmarkRunner()
    results = runner.run_all_backtests()
    
    if not results:
        logger.error("No results generated")
        return
    
    # Generate and save report
    report = runner.generate_performance_report(results)
    
    os.makedirs("benchmarks", exist_ok=True)
    report_path = "benchmarks/backtest_results.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save results as JSON for further processing
    json_path = "benchmarks/backtest_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Benchmark results saved to {report_path}")
    logger.info(f"Raw data saved to {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    for strategy_name, metrics in results.items():
        if 'error' in metrics:
            print(f"{strategy_name}: ERROR - {metrics['error']}")
        else:
            print(f"{strategy_name}:")
            print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            print()

if __name__ == "__main__":
    main()