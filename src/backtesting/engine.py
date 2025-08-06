"""Comprehensive backtesting engine for A-Share strategies"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from src.risk.risk_manager import RiskManager, Order, OrderType, Position
from src.execution.broker_interface import PaperTradingBroker, OrderRequest, ExecutionType

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    initial_capital: float = 1000000
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0005
    min_commission: float = 5.0
    benchmark: str = "000300.XSHG"  # CSI 300
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    max_positions: int = 20
    
@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    side: str  # buy/sell
    pnl: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    hold_days: Optional[int] = None

@dataclass
class BacktestResult:
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    trades: List[Trade] = field(default_factory=list)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    portfolio_values: pd.Series = field(default_factory=pd.Series)
    benchmark_returns: pd.Series = field(default_factory=pd.Series)
    positions_history: Dict[str, pd.Series] = field(default_factory=dict)

class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, float]:
        """
        Generate trading signals
        
        Args:
            data: Dictionary of DataFrames with market data
            date: Current date
            
        Returns:
            Dictionary mapping symbols to signal strength (-1 to 1)
        """
        pass
    
    @abstractmethod
    def get_position_sizes(self, signals: Dict[str, float], portfolio_value: float) -> Dict[str, int]:
        """
        Convert signals to position sizes
        
        Args:
            signals: Trading signals
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary mapping symbols to position sizes (shares)
        """
        pass

class BacktestEngine:
    """Comprehensive backtesting engine with realistic execution simulation"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        
        # Initialize components
        self.risk_manager = RiskManager({
            'initial_capital': config.initial_capital,
            'max_position_pct': 0.1,  # 10% max per position
            'max_daily_loss': 0.05,   # 5% max daily loss
            'max_drawdown': 0.20,     # 20% max drawdown
            'commission_rate': config.commission_rate
        })
        
        self.broker = PaperTradingBroker({
            'initial_capital': config.initial_capital,
            'commission_rate': config.commission_rate,
            'slippage_rate': config.slippage_rate,
            'min_commission': config.min_commission
        })
        
        # Backtest state
        self.current_positions: Dict[str, int] = {}
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.trades: List[Trade] = []
        self.daily_values: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        
    def load_data(self, data: Dict[str, pd.DataFrame], benchmark_data: pd.DataFrame = None):
        """Load market data for backtesting"""
        self.data = data
        self.benchmark_data = benchmark_data
        
        # Ensure all data has the same date range
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        for symbol in self.data:
            df = self.data[symbol]
            df.index = pd.to_datetime(df.index)
            self.data[symbol] = df[(df.index >= start_date) & (df.index <= end_date)]
    
    def run_backtest(self, strategy: Strategy) -> BacktestResult:
        """Run complete backtest"""
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Reset state
        self._reset_state()
        
        # Get trading dates
        trading_dates = self._get_trading_dates()
        
        # Connect broker
        self.broker.connect()
        
        # Run backtest day by day
        for i, date in enumerate(trading_dates):
            try:
                self._process_trading_day(strategy, date, i)
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                continue
        
        # Calculate final results
        result = self._calculate_results()
        
        logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
        return result
    
    def _reset_state(self):
        """Reset backtest state"""
        self.current_positions = {}
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.trades = []
        self.daily_values = []
        self.daily_returns = []
        self.broker.cash = self.config.initial_capital
        self.broker.positions = {}
    
    def _get_trading_dates(self) -> List[datetime]:
        """Get list of trading dates"""
        if not self.data:
            return []
        
        # Get all unique dates from loaded data
        all_dates = set()
        for symbol, df in self.data.items():
            all_dates.update(df.index)
        
        # Filter by date range and sort
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        trading_dates = sorted([d for d in all_dates if start_date <= d <= end_date])
        return trading_dates
    
    def _process_trading_day(self, strategy: Strategy, date: datetime, day_index: int):
        """Process a single trading day"""
        
        # Check if we should rebalance
        should_rebalance = self._should_rebalance(date, day_index)
        
        if should_rebalance:
            # Get current market data
            current_data = self._get_current_data(date)
            if not current_data:
                return
            
            # Generate signals
            signals = strategy.generate_signals(current_data, date)
            
            # Convert signals to target positions
            target_positions = strategy.get_position_sizes(signals, self.portfolio_value)
            
            # Execute rebalancing
            self._rebalance_portfolio(target_positions, date, current_data)
        
        # Update portfolio value
        self._update_portfolio_value(date)
        
        # Update risk manager
        self._update_risk_manager(date)
    
    def _should_rebalance(self, date: datetime, day_index: int) -> bool:
        """Determine if portfolio should be rebalanced"""
        if self.config.rebalance_frequency == "daily":
            return True
        elif self.config.rebalance_frequency == "weekly":
            return date.weekday() == 0  # Monday
        elif self.config.rebalance_frequency == "monthly":
            return date.day <= 7 and date.weekday() == 0  # First Monday of month
        else:
            return False
    
    def _get_current_data(self, date: datetime) -> Dict[str, pd.DataFrame]:
        """Get market data available at current date"""
        current_data = {}
        
        for symbol, df in self.data.items():
            if date in df.index:
                # Include data up to current date
                historical_data = df[df.index <= date].copy()
                if len(historical_data) > 0:
                    current_data[symbol] = historical_data
        
        return current_data
    
    def _rebalance_portfolio(self, target_positions: Dict[str, int], 
                           date: datetime, current_data: Dict[str, pd.DataFrame]):
        """Rebalance portfolio to target positions"""
        
        # Calculate position changes needed
        orders = []
        
        for symbol, target_qty in target_positions.items():
            current_qty = self.current_positions.get(symbol, 0)
            qty_diff = target_qty - current_qty
            
            if qty_diff != 0 and symbol in current_data:
                # Get current price
                current_price = current_data[symbol].loc[date, 'Close']
                
                # Create order request
                order_request = OrderRequest(
                    symbol=symbol,
                    quantity=abs(qty_diff),
                    side="buy" if qty_diff > 0 else "sell",
                    order_type=ExecutionType.MARKET,
                    price=current_price
                )
                
                orders.append(order_request)
        
        # Execute orders through broker
        if orders:
            execution_reports = []
            for order_request in orders:
                report = self.broker.place_order(order_request)
                execution_reports.append(report)
                
                # Update positions and create trade records
                if report.status.value == "filled":
                    self._record_trade(order_request, report, date)
                    
                    # Update current positions
                    symbol = order_request.symbol
                    if order_request.side == "buy":
                        self.current_positions[symbol] = self.current_positions.get(symbol, 0) + order_request.quantity
                    else:
                        self.current_positions[symbol] = max(0, self.current_positions.get(symbol, 0) - order_request.quantity)
                        if self.current_positions[symbol] == 0:
                            del self.current_positions[symbol]
    
    def _record_trade(self, order_request: OrderRequest, execution_report, date: datetime):
        """Record executed trade"""
        trade = Trade(
            symbol=order_request.symbol,
            entry_date=date,
            exit_date=None,  # Will be set when position is closed
            entry_price=execution_report.avg_fill_price,
            exit_price=None,
            quantity=order_request.quantity,
            side=order_request.side,
            commission=execution_report.commission,
            slippage=execution_report.slippage
        )
        
        self.trades.append(trade)
    
    def _update_portfolio_value(self, date: datetime):
        """Update portfolio value based on current positions"""
        total_value = self.broker.cash
        
        # Add value of all positions
        for symbol, quantity in self.current_positions.items():
            if symbol in self.data and date in self.data[symbol].index:
                current_price = self.data[symbol].loc[date, 'Close']
                position_value = quantity * current_price
                total_value += position_value
        
        # Record daily value and return
        self.daily_values.append((date, total_value))
        
        if len(self.daily_values) > 1:
            prev_value = self.daily_values[-2][1]
            daily_return = (total_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
        
        self.portfolio_value = total_value
    
    def _update_risk_manager(self, date: datetime):
        """Update risk manager with current state"""
        # Convert positions to risk manager format
        risk_positions = {}
        
        for symbol, quantity in self.current_positions.items():
            if symbol in self.data and date in self.data[symbol].index:
                current_price = self.data[symbol].loc[date, 'Close']
                market_value = quantity * current_price
                weight = market_value / self.portfolio_value if self.portfolio_value > 0 else 0
                
                risk_positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=current_price,  # Simplified
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=0,  # Simplified
                    weight=weight
                )
        
        self.risk_manager.update_portfolio_state(risk_positions, self.portfolio_value)
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        
        if len(self.daily_values) < 2:
            return self._empty_result()
        
        # Convert to pandas Series
        dates, values = zip(*self.daily_values)
        portfolio_series = pd.Series(values, index=dates)
        returns_series = pd.Series(self.daily_returns, index=dates[1:])
        
        # Calculate benchmark returns if available
        benchmark_returns = self._calculate_benchmark_returns()
        
        # Basic performance metrics
        total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
        
        # Annualized return
        days = (dates[-1] - dates[0]).days
        annual_return = (1 + total_return) ** (365.25 / days) - 1
        
        # Maximum drawdown
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Volatility
        volatility = returns_series.std() * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.03  # 3% risk-free rate
        excess_returns = returns_series - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade analysis
        completed_trades = [t for t in self.trades if t.exit_date is not None]
        win_rate = self._calculate_win_rate(completed_trades)
        profit_factor = self._calculate_profit_factor(completed_trades)
        
        # Beta and Alpha (if benchmark available)
        beta, alpha, information_ratio = self._calculate_risk_metrics(returns_series, benchmark_returns)
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            volatility=volatility,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            trades=self.trades,
            daily_returns=returns_series,
            portfolio_values=portfolio_series,
            benchmark_returns=benchmark_returns
        )
    
    def _calculate_benchmark_returns(self) -> Optional[pd.Series]:
        """Calculate benchmark returns"""
        if self.benchmark_data is None or self.benchmark_data.empty:
            return None
        
        # Align benchmark data with portfolio dates
        dates = [d for d, _ in self.daily_values]
        benchmark_aligned = self.benchmark_data.reindex(dates, method='ffill')
        
        if 'Close' in benchmark_aligned.columns:
            benchmark_returns = benchmark_aligned['Close'].pct_change().dropna()
            return benchmark_returns
        
        return None
    
    def _calculate_win_rate(self, trades: List[Trade]) -> float:
        """Calculate win rate from completed trades"""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for t in trades if t.pnl and t.pnl > 0)
        return winning_trades / len(trades)
    
    def _calculate_profit_factor(self, trades: List[Trade]) -> float:
        """Calculate profit factor"""
        if not trades:
            return 0.0
        
        gross_profit = sum(t.pnl for t in trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl and t.pnl < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_risk_metrics(self, returns: pd.Series, 
                               benchmark_returns: Optional[pd.Series]) -> Tuple[float, float, float]:
        """Calculate beta, alpha, and information ratio"""
        if benchmark_returns is None or benchmark_returns.empty:
            return 0.0, 0.0, 0.0
        
        # Align returns
        aligned_returns = returns.align(benchmark_returns, join='inner')
        portfolio_returns, bench_returns = aligned_returns
        
        if len(portfolio_returns) < 30:  # Need sufficient data
            return 0.0, 0.0, 0.0
        
        # Beta
        covariance = np.cov(portfolio_returns, bench_returns)[0, 1]
        benchmark_variance = np.var(bench_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha
        risk_free_rate = 0.03 / 252  # Daily risk-free rate
        portfolio_excess = portfolio_returns.mean() - risk_free_rate
        benchmark_excess = bench_returns.mean() - risk_free_rate
        alpha = portfolio_excess - beta * benchmark_excess
        
        # Information ratio
        tracking_error = (portfolio_returns - bench_returns).std()
        information_ratio = (portfolio_returns.mean() - bench_returns.mean()) / tracking_error if tracking_error > 0 else 0
        
        return beta, alpha, information_ratio
    
    def _empty_result(self) -> BacktestResult:
        """Return empty result when backtest fails"""
        return BacktestResult(
            total_return=0.0,
            annual_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            volatility=0.0,
            beta=0.0,
            alpha=0.0,
            information_ratio=0.0
        )

class SimpleStrategy(Strategy):
    """Simple example strategy for testing"""
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], date: datetime) -> Dict[str, float]:
        """Generate simple mean reversion signals"""
        signals = {}
        
        for symbol, df in data.items():
            if len(df) < self.lookback:
                continue
            
            # Calculate simple moving average
            sma = df['Close'].rolling(self.lookback).mean()
            current_price = df.loc[date, 'Close']
            current_sma = sma.loc[date]
            
            # Mean reversion signal
            if pd.notna(current_sma):
                deviation = (current_price - current_sma) / current_sma
                # Signal strength based on deviation (reversed for mean reversion)
                signal = -np.clip(deviation * 2, -1, 1)
                signals[symbol] = signal
        
        return signals
    
    def get_position_sizes(self, signals: Dict[str, float], portfolio_value: float) -> Dict[str, int]:
        """Convert signals to equal-weighted positions"""
        target_positions = {}
        
        # Filter strong signals
        strong_signals = {k: v for k, v in signals.items() if abs(v) > 0.3}
        
        if not strong_signals:
            return target_positions
        
        # Equal weight allocation
        target_weight_per_position = 0.8 / len(strong_signals)  # 80% invested
        
        for symbol, signal in strong_signals.items():
            # Only take long positions for simplicity
            if signal > 0:
                position_value = portfolio_value * target_weight_per_position
                # Get current price (simplified)
                target_positions[symbol] = int(position_value / 100)  # Simplified price assumption
        
        return target_positions