"""Comprehensive risk management system"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Order:
    symbol: str
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float

@dataclass
class RiskMetrics:
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    max_drawdown: float
    sharpe_ratio: float
    beta: float
    volatility: float
    concentration_risk: float

class RiskManager:
    """Multi-layered risk management system"""
    
    def __init__(self, config: Dict[str, Any]):
        # Risk limits from config
        self.max_position_size = config.get('max_position_pct', 0.05)  # 5% per position
        self.max_daily_loss = config.get('max_daily_loss', 0.02)       # 2% daily loss
        self.max_drawdown = config.get('max_drawdown', 0.10)           # 10% max drawdown
        self.max_leverage = config.get('max_leverage', 1.0)            # No leverage
        self.max_sector_concentration = config.get('max_sector_pct', 0.30)  # 30% per sector
        self.stop_loss_pct = config.get('stop_loss_pct', 0.03)         # 3% stop loss
        self.max_portfolio_correlation = config.get('max_correlation', 0.7)  # Max correlation
        
        # Portfolio state
        self.portfolio_value = config.get('initial_capital', 1000000)
        self.daily_start_value = self.portfolio_value
        self.current_positions: Dict[str, Position] = {}
        self.risk_alerts: List[Dict] = []
        
        # Risk tracking
        self.daily_returns: List[float] = []
        self.portfolio_history: List[float] = []
        
    def validate_order(self, order: Order, current_portfolio: Dict[str, Position]) -> Tuple[bool, List[str]]:
        """
        Comprehensive order validation with multiple risk checks
        
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Update current positions
        self.current_positions = current_portfolio
        
        # 1. Position size check
        if not self._check_position_size(order):
            violations.append(f"Position size exceeds {self.max_position_size*100}% limit")
        
        # 2. Daily loss limit check
        if not self._check_daily_loss_limit():
            violations.append(f"Daily loss limit of {self.max_daily_loss*100}% reached")
        
        # 3. Maximum drawdown check
        if not self._check_drawdown_limit():
            violations.append(f"Maximum drawdown of {self.max_drawdown*100}% reached")
        
        # 4. Sector concentration check
        if not self._check_sector_concentration(order):
            violations.append(f"Sector concentration exceeds {self.max_sector_concentration*100}% limit")
        
        # 5. Leverage check
        if not self._check_leverage():
            violations.append(f"Leverage exceeds {self.max_leverage} limit")
        
        # 6. Correlation check
        if not self._check_portfolio_correlation(order):
            violations.append(f"Portfolio correlation exceeds {self.max_portfolio_correlation} limit")
        
        # 7. Market hours check
        if not self._check_market_hours():
            violations.append("Trading outside market hours")
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def _check_position_size(self, order: Order) -> bool:
        """Check if position size is within limits"""
        if order.price is None:
            return True  # Cannot validate without price
        
        position_value = order.quantity * order.price
        max_position_value = self.portfolio_value * self.max_position_size
        
        # For existing positions, check the total position size
        if order.symbol in self.current_positions:
            current_position = self.current_positions[order.symbol]
            if order.order_type == OrderType.BUY:
                total_value = current_position.market_value + position_value
            else:
                total_value = max(0, current_position.market_value - position_value)
        else:
            total_value = position_value
        
        return total_value <= max_position_value
    
    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is reached"""
        current_loss = (self.portfolio_value - self.daily_start_value) / self.daily_start_value
        return current_loss >= -self.max_daily_loss
    
    def _check_drawdown_limit(self) -> bool:
        """Check if maximum drawdown limit is reached"""
        if not self.portfolio_history:
            return True
        
        peak = max(self.portfolio_history)
        current_drawdown = (peak - self.portfolio_value) / peak
        return current_drawdown <= self.max_drawdown
    
    def _check_sector_concentration(self, order: Order) -> bool:
        """Check sector concentration limits"""
        # This would need sector classification data
        # For now, implement basic industry grouping based on symbol
        sector = self._get_sector(order.symbol)
        
        sector_value = 0
        for symbol, position in self.current_positions.items():
            if self._get_sector(symbol) == sector:
                sector_value += position.market_value
        
        # Add order value if it's a buy
        if order.order_type == OrderType.BUY and order.price:
            sector_value += order.quantity * order.price
        
        sector_weight = sector_value / self.portfolio_value
        return sector_weight <= self.max_sector_concentration
    
    def _check_leverage(self) -> bool:
        """Check leverage limits"""
        total_position_value = sum(pos.market_value for pos in self.current_positions.values())
        leverage = total_position_value / self.portfolio_value
        return leverage <= self.max_leverage
    
    def _check_portfolio_correlation(self, order: Order) -> bool:
        """Check portfolio correlation limits"""
        # Simplified correlation check
        # In practice, would use historical correlation matrix
        if len(self.current_positions) < 10:  # Allow more concentration for smaller portfolios
            return True
        
        # Check if adding this symbol would create over-concentration
        if order.symbol in self.current_positions:
            return True
        
        return len(self.current_positions) < 20  # Limit number of positions
    
    def _check_market_hours(self) -> bool:
        """Check if trading is within market hours"""
        now = datetime.now()
        weekday = now.weekday()
        
        # Skip weekends
        if weekday >= 5:
            return False
        
        # A-Share market hours: 9:30-11:30, 13:00-15:00
        time = now.time()
        morning_start = datetime.strptime("09:30", "%H:%M").time()
        morning_end = datetime.strptime("11:30", "%H:%M").time()
        afternoon_start = datetime.strptime("13:00", "%H:%M").time()
        afternoon_end = datetime.strptime("15:00", "%H:%M").time()
        
        return (morning_start <= time <= morning_end) or (afternoon_start <= time <= afternoon_end)
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector classification for symbol"""
        # Simplified sector classification based on symbol prefix
        if symbol.startswith('60'):
            return 'large_cap'
        elif symbol.startswith('00'):
            return 'main_board'
        elif symbol.startswith('30'):
            return 'growth'
        else:
            return 'other'
    
    def calculate_position_size(self, signal_strength: float, volatility: float, 
                              symbol: str, price: float) -> int:
        """
        Calculate optimal position size using Kelly Criterion with modifications
        
        Args:
            signal_strength: Signal confidence (0-1)
            volatility: Historical volatility
            symbol: Stock symbol
            price: Current price
            
        Returns:
            Recommended position size in shares
        """
        # Kelly Criterion: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        
        # Convert signal strength to win probability
        win_prob = 0.5 + (signal_strength - 0.5) * 0.3  # Scale to 0.35-0.65 range
        loss_prob = 1 - win_prob
        
        # Assume average win/loss ratio based on volatility
        avg_win_loss_ratio = max(1.2, 2.0 - volatility)  # Higher vol = lower ratio
        
        # Kelly fraction
        kelly_fraction = (avg_win_loss_ratio * win_prob - loss_prob) / avg_win_loss_ratio
        
        # Apply conservative scaling (use 25% of Kelly)
        conservative_fraction = kelly_fraction * 0.25
        
        # Cap at maximum position size
        max_fraction = min(conservative_fraction, self.max_position_size)
        
        # Calculate position value and shares
        position_value = self.portfolio_value * max(0, max_fraction)
        position_shares = int(position_value / price)
        
        # Minimum lot size for A-shares is 100
        position_shares = (position_shares // 100) * 100
        
        return max(0, position_shares)
    
    def calculate_stop_loss_price(self, symbol: str, entry_price: float, 
                                 order_type: OrderType) -> float:
        """Calculate dynamic stop-loss price"""
        if order_type == OrderType.BUY:
            return entry_price * (1 - self.stop_loss_pct)
        else:  # SELL/Short
            return entry_price * (1 + self.stop_loss_pct)
    
    def update_portfolio_state(self, new_positions: Dict[str, Position], 
                              portfolio_value: float):
        """Update portfolio state for risk monitoring"""
        self.current_positions = new_positions
        
        # Calculate daily return
        if self.portfolio_value > 0:
            daily_return = (portfolio_value - self.portfolio_value) / self.portfolio_value
            self.daily_returns.append(daily_return)
        
        self.portfolio_value = portfolio_value
        self.portfolio_history.append(portfolio_value)
        
        # Check for risk alerts
        self._generate_risk_alerts()
    
    def _generate_risk_alerts(self):
        """Generate risk alerts based on current state"""
        alerts = []
        
        # Daily loss alert
        daily_loss = (self.portfolio_value - self.daily_start_value) / self.daily_start_value
        if daily_loss <= -self.max_daily_loss * 0.8:  # 80% of limit
            alerts.append({
                'level': RiskLevel.HIGH,
                'type': 'daily_loss',
                'message': f'Daily loss approaching limit: {daily_loss:.2%}',
                'timestamp': datetime.now()
            })
        
        # Drawdown alert
        if self.portfolio_history:
            peak = max(self.portfolio_history)
            drawdown = (peak - self.portfolio_value) / peak
            if drawdown >= self.max_drawdown * 0.8:  # 80% of limit
                alerts.append({
                    'level': RiskLevel.HIGH,
                    'type': 'drawdown',
                    'message': f'Drawdown approaching limit: {drawdown:.2%}',
                    'timestamp': datetime.now()
                })
        
        # Concentration alert
        if self.current_positions:
            max_position_weight = max(pos.weight for pos in self.current_positions.values())
            if max_position_weight >= self.max_position_size * 0.9:  # 90% of limit
                alerts.append({
                    'level': RiskLevel.MEDIUM,
                    'type': 'concentration',
                    'message': f'High position concentration: {max_position_weight:.2%}',
                    'timestamp': datetime.now()
                })
        
        self.risk_alerts.extend(alerts)
    
    def calculate_portfolio_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        if len(self.daily_returns) < 30:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)  # Insufficient data
        
        returns = np.array(self.daily_returns[-252:])  # Last year
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Maximum Drawdown
        portfolio_values = np.array(self.portfolio_history)
        peak = np.maximum.accumulate(portfolio_values)
        drawdowns = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdowns)
        
        # Sharpe Ratio
        risk_free_rate = 0.03 / 252  # 3% annual risk-free rate
        excess_returns = returns - risk_free_rate
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        # Beta (simplified, would need market index data)
        beta = 1.0  # Placeholder
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Concentration Risk
        if self.current_positions:
            weights = [pos.weight for pos in self.current_positions.values()]
            concentration_risk = max(weights) if weights else 0
        else:
            concentration_risk = 0
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            beta=beta,
            volatility=volatility,
            concentration_risk=concentration_risk
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        metrics = self.calculate_portfolio_risk_metrics()
        
        return {
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.portfolio_value - self.daily_start_value,
            'daily_pnl_pct': (self.portfolio_value - self.daily_start_value) / self.daily_start_value,
            'risk_metrics': {
                'var_95': metrics.var_95,
                'var_99': metrics.var_99,
                'max_drawdown': metrics.max_drawdown,
                'sharpe_ratio': metrics.sharpe_ratio,
                'volatility': metrics.volatility,
                'concentration_risk': metrics.concentration_risk
            },
            'risk_limits': {
                'max_position_size': self.max_position_size,
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown': self.max_drawdown,
                'max_leverage': self.max_leverage
            },
            'active_alerts': len([a for a in self.risk_alerts if a['timestamp'] > datetime.now() - timedelta(hours=24)]),
            'positions_count': len(self.current_positions),
            'total_exposure': sum(pos.market_value for pos in self.current_positions.values())
        }
    
    def reset_daily_tracking(self):
        """Reset daily tracking at market open"""
        self.daily_start_value = self.portfolio_value
        # Keep only recent alerts
        cutoff = datetime.now() - timedelta(days=7)
        self.risk_alerts = [alert for alert in self.risk_alerts if alert['timestamp'] > cutoff]