"""Broker interface and execution engine"""

import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import time
import random
import pandas as pd

from src.risk.risk_manager import Order, OrderType

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ExecutionType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

@dataclass
class ExecutionReport:
    order_id: str
    symbol: str
    side: str
    quantity: int
    filled_quantity: int
    remaining_quantity: int
    avg_fill_price: float
    status: OrderStatus
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class OrderRequest:
    symbol: str
    quantity: int
    side: str  # "buy" or "sell"
    order_type: ExecutionType = ExecutionType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class BrokerInterface(ABC):
    """Abstract interface for broker integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
        self.orders: Dict[str, ExecutionReport] = {}
        self.positions: Dict[str, float] = {}
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker API"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from broker API"""
        pass
    
    @abstractmethod
    def place_order(self, order_request: OrderRequest) -> ExecutionReport:
        """Place order with broker"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[ExecutionReport]:
        """Get order status"""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass
    
    @abstractmethod
    def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        pass

class PaperTradingBroker(BrokerInterface):
    """Paper trading implementation with realistic simulation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.initial_capital = config.get('initial_capital', 1000000)
        self.cash = self.initial_capital
        self.commission_rate = config.get('commission_rate', 0.0003)  # 3 basis points
        self.min_commission = config.get('min_commission', 5.0)
        self.slippage_rate = config.get('slippage_rate', 0.0005)  # 5 basis points
        
        # Simulation state
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}
        self.execution_delay = config.get('execution_delay_ms', 100)  # milliseconds
        
    def connect(self) -> bool:
        """Connect to paper trading system"""
        try:
            logger.info("Connecting to paper trading system")
            # Load market data for simulation
            self._load_market_data()
            self.connected = True
            logger.info("Paper trading connection established")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to paper trading: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from paper trading"""
        self.connected = False
        logger.info("Paper trading disconnected")
    
    def place_order(self, order_request: OrderRequest) -> ExecutionReport:
        """Place order with realistic simulation"""
        if not self.connected:
            return self._create_rejected_report(order_request, "Not connected")
        
        try:
            # Simulate execution delay
            if self.execution_delay > 0:
                time.sleep(self.execution_delay / 1000)
            
            # Get current market price
            market_price = self.get_market_price(order_request.symbol)
            if market_price is None:
                return self._create_rejected_report(order_request, "No market data")
            
            # Calculate execution price with slippage
            execution_price = self._calculate_execution_price(
                market_price, order_request.side, order_request.quantity
            )
            
            # Check if we can fill the order
            if not self._can_execute_order(order_request, execution_price):
                return self._create_rejected_report(order_request, "Insufficient funds/shares")
            
            # Execute the order
            commission = self._calculate_commission(order_request.quantity, execution_price)
            slippage = abs(execution_price - market_price) / market_price
            
            # Update positions and cash
            self._update_positions(order_request, execution_price, commission)
            
            # Create execution report
            execution_report = ExecutionReport(
                order_id=order_request.order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                filled_quantity=order_request.quantity,
                remaining_quantity=0,
                avg_fill_price=execution_price,
                status=OrderStatus.FILLED,
                timestamp=datetime.now(),
                commission=commission,
                slippage=slippage
            )
            
            self.orders[order_request.order_id] = execution_report
            
            logger.info(f"Order executed: {order_request.symbol} {order_request.side} "
                       f"{order_request.quantity} @ {execution_price:.2f}")
            
            return execution_report
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return self._create_rejected_report(order_request, str(e))
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order (simulation)"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Order {order_id} cancelled")
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[ExecutionReport]:
        """Get order status"""
        return self.orders.get(order_id)
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions"""
        return self.positions.copy()
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        total_value = self.cash
        
        # Add position values
        for symbol, quantity in self.positions.items():
            if quantity != 0:
                price = self.get_market_price(symbol)
                if price:
                    total_value += quantity * price
        
        return {
            'cash': self.cash,
            'total_value': total_value,
            'initial_capital': self.initial_capital,
            'pnl': total_value - self.initial_capital,
            'pnl_pct': (total_value - self.initial_capital) / self.initial_capital,
            'positions': self.positions,
            'orders_count': len(self.orders)
        }
    
    def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price (simulated)"""
        # In real implementation, this would connect to market data
        # For simulation, use stored prices or generate realistic prices
        
        if symbol in self.current_prices:
            # Add some random movement
            base_price = self.current_prices[symbol]
            volatility = 0.02  # 2% daily volatility
            random_change = random.gauss(0, volatility / (252**0.5))  # Daily change
            return base_price * (1 + random_change)
        
        # If no stored price, try to load from data
        return self._get_price_from_data(symbol)
    
    def _load_market_data(self):
        """Load market data for simulation"""
        # In production, this would load real market data
        # For demo, create some sample data
        import os
        data_dir = "data"
        
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.csv') and ('.SH' in filename or '.SZ' in filename):
                    symbol = filename.replace('.csv', '')
                    try:
                        df = pd.read_csv(os.path.join(data_dir, filename))
                        if not df.empty:
                            self.market_data[symbol] = df
                            # Store latest price
                            if 'Close' in df.columns:
                                self.current_prices[symbol] = float(df['Close'].iloc[-1])
                    except Exception as e:
                        logger.warning(f"Failed to load data for {symbol}: {e}")
    
    def _get_price_from_data(self, symbol: str) -> Optional[float]:
        """Get price from loaded data"""
        if symbol in self.market_data:
            df = self.market_data[symbol]
            if 'Close' in df.columns and not df.empty:
                return float(df['Close'].iloc[-1])
        return None
    
    def _calculate_execution_price(self, market_price: float, side: str, quantity: int) -> float:
        """Calculate execution price with slippage"""
        # Market impact based on order size (simplified)
        market_impact = min(0.001, (quantity / 100000) * 0.0001)  # Impact based on size
        
        # Random slippage
        random_slippage = random.gauss(0, self.slippage_rate)
        
        total_slippage = market_impact + abs(random_slippage)
        
        if side.lower() == 'buy':
            return market_price * (1 + total_slippage)
        else:
            return market_price * (1 - total_slippage)
    
    def _calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate trading commission"""
        trade_value = quantity * price
        commission = trade_value * self.commission_rate
        return max(commission, self.min_commission)
    
    def _can_execute_order(self, order_request: OrderRequest, execution_price: float) -> bool:
        """Check if order can be executed"""
        if order_request.side.lower() == 'buy':
            total_cost = order_request.quantity * execution_price
            commission = self._calculate_commission(order_request.quantity, execution_price)
            return self.cash >= (total_cost + commission)
        else:  # sell
            current_position = self.positions.get(order_request.symbol, 0)
            return current_position >= order_request.quantity
    
    def _update_positions(self, order_request: OrderRequest, execution_price: float, commission: float):
        """Update positions and cash after execution"""
        symbol = order_request.symbol
        quantity = order_request.quantity
        
        if order_request.side.lower() == 'buy':
            # Add to position
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            # Reduce cash
            self.cash -= (quantity * execution_price + commission)
        else:  # sell
            # Reduce position
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
            # Add cash
            self.cash += (quantity * execution_price - commission)
            
            # Remove zero positions
            if self.positions[symbol] == 0:
                del self.positions[symbol]
    
    def _create_rejected_report(self, order_request: OrderRequest, reason: str) -> ExecutionReport:
        """Create rejected execution report"""
        return ExecutionReport(
            order_id=order_request.order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            filled_quantity=0,
            remaining_quantity=order_request.quantity,
            avg_fill_price=0.0,
            status=OrderStatus.REJECTED,
            timestamp=datetime.now(),
            commission=0.0,
            slippage=0.0
        )

class ExecutionEngine:
    """Main execution engine coordinating orders and risk management"""
    
    def __init__(self, broker: BrokerInterface, risk_manager):
        self.broker = broker
        self.risk_manager = risk_manager
        self.order_history: List[ExecutionReport] = []
        
    def execute_orders(self, orders: List[OrderRequest]) -> List[ExecutionReport]:
        """Execute multiple orders with risk validation"""
        results = []
        
        for order_request in orders:
            # Convert to risk manager format
            risk_order = Order(
                symbol=order_request.symbol,
                quantity=order_request.quantity,
                order_type=OrderType.BUY if order_request.side.lower() == 'buy' else OrderType.SELL,
                price=order_request.price
            )
            
            # Risk validation
            is_valid, violations = self.risk_manager.validate_order(
                risk_order, 
                self._get_current_positions()
            )
            
            if not is_valid:
                logger.warning(f"Order rejected by risk manager: {violations}")
                execution_report = ExecutionReport(
                    order_id=order_request.order_id,
                    symbol=order_request.symbol,
                    side=order_request.side,
                    quantity=order_request.quantity,
                    filled_quantity=0,
                    remaining_quantity=order_request.quantity,
                    avg_fill_price=0.0,
                    status=OrderStatus.REJECTED,
                    timestamp=datetime.now(),
                    commission=0.0,
                    slippage=0.0
                )
                results.append(execution_report)
                continue
            
            # Execute order
            execution_report = self.broker.place_order(order_request)
            results.append(execution_report)
            self.order_history.append(execution_report)
        
        return results
    
    def _get_current_positions(self) -> Dict[str, Any]:
        """Get current positions in risk manager format"""
        broker_positions = self.broker.get_positions()
        positions = {}
        
        for symbol, quantity in broker_positions.items():
            if quantity != 0:
                current_price = self.broker.get_market_price(symbol) or 0
                market_value = quantity * current_price
                
                positions[symbol] = type('Position', (), {
                    'symbol': symbol,
                    'quantity': quantity,
                    'market_value': market_value,
                    'weight': market_value / self.risk_manager.portfolio_value if self.risk_manager.portfolio_value > 0 else 0
                })()
        
        return positions
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.order_history:
            return {'total_orders': 0}
        
        filled_orders = [o for o in self.order_history if o.status == OrderStatus.FILLED]
        rejected_orders = [o for o in self.order_history if o.status == OrderStatus.REJECTED]
        
        avg_slippage = np.mean([o.slippage for o in filled_orders]) if filled_orders else 0
        avg_commission = np.mean([o.commission for o in filled_orders]) if filled_orders else 0
        
        return {
            'total_orders': len(self.order_history),
            'filled_orders': len(filled_orders),
            'rejected_orders': len(rejected_orders),
            'fill_rate': len(filled_orders) / len(self.order_history),
            'avg_slippage': avg_slippage,
            'avg_commission': avg_commission,
            'total_commission': sum(o.commission for o in filled_orders)
        }