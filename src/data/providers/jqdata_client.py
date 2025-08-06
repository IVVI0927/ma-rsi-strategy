"""JQData API client for A-Share market data"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
import jqdatasdk as jq
from datetime import datetime, timedelta
import logging
import time

from config import JQDATA_USER, JQDATA_PASS

logger = logging.getLogger(__name__)

class JQDataClient:
    """JQData API integration for A-Share market data"""
    
    def __init__(self):
        self.authenticated = False
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with JQData"""
        try:
            jq.auth(JQDATA_USER, JQDATA_PASS)
            self.authenticated = True
            logger.info("JQData authentication successful")
        except Exception as e:
            logger.error(f"JQData authentication failed: {e}")
            self.authenticated = False
    
    def get_stock_list(self, date: str = None) -> List[str]:
        """Get list of all A-Share stocks"""
        if not self.authenticated:
            raise Exception("JQData not authenticated")
        
        try:
            stocks = jq.get_all_securities(types=['stock'], date=date)
            return stocks.index.tolist()
        except Exception as e:
            logger.error(f"Error fetching stock list: {e}")
            return []
    
    def fetch_price_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV price data for multiple symbols"""
        if not self.authenticated:
            raise Exception("JQData not authenticated")
        
        data = {}
        for symbol in symbols:
            try:
                df = jq.get_price(
                    symbol,
                    start_date=start_date,
                    end_date=end_date,
                    frequency='daily',
                    fields=['open', 'close', 'low', 'high', 'volume', 'money']
                )
                df.columns = ['Open', 'Close', 'Low', 'High', 'Volume', 'Amount']
                data[symbol] = df
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
        
        return data
    
    def get_fundamentals(self, symbols: List[str], date: str = None) -> Dict[str, Dict]:
        """Get fundamental data for symbols"""
        if not self.authenticated:
            raise Exception("JQData not authenticated")
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        fundamentals = {}
        
        try:
            # Get valuation data
            q = jq.query(
                jq.valuation.code,
                jq.valuation.pe_ratio,
                jq.valuation.pb_ratio,
                jq.valuation.market_cap,
                jq.valuation.circulating_market_cap
            ).filter(jq.valuation.code.in_(symbols))
            
            valuation_data = jq.get_fundamentals(q, date)
            
            for _, row in valuation_data.iterrows():
                fundamentals[row['code']] = {
                    'pe_ttm': row['pe_ratio'],
                    'pb': row['pb_ratio'],
                    'market_cap': row['market_cap'],
                    'circulating_market_cap': row['circulating_market_cap']
                }
                
        except Exception as e:
            logger.error(f"Error fetching fundamentals: {e}")
        
        return fundamentals
    
    def get_financial_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get financial statement data"""
        if not self.authenticated:
            raise Exception("JQData not authenticated")
        
        try:
            # Get income statement
            q = jq.query(
                jq.income.code,
                jq.income.statDate,
                jq.income.total_operating_revenue,
                jq.income.net_profit,
                jq.income.operating_profit
            ).filter(
                jq.income.code == symbol
            ).filter(
                jq.income.statDate >= start_date
            ).filter(
                jq.income.statDate <= end_date
            )
            
            income_data = jq.get_fundamentals_continuously(q, end_date, 1000)
            return income_data
            
        except Exception as e:
            logger.error(f"Error fetching financial data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get index data (e.g., CSI300, SSE50)"""
        if not self.authenticated:
            raise Exception("JQData not authenticated")
        
        try:
            df = jq.get_price(
                index_code,
                start_date=start_date,
                end_date=end_date,
                frequency='daily',
                fields=['open', 'close', 'low', 'high', 'volume']
            )
            df.columns = ['Open', 'Close', 'Low', 'High', 'Volume']
            return df
        except Exception as e:
            logger.error(f"Error fetching index data for {index_code}: {e}")
            return pd.DataFrame()
    
    def health_check(self) -> bool:
        """Check if the data source is healthy"""
        try:
            # Try to fetch a simple query
            stocks = jq.get_all_securities(types=['stock'])
            return len(stocks) > 0
        except:
            return False