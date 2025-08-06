"""Tushare API client as backup data source"""

import tushare as ts
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class TushareClient:
    """Tushare API integration as backup data source"""
    
    def __init__(self, token: str = None):
        self.token = token
        if token:
            ts.set_token(token)
        self.pro = ts.pro_api()
        
    def get_stock_list(self, exchange: str = None) -> List[str]:
        """Get list of A-Share stocks"""
        try:
            if exchange:
                df = self.pro.stock_basic(exchange=exchange, list_status='L')
            else:
                df = self.pro.stock_basic(list_status='L')
            
            # Convert to JQData format
            stocks = []
            for _, row in df.iterrows():
                if row['exchange'] == 'SSE':
                    stocks.append(f"{row['ts_code'][:6]}.SH")
                elif row['exchange'] == 'SZSE':
                    stocks.append(f"{row['ts_code'][:6]}.SZ")
            
            return stocks
        except Exception as e:
            logger.error(f"Error fetching stock list: {e}")
            return []
    
    def fetch_price_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV price data for multiple symbols"""
        data = {}
        
        for symbol in symbols:
            try:
                # Convert JQData format to Tushare format
                ts_code = self._convert_symbol_format(symbol)
                
                df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', '')
                )
                
                if not df.empty:
                    df = df.sort_values('trade_date')
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df.set_index('trade_date', inplace=True)
                    
                    # Rename columns to match JQData format
                    df = df.rename(columns={
                        'open': 'Open',
                        'close': 'Close',
                        'low': 'Low',
                        'high': 'High',
                        'vol': 'Volume',
                        'amount': 'Amount'
                    })
                    
                    data[symbol] = df[['Open', 'Close', 'Low', 'High', 'Volume', 'Amount']]
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
        
        return data
    
    def get_fundamentals(self, symbols: List[str], date: str = None) -> Dict[str, Dict]:
        """Get fundamental data for symbols"""
        fundamentals = {}
        
        for symbol in symbols:
            try:
                ts_code = self._convert_symbol_format(symbol)
                
                # Get basic info
                basic_info = self.pro.daily_basic(
                    ts_code=ts_code,
                    trade_date=date.replace('-', '') if date else None,
                    fields='ts_code,trade_date,pe_ttm,pb,total_mv,circ_mv'
                )
                
                if not basic_info.empty:
                    row = basic_info.iloc[0]
                    fundamentals[symbol] = {
                        'pe_ttm': row['pe_ttm'],
                        'pb': row['pb'],
                        'market_cap': row['total_mv'],
                        'circulating_market_cap': row['circ_mv']
                    }
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching fundamentals for {symbol}: {e}")
                continue
        
        return fundamentals
    
    def get_financial_data(self, symbol: str, period: str = None) -> pd.DataFrame:
        """Get financial statement data"""
        try:
            ts_code = self._convert_symbol_format(symbol)
            
            # Get income statement
            income = self.pro.income(ts_code=ts_code, period=period)
            
            if not income.empty:
                income['end_date'] = pd.to_datetime(income['end_date'])
                return income.sort_values('end_date')
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching financial data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get index data"""
        try:
            # Map common indices
            index_mapping = {
                '000300.XSHG': '399300.SZ',  # CSI300
                '000016.XSHG': '000016.SH',  # SSE50
                '000001.XSHG': '000001.SH',  # SSE Composite
            }
            
            ts_code = index_mapping.get(index_code, index_code)
            
            df = self.pro.index_daily(
                ts_code=ts_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            
            if not df.empty:
                df = df.sort_values('trade_date')
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                
                df = df.rename(columns={
                    'open': 'Open',
                    'close': 'Close',
                    'low': 'Low',
                    'high': 'High',
                    'vol': 'Volume'
                })
                
                return df[['Open', 'Close', 'Low', 'High', 'Volume']]
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching index data for {index_code}: {e}")
            return pd.DataFrame()
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert JQData format to Tushare format"""
        # Convert 000001.SZ to 000001.SZ (already correct)
        # Convert 600000.SH to 600000.SH (already correct)
        if '.SH' in symbol or '.SZ' in symbol:
            return symbol
        return symbol
    
    def health_check(self) -> bool:
        """Check if the data source is healthy"""
        try:
            df = self.pro.stock_basic(list_status='L')
            return len(df) > 0
        except:
            return False