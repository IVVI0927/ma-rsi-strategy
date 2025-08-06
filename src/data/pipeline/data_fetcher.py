"""Data pipeline orchestrator with multi-source fallback"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import asyncio
import concurrent.futures
from dataclasses import dataclass

from src.data.providers.jqdata_client import JQDataClient
from src.data.providers.tushare_client import TushareClient
from src.data.storage.cache_manager import cache_manager, cached_function

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality metrics for validation"""
    missing_data_pct: float
    outlier_count: int
    data_freshness_hours: float
    consistency_score: float
    completeness_score: float

class DataPipeline:
    """Main data pipeline orchestrator with multi-source aggregation"""
    
    def __init__(self, tushare_token: str = None):
        self.primary_source = JQDataClient()
        self.backup_source = TushareClient(token=tushare_token)
        self.cache = cache_manager
        
        # Health status
        self.primary_healthy = False
        self.backup_healthy = False
        self._check_sources_health()
        
    def _check_sources_health(self):
        """Check health of data sources"""
        try:
            self.primary_healthy = self.primary_source.health_check()
            logger.info(f"Primary source (JQData) health: {self.primary_healthy}")
        except:
            self.primary_healthy = False
            
        try:
            self.backup_healthy = self.backup_source.health_check()
            logger.info(f"Backup source (Tushare) health: {self.backup_healthy}")
        except:
            self.backup_healthy = False
    
    @cached_function(ttl_hours=1)
    def get_market_data(self, symbols: Union[str, List[str]], start_date: str, end_date: str, 
                       use_backup: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data with automatic fallback
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_backup: Force use of backup source
            
        Returns:
            Dict mapping symbols to DataFrames with OHLCV data
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        logger.info(f"Fetching market data for {len(symbols)} symbols")
        
        # Try primary source first
        if not use_backup and self.primary_healthy:
            try:
                data = self.primary_source.fetch_price_data(symbols, start_date, end_date)
                if data:
                    logger.info(f"Successfully fetched {len(data)} symbols from primary source")
                    return self._validate_and_clean_data(data)
            except Exception as e:
                logger.error(f"Primary source failed: {e}")
                self.primary_healthy = False
        
        # Fallback to backup source
        if self.backup_healthy:
            try:
                logger.info("Falling back to backup source")
                data = self.backup_source.fetch_price_data(symbols, start_date, end_date)
                if data:
                    logger.info(f"Successfully fetched {len(data)} symbols from backup source")
                    return self._validate_and_clean_data(data)
            except Exception as e:
                logger.error(f"Backup source failed: {e}")
                self.backup_healthy = False
        
        # Both sources failed
        logger.error("All data sources failed")
        return {}
    
    def get_fundamentals_data(self, symbols: List[str], date: str = None) -> Dict[str, Dict]:
        """Get fundamental data with fallback"""
        
        # Try cache first
        cache_key = f"fundamentals_{','.join(symbols)}_{date}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Try primary source
        if self.primary_healthy:
            try:
                data = self.primary_source.get_fundamentals(symbols, date)
                if data:
                    self.cache.set(cache_key, data)
                    return data
            except Exception as e:
                logger.error(f"Primary source fundamentals failed: {e}")
        
        # Try backup source
        if self.backup_healthy:
            try:
                data = self.backup_source.get_fundamentals(symbols, date)
                if data:
                    self.cache.set(cache_key, data)
                    return data
            except Exception as e:
                logger.error(f"Backup source fundamentals failed: {e}")
        
        return {}
    
    def get_index_data(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get index data with fallback"""
        
        # Try cache first
        cache_key = f"index_{index_code}_{start_date}_{end_date}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None and not cached_data.empty:
            return cached_data
        
        # Try primary source
        if self.primary_healthy:
            try:
                data = self.primary_source.get_index_data(index_code, start_date, end_date)
                if not data.empty:
                    self.cache.set(cache_key, data)
                    return data
            except Exception as e:
                logger.error(f"Primary source index data failed: {e}")
        
        # Try backup source
        if self.backup_healthy:
            try:
                data = self.backup_source.get_index_data(index_code, start_date, end_date)
                if not data.empty:
                    self.cache.set(cache_key, data)
                    return data
            except Exception as e:
                logger.error(f"Backup source index data failed: {e}")
        
        return pd.DataFrame()
    
    def _validate_and_clean_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Validate and clean fetched data"""
        cleaned_data = {}
        
        for symbol, df in data.items():
            if df.empty:
                logger.warning(f"Empty data for {symbol}")
                continue
            
            try:
                # Basic validation
                if not self._validate_ohlcv_data(df):
                    logger.warning(f"Invalid OHLCV data for {symbol}")
                    continue
                
                # Clean data
                df_cleaned = self._clean_ohlcv_data(df)
                
                # Quality check
                quality = self._calculate_data_quality(df_cleaned)
                if quality.completeness_score < 0.8:
                    logger.warning(f"Low quality data for {symbol}: {quality.completeness_score}")
                
                cleaned_data[symbol] = df_cleaned
                
            except Exception as e:
                logger.error(f"Error validating data for {symbol}: {e}")
                continue
        
        return cleaned_data
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """Validate OHLCV data structure and content"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check required columns
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check for reasonable OHLC relationships
        if not ((df['High'] >= df['Low']) & 
                (df['High'] >= df['Open']) & 
                (df['High'] >= df['Close']) & 
                (df['Low'] <= df['Open']) & 
                (df['Low'] <= df['Close'])).all():
            return False
        
        # Check for positive values
        if not (df[['Open', 'High', 'Low', 'Close', 'Volume']] > 0).all().all():
            return False
        
        return True
    
    def _clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean OHLCV data"""
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Handle missing values
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df_clean.columns:
                # Forward fill then backward fill
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers (using IQR method)
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        # Ensure proper data types
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Calculate data quality metrics"""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100
        
        # Simple outlier detection (values beyond 3 standard deviations)
        numeric_df = df.select_dtypes(include=[np.number])
        outlier_count = 0
        for col in numeric_df.columns:
            z_scores = np.abs((numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std())
            outlier_count += (z_scores > 3).sum()
        
        # Data freshness (assuming index is datetime)
        if hasattr(df.index, 'max'):
            latest_date = df.index.max()
            if isinstance(latest_date, pd.Timestamp):
                freshness_hours = (datetime.now() - latest_date.to_pydatetime()).total_seconds() / 3600
            else:
                freshness_hours = 0
        else:
            freshness_hours = 0
        
        # Consistency score (OHLC relationships)
        consistency_violations = 0
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            consistency_violations = (~((df['High'] >= df['Low']) & 
                                       (df['High'] >= df['Open']) & 
                                       (df['High'] >= df['Close']) & 
                                       (df['Low'] <= df['Open']) & 
                                       (df['Low'] <= df['Close']))).sum()
        
        consistency_score = 1.0 - (consistency_violations / len(df)) if len(df) > 0 else 0.0
        completeness_score = 1.0 - (missing_pct / 100.0)
        
        return DataQualityMetrics(
            missing_data_pct=missing_pct,
            outlier_count=outlier_count,
            data_freshness_hours=freshness_hours,
            consistency_score=consistency_score,
            completeness_score=completeness_score
        )
    
    def batch_fetch_data(self, symbols: List[str], start_date: str, end_date: str, 
                        batch_size: int = 50, max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols in parallel batches"""
        
        results = {}
        symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(self.get_market_data, batch, start_date, end_date): batch
                for batch in symbol_batches
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_data = future.result()
                    results.update(batch_data)
                    logger.info(f"Completed batch of {len(batch)} symbols")
                except Exception as e:
                    logger.error(f"Batch failed: {e}")
        
        return results
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status and health metrics"""
        return {
            'primary_source_healthy': self.primary_healthy,
            'backup_source_healthy': self.backup_healthy,
            'cache_stats': self.cache.stats(),
            'last_health_check': datetime.now().isoformat(),
            'sources_available': {
                'jqdata': self.primary_healthy,
                'tushare': self.backup_healthy
            }
        }

# Global pipeline instance
data_pipeline = DataPipeline()