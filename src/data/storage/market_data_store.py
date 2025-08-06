"""Optimized market data storage using SQLite and HDF5"""

import sqlite3
import pandas as pd
import numpy as np
import h5py
import os
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
import threading
from contextlib import contextmanager
import pickle
import zlib
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataMetadata:
    symbol: str
    start_date: datetime
    end_date: datetime
    record_count: int
    columns: List[str]
    file_size_mb: float
    last_updated: datetime
    data_source: str
    compression_ratio: float = 1.0

class SQLiteManager:
    """SQLite database manager for metadata and small datasets"""
    
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.connection_pool = {}
        self.lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            # Metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_metadata (
                    symbol TEXT PRIMARY KEY,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    columns TEXT NOT NULL,
                    file_size_mb REAL NOT NULL,
                    last_updated TEXT NOT NULL,
                    data_source TEXT NOT NULL,
                    compression_ratio REAL DEFAULT 1.0,
                    storage_type TEXT DEFAULT 'hdf5'
                )
            """)
            
            # Index table for fast lookups
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_index (
                    symbol TEXT,
                    date TEXT,
                    open_price REAL,
                    close_price REAL,
                    high_price REAL,
                    low_price REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_data_index_symbol ON data_index(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_data_index_date ON data_index(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_updated ON data_metadata(last_updated)")
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with connection pooling"""
        thread_id = threading.get_ident()
        
        with self.lock:
            if thread_id not in self.connection_pool:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                # Optimize SQLite settings
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=MEMORY")
                self.connection_pool[thread_id] = conn
        
        try:
            yield self.connection_pool[thread_id]
        except Exception as e:
            self.connection_pool[thread_id].rollback()
            raise
    
    def store_metadata(self, metadata: DataMetadata):
        """Store data metadata"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO data_metadata 
                (symbol, start_date, end_date, record_count, columns, file_size_mb, 
                 last_updated, data_source, compression_ratio, storage_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.symbol,
                metadata.start_date.isoformat(),
                metadata.end_date.isoformat(),
                metadata.record_count,
                ','.join(metadata.columns),
                metadata.file_size_mb,
                metadata.last_updated.isoformat(),
                metadata.data_source,
                metadata.compression_ratio,
                'hdf5'
            ))
            conn.commit()
    
    def get_metadata(self, symbol: str = None) -> Union[DataMetadata, List[DataMetadata]]:
        """Get data metadata"""
        with self.get_connection() as conn:
            if symbol:
                cursor = conn.execute(
                    "SELECT * FROM data_metadata WHERE symbol = ?", (symbol,)
                )
                row = cursor.fetchone()
                if row:
                    return DataMetadata(
                        symbol=row['symbol'],
                        start_date=datetime.fromisoformat(row['start_date']),
                        end_date=datetime.fromisoformat(row['end_date']),
                        record_count=row['record_count'],
                        columns=row['columns'].split(','),
                        file_size_mb=row['file_size_mb'],
                        last_updated=datetime.fromisoformat(row['last_updated']),
                        data_source=row['data_source'],
                        compression_ratio=row['compression_ratio']
                    )
                return None
            else:
                cursor = conn.execute("SELECT * FROM data_metadata ORDER BY symbol")
                rows = cursor.fetchall()
                return [
                    DataMetadata(
                        symbol=row['symbol'],
                        start_date=datetime.fromisoformat(row['start_date']),
                        end_date=datetime.fromisoformat(row['end_date']),
                        record_count=row['record_count'],
                        columns=row['columns'].split(','),
                        file_size_mb=row['file_size_mb'],
                        last_updated=datetime.fromisoformat(row['last_updated']),
                        data_source=row['data_source'],
                        compression_ratio=row['compression_ratio']
                    ) for row in rows
                ]
    
    def update_index(self, symbol: str, df: pd.DataFrame):
        """Update search index with OHLCV data"""
        with self.get_connection() as conn:
            # Clear existing data for symbol
            conn.execute("DELETE FROM data_index WHERE symbol = ?", (symbol,))
            
            # Insert new data
            for date, row in df.iterrows():
                conn.execute("""
                    INSERT INTO data_index 
                    (symbol, date, open_price, close_price, high_price, low_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    date.strftime('%Y-%m-%d'),
                    float(row.get('Open', 0)),
                    float(row.get('Close', 0)),
                    float(row.get('High', 0)),
                    float(row.get('Low', 0)),
                    int(row.get('Volume', 0))
                ))
            
            conn.commit()
    
    def search_symbols(self, criteria: Dict[str, Any]) -> List[str]:
        """Search symbols based on criteria"""
        conditions = []
        params = []
        
        if 'min_price' in criteria:
            conditions.append("close_price >= ?")
            params.append(criteria['min_price'])
        
        if 'max_price' in criteria:
            conditions.append("close_price <= ?")
            params.append(criteria['max_price'])
        
        if 'min_volume' in criteria:
            conditions.append("volume >= ?")
            params.append(criteria['min_volume'])
        
        if 'date_range' in criteria:
            start_date, end_date = criteria['date_range']
            conditions.append("date BETWEEN ? AND ?")
            params.extend([start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
        
        query = "SELECT DISTINCT symbol FROM data_index"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return [row[0] for row in cursor.fetchall()]

class HDF5Manager:
    """HDF5 manager for large time series data"""
    
    def __init__(self, data_dir: str = "data/hdf5"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.file_cache = {}
        self.lock = threading.RLock()
    
    def _get_file_path(self, symbol: str) -> str:
        """Get HDF5 file path for symbol"""
        # Group symbols into files by first character to avoid too many files
        prefix = symbol[0] if symbol else 'other'
        return os.path.join(self.data_dir, f"market_data_{prefix}.h5")
    
    @contextmanager
    def _get_hdf5_file(self, symbol: str, mode: str = 'r'):
        """Get HDF5 file handle with caching"""
        file_path = self._get_file_path(symbol)
        
        try:
            with h5py.File(file_path, mode) as f:
                yield f
        except Exception as e:
            logger.error(f"HDF5 file error for {symbol}: {e}")
            raise
    
    def store_data(self, symbol: str, df: pd.DataFrame, compression: str = 'gzip') -> float:
        """Store DataFrame in HDF5 format"""
        try:
            with self._get_hdf5_file(symbol, 'a') as f:
                # Convert DataFrame to numpy arrays for better compression
                group_name = f"/{symbol}"
                
                if group_name in f:
                    del f[group_name]  # Replace existing data
                
                group = f.create_group(group_name)
                
                # Store index (dates)
                dates = df.index.astype(str).values
                group.create_dataset('dates', data=dates, compression=compression)
                
                # Store each column
                for col in df.columns:
                    data = df[col].values.astype(np.float64)
                    group.create_dataset(col, data=data, compression=compression)
                
                # Store metadata
                group.attrs['start_date'] = df.index[0].isoformat()
                group.attrs['end_date'] = df.index[-1].isoformat()
                group.attrs['record_count'] = len(df)
                group.attrs['columns'] = ','.join(df.columns)
                
                # Calculate compression ratio
                uncompressed_size = df.memory_usage(deep=True).sum()
                file_size = os.path.getsize(self._get_file_path(symbol))
                compression_ratio = uncompressed_size / file_size if file_size > 0 else 1.0
                
                group.attrs['compression_ratio'] = compression_ratio
                
                logger.debug(f"Stored {symbol}: {len(df)} records, compression: {compression_ratio:.2f}x")
                return compression_ratio
                
        except Exception as e:
            logger.error(f"Failed to store {symbol} in HDF5: {e}")
            raise
    
    def load_data(self, symbol: str, start_date: datetime = None, 
                  end_date: datetime = None) -> Optional[pd.DataFrame]:
        """Load DataFrame from HDF5"""
        try:
            with self._get_hdf5_file(symbol, 'r') as f:
                group_name = f"/{symbol}"
                
                if group_name not in f:
                    return None
                
                group = f[group_name]
                
                # Load dates
                dates = pd.to_datetime(group['dates'][:].astype(str))
                
                # Load data columns
                data = {}
                for key in group.keys():
                    if key != 'dates':
                        data[key] = group[key][:]
                
                df = pd.DataFrame(data, index=dates)
                
                # Filter by date range if specified
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                logger.debug(f"Loaded {symbol}: {len(df)} records")
                return df
                
        except Exception as e:
            logger.error(f"Failed to load {symbol} from HDF5: {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols"""
        symbols = []
        
        for filename in os.listdir(self.data_dir):
            if filename.startswith('market_data_') and filename.endswith('.h5'):
                try:
                    file_path = os.path.join(self.data_dir, filename)
                    with h5py.File(file_path, 'r') as f:
                        symbols.extend([key.strip('/') for key in f.keys()])
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")
        
        return sorted(symbols)
    
    def delete_symbol_data(self, symbol: str) -> bool:
        """Delete data for a symbol"""
        try:
            with self._get_hdf5_file(symbol, 'a') as f:
                group_name = f"/{symbol}"
                if group_name in f:
                    del f[group_name]
                    logger.info(f"Deleted data for {symbol}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete {symbol}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'symbols_count': 0,
            'files_info': []
        }
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.h5'):
                file_path = os.path.join(self.data_dir, filename)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                try:
                    with h5py.File(file_path, 'r') as f:
                        symbols_in_file = len(f.keys())
                        
                        stats['files_info'].append({
                            'filename': filename,
                            'size_mb': round(file_size, 2),
                            'symbols_count': symbols_in_file
                        })
                        
                        stats['symbols_count'] += symbols_in_file
                        
                except Exception as e:
                    logger.error(f"Error reading stats for {filename}: {e}")
                
                stats['total_files'] += 1
                stats['total_size_mb'] += file_size
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        return stats

class MarketDataStore:
    """Main market data storage manager"""
    
    def __init__(self, db_path: str = "data/market_data.db", hdf5_dir: str = "data/hdf5"):
        self.sqlite_manager = SQLiteManager(db_path)
        self.hdf5_manager = HDF5Manager(hdf5_dir)
        
        # Cache for frequently accessed data
        self.memory_cache = {}
        self.cache_max_size = 100  # Max symbols in memory cache
        self.cache_ttl_hours = 1   # Cache TTL in hours
        
    def store_symbol_data(self, symbol: str, df: pd.DataFrame, 
                         data_source: str = "unknown") -> bool:
        """Store market data for a symbol"""
        try:
            # Store in HDF5
            compression_ratio = self.hdf5_manager.store_data(symbol, df)
            
            # Update SQLite index
            self.sqlite_manager.update_index(symbol, df)
            
            # Store metadata
            file_size = os.path.getsize(self.hdf5_manager._get_file_path(symbol)) / (1024 * 1024)
            metadata = DataMetadata(
                symbol=symbol,
                start_date=df.index[0].to_pydatetime(),
                end_date=df.index[-1].to_pydatetime(),
                record_count=len(df),
                columns=list(df.columns),
                file_size_mb=file_size,
                last_updated=datetime.now(),
                data_source=data_source,
                compression_ratio=compression_ratio
            )
            
            self.sqlite_manager.store_metadata(metadata)
            
            # Update memory cache
            cache_key = f"{symbol}_{datetime.now().date()}"
            self.memory_cache[cache_key] = {
                'data': df.copy(),
                'timestamp': datetime.now()
            }
            
            # Clean cache if too large
            self._clean_memory_cache()
            
            logger.info(f"Stored {symbol}: {len(df)} records, {compression_ratio:.2f}x compression")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store {symbol}: {e}")
            return False
    
    def load_symbol_data(self, symbol: str, start_date: datetime = None, 
                        end_date: datetime = None, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Load market data for a symbol"""
        
        # Check memory cache first
        if use_cache:
            cache_key = f"{symbol}_{datetime.now().date()}"
            if cache_key in self.memory_cache:
                cached_item = self.memory_cache[cache_key]
                if datetime.now() - cached_item['timestamp'] < timedelta(hours=self.cache_ttl_hours):
                    df = cached_item['data']
                    # Apply date filtering
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]
                    return df
        
        # Load from HDF5
        df = self.hdf5_manager.load_data(symbol, start_date, end_date)
        
        if df is not None and use_cache:
            # Cache the result
            cache_key = f"{symbol}_{datetime.now().date()}"
            self.memory_cache[cache_key] = {
                'data': df.copy(),
                'timestamp': datetime.now()
            }
            self._clean_memory_cache()
        
        return df
    
    def get_symbol_metadata(self, symbol: str = None) -> Union[DataMetadata, List[DataMetadata]]:
        """Get symbol metadata"""
        return self.sqlite_manager.get_metadata(symbol)
    
    def search_symbols(self, criteria: Dict[str, Any]) -> List[str]:
        """Search symbols based on criteria"""
        return self.sqlite_manager.search_symbols(criteria)
    
    def get_available_symbols(self) -> List[str]:
        """Get all available symbols"""
        return self.hdf5_manager.get_available_symbols()
    
    def delete_symbol_data(self, symbol: str) -> bool:
        """Delete all data for a symbol"""
        success = True
        
        # Delete from HDF5
        if not self.hdf5_manager.delete_symbol_data(symbol):
            success = False
        
        # Delete from SQLite
        try:
            with self.sqlite_manager.get_connection() as conn:
                conn.execute("DELETE FROM data_metadata WHERE symbol = ?", (symbol,))
                conn.execute("DELETE FROM data_index WHERE symbol = ?", (symbol,))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to delete {symbol} from SQLite: {e}")
            success = False
        
        # Remove from cache
        cache_keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(f"{symbol}_")]
        for key in cache_keys_to_remove:
            del self.memory_cache[key]
        
        return success
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        hdf5_stats = self.hdf5_manager.get_storage_stats()
        
        # Get SQLite stats
        sqlite_size = os.path.getsize(self.sqlite_manager.db_path) / (1024 * 1024)  # MB
        
        # Memory cache stats
        cache_size_mb = sum(
            item['data'].memory_usage(deep=True).sum() for item in self.memory_cache.values()
        ) / (1024 * 1024)
        
        return {
            'hdf5': hdf5_stats,
            'sqlite': {
                'size_mb': round(sqlite_size, 2),
                'metadata_records': len(self.sqlite_manager.get_metadata() or [])
            },
            'memory_cache': {
                'symbols_cached': len(self.memory_cache),
                'size_mb': round(cache_size_mb, 2),
                'max_size': self.cache_max_size
            },
            'total_storage_mb': round(hdf5_stats['total_size_mb'] + sqlite_size + cache_size_mb, 2)
        }
    
    def _clean_memory_cache(self):
        """Clean memory cache to maintain size limits"""
        if len(self.memory_cache) > self.cache_max_size:
            # Remove oldest entries
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            items_to_remove = len(self.memory_cache) - self.cache_max_size
            for key, _ in sorted_items[:items_to_remove]:
                del self.memory_cache[key]
        
        # Remove expired entries
        now = datetime.now()
        expired_keys = [
            key for key, item in self.memory_cache.items()
            if now - item['timestamp'] > timedelta(hours=self.cache_ttl_hours)
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]

# Global market data store instance
market_data_store = MarketDataStore()