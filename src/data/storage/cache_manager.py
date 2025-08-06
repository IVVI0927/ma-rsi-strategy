"""LRU Cache manager for data optimization"""

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import pickle
import os
import hashlib
import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class LRUCacheManager:
    """Advanced LRU cache with persistence and TTL support"""
    
    def __init__(self, maxsize: int = 1000, cache_dir: str = "cache", ttl_hours: int = 24):
        self.maxsize = maxsize
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        self.memory_cache = {}
        self.access_times = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired"""
        return time.time() - timestamp > self.ttl_seconds
    
    def _get_cache_file(self, key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def _cleanup_old_entries(self):
        """Remove expired and least recently used entries"""
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if self._is_expired(access_time)
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
        
        # Remove LRU entries if over maxsize
        if len(self.memory_cache) > self.maxsize:
            # Sort by access time and remove oldest
            sorted_keys = sorted(
                self.access_times.items(),
                key=lambda x: x[1]
            )
            
            remove_count = len(self.memory_cache) - self.maxsize
            for key, _ in sorted_keys[:remove_count]:
                self._remove_entry(key)
    
    def _remove_entry(self, key: str):
        """Remove cache entry from both memory and disk"""
        self.memory_cache.pop(key, None)
        self.access_times.pop(key, None)
        
        cache_file = self._get_cache_file(key)
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except OSError:
                pass
    
    def get(self, key: str, *args, **kwargs) -> Optional[Any]:
        """Get value from cache"""
        cache_key = key if isinstance(key, str) else self._generate_key(key, *args, **kwargs)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            if not self._is_expired(self.access_times[cache_key]):
                self.access_times[cache_key] = time.time()  # Update access time
                return self.memory_cache[cache_key]
            else:
                self._remove_entry(cache_key)
        
        # Check disk cache
        cache_file = self._get_cache_file(cache_key)
        if os.path.exists(cache_file):
            try:
                file_mtime = os.path.getmtime(cache_file)
                if not self._is_expired(file_mtime):
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Load to memory cache
                    self.memory_cache[cache_key] = data
                    self.access_times[cache_key] = time.time()
                    return data
                else:
                    os.remove(cache_file)
            except Exception as e:
                logger.error(f"Error loading cache file {cache_file}: {e}")
        
        return None
    
    def set(self, key: str, value: Any, *args, **kwargs):
        """Set value in cache"""
        cache_key = key if isinstance(key, str) else self._generate_key(key, *args, **kwargs)
        
        # Cleanup if needed
        self._cleanup_old_entries()
        
        # Set in memory cache
        self.memory_cache[cache_key] = value
        self.access_times[cache_key] = time.time()
        
        # Persist to disk for DataFrames and large objects
        if isinstance(value, (pd.DataFrame, dict, list)) and len(str(value)) > 1000:
            cache_file = self._get_cache_file(cache_key)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                logger.error(f"Error saving cache file {cache_file}: {e}")
    
    def invalidate(self, key: str, *args, **kwargs):
        """Remove specific key from cache"""
        cache_key = key if isinstance(key, str) else self._generate_key(key, *args, **kwargs)
        self._remove_entry(cache_key)
    
    def clear(self):
        """Clear all cache"""
        self.memory_cache.clear()
        self.access_times.clear()
        
        # Remove cache files
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
        except OSError:
            pass
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'memory_entries': len(self.memory_cache),
            'max_size': self.maxsize,
            'ttl_hours': self.ttl_seconds / 3600,
            'cache_dir': self.cache_dir,
            'disk_files': len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')])
        }

# Global cache instance
cache_manager = LRUCacheManager(maxsize=1000, ttl_hours=24)

def cached_function(ttl_hours: int = 24):
    """Decorator for caching function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            func_name = f"{func.__module__}.{func.__name__}"
            cache_key = cache_manager._generate_key(func_name, *args, **kwargs)
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func_name}")
                return result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func_name}")
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result)
            
            return result
        
        # Add cache control methods
        wrapper.cache_clear = lambda: cache_manager.clear()
        wrapper.cache_info = lambda: cache_manager.stats()
        
        return wrapper
    return decorator

@cached_function(ttl_hours=1)
def cached_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Example cached function for stock data"""
    # This would be implemented by the data fetcher
    pass