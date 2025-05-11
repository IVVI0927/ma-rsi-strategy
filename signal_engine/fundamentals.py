import akshare as ak
import logging
from typing import Dict, Optional

def get_fundamentals(code: str) -> Dict[str, Optional[float]]:
    """
    获取A股基本面数据
    
    Args:
        code (str): 股票代码，例如 '600519.SH' 或 '000001.SZ'
        
    Returns:
        dict: 包含以下字段的字典：
            - pe_ttm: 市盈率(TTM)
            - pb: 市净率
            - market_cap: 市值（亿元）
            - dividend_yield: 股息率（百分比）
            - beta: 贝塔系数
            - sector: 行业板块
            - industry: 具体行业
    """
    try:
        # 去掉后缀
        raw_code = code.split(".")[0]
        
        # 获取基本面指标
        df = ak.stock_a_indicator_lg()
        logging.info(f"Available columns in indicator data: {df.columns.tolist()}")
        
        # 尝试不同的可能的列名
        code_column = None
        for possible_code_column in ['代码', 'symbol', 'stock_code', 'code']:
            if possible_code_column in df.columns:
                code_column = possible_code_column
                break
                
        if code_column is None:
            logging.error(f"No code column found in indicator data. Available columns: {df.columns.tolist()}")
            return {
                "pe_ttm": None,
                "pb": None,
                "market_cap": None,
                "dividend_yield": None,
                "beta": None,
                "sector": None,
                "industry": None
            }
            
        row = df[df[code_column] == raw_code]
        
        # 获取实时行情
        spot = ak.stock_zh_a_spot()
        logging.info(f"Available columns in spot data: {spot.columns.tolist()}")
        
        # 尝试不同的可能的列名
        spot_code_column = None
        for possible_code_column in ['代码', 'symbol', 'stock_code', 'code']:
            if possible_code_column in spot.columns:
                spot_code_column = possible_code_column
                break
                
        if spot_code_column is None:
            logging.error(f"No code column found in spot data. Available columns: {spot.columns.tolist()}")
            return {
                "pe_ttm": None,
                "pb": None,
                "market_cap": None,
                "dividend_yield": None,
                "beta": None,
                "sector": None,
                "industry": None
            }
            
        row2 = spot[spot[spot_code_column] == raw_code]
        
        if row.empty or row2.empty:
            logging.warning(f"No data found for {code}")
            return {
                "pe_ttm": None,
                "pb": None,
                "market_cap": None,
                "dividend_yield": None,
                "beta": None,
                "sector": None,
                "industry": None
            }
            
        # 尝试不同的可能的列名
        pe_column = None
        for possible_pe_column in ['市盈率(TTM)', 'pe_ttm', 'pe', '市盈率']:
            if possible_pe_column in row.columns:
                pe_column = possible_pe_column
                break
                
        pb_column = None
        for possible_pb_column in ['市净率', 'pb', 'pb_ratio']:
            if possible_pb_column in row.columns:
                pb_column = possible_pb_column
                break
                
        market_cap_column = None
        for possible_mc_column in ['总市值', 'market_cap', 'total_mv']:
            if possible_mc_column in row2.columns:
                market_cap_column = possible_mc_column
                break
                
        result = {
            "pe_ttm": float(row[pe_column].values[0]) if pe_column and not row.empty else None,
            "pb": float(row[pb_column].values[0]) if pb_column and not row.empty else None,
            "market_cap": float(row2[market_cap_column].values[0]) if market_cap_column and not row2.empty else None,
            "dividend_yield": None,  # 暂时不获取股息率
            "beta": None,  # 暂时不获取贝塔系数
            "sector": None,  # 暂时不获取行业板块
            "industry": None  # 暂时不获取具体行业
        }
        
        logging.info(f"Successfully got fundamentals for {code}: {result}")
        return result
        
    except Exception as e:
        logging.error(f"Error getting fundamentals for {code}: {str(e)}")
        return {
            "pe_ttm": None,
            "pb": None,
            "market_cap": None,
            "dividend_yield": None,
            "beta": None,
            "sector": None,
            "industry": None
        }

def get_roe(code: str) -> Optional[float]:
    """
    获取股本回报率(ROE)
    
    Args:
        code (str): 股票代码，例如 '600519.SH'
        
    Returns:
        float: ROE值（百分比），如果获取失败则返回None
    """
    try:
        raw_code = code.split(".")[0]
        df = ak.stock_a_indicator_lg()
        row = df[df["代码"] == raw_code]
        if row.empty:
            return None
        return float(row["净资产收益率"].values[0])
    except Exception as e:
        logging.error(f"Error getting ROE for {code}: {e}")
        return None

def get_profit_margin(code: str) -> Optional[float]:
    """
    获取净利润率
    
    Args:
        code (str): 股票代码，例如 '600519.SH'
        
    Returns:
        float: 净利润率（百分比），如果获取失败则返回None
    """
    try:
        raw_code = code.split(".")[0]
        df = ak.stock_a_indicator_lg()
        row = df[df["代码"] == raw_code]
        if row.empty:
            return None
        return float(row["净利率"].values[0])
    except Exception as e:
        logging.error(f"Error getting profit margin for {code}: {e}")
        return None

def get_debt_ratio(code: str) -> Optional[float]:
    """
    获取资产负债率
    
    Args:
        code (str): 股票代码，例如 '600519.SH'
        
    Returns:
        float: 资产负债率（百分比），如果获取失败则返回None
    """
    try:
        raw_code = code.split(".")[0]
        df = ak.stock_a_indicator_lg()
        row = df[df["代码"] == raw_code]
        if row.empty:
            return None
        return float(row["资产负债率"].values[0])
    except Exception as e:
        logging.error(f"Error getting debt ratio for {code}: {e}")
        return None

def get_profit_growth(code: str) -> Optional[float]:
    """
    获取利润增长率
    
    Args:
        code (str): 股票代码，例如 '600519.SH'
        
    Returns:
        float: 利润增长率（百分比），如果获取失败则返回None
    """
    try:
        raw_code = code.split(".")[0]
        df = ak.stock_a_indicator_lg()
        row = df[df["代码"] == raw_code]
        if row.empty:
            return None
        return float(row["净利润增长率"].values[0])
    except Exception as e:
        logging.error(f"Error getting profit growth for {code}: {e}")
        return None

def get_ps(code: str) -> Optional[float]:
    """
    获取市销率(PS)
    
    Args:
        code (str): 股票代码，例如 '600519.SH'
        
    Returns:
        float: 市销率，如果获取失败则返回None
    """
    try:
        raw_code = code.split(".")[0]
        df = ak.stock_a_indicator_lg()
        row = df[df["代码"] == raw_code]
        if row.empty:
            return None
        return float(row["市销率"].values[0])
    except Exception as e:
        logging.error(f"Error getting PS for {code}: {e}")
        return None

def get_turnover_rate(code: str) -> Optional[float]:
    """
    获取换手率
    
    Args:
        code (str): 股票代码，例如 '600519.SH'
        
    Returns:
        float: 换手率（百分比），如果获取失败则返回None
    """
    try:
        raw_code = code.split(".")[0]
        spot = ak.stock_zh_a_spot()
        row = spot[spot["代码"] == raw_code]
        if row.empty:
            return None
        return float(row["换手率"].values[0])
    except Exception as e:
        logging.error(f"Error getting turnover rate for {code}: {e}")
        return None

def get_volume_change(code: str, window: int = 5) -> Optional[float]:
    """
    获取成交量变化率
    
    Args:
        code (str): 股票代码，例如 '600519.SH'
        window (int): 计算窗口大小，默认5天
        
    Returns:
        float: 成交量变化率（百分比），如果获取失败则返回None
    """
    try:
        raw_code = code.split(".")[0]
        # 获取历史行情数据
        hist = ak.stock_zh_a_hist(symbol=raw_code, period="daily", 
                                 start_date=None, end_date=None, adjust="qfq")
        if len(hist) < window + 1:
            return None
            
        prev_avg = hist["成交量"].iloc[-(window+1):-1].mean()
        curr = hist["成交量"].iloc[-1]
        return ((curr - prev_avg) / prev_avg * 100) if prev_avg != 0 else None
    except Exception as e:
        logging.error(f"Error getting volume change for {code}: {e}")
        return None