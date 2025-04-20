from jqdatasdk import auth, get_fundamentals, query, valuation

# 1) 聚宽登录
auth("13646502365", "yzx0927zufeI")

def get_pe_pb(code, date="2025-01-16"):
    """
    使用聚宽 API 获取 PE、PB、总市值（单位：亿元）。

    Args:
        code (str): 股票代码，格式 "600519.SH" 或 "000001.SZ"
        date (str): 交易日，格式 "YYYY-MM-DD"

    Returns:
        dict: {'pe_ttm': ..., 'pb': ..., 'market_cap': ...}
    """
    # 聚宽需要 .XSHG/.XSHE 格式
    jq_code = code.replace(".SH", ".XSHG").replace(".SZ", ".XSHE")
    try:
        q = query(
            valuation.pe_ratio,     # 市盈率
            valuation.pb_ratio,     # 市净率
            valuation.market_cap    # 总市值（元）
        ).filter(valuation.code == jq_code)

        df = get_fundamentals(q, date=date)
        if df.empty:
            return {"pe_ttm": None, "pb": None, "market_cap": None}

        row = df.iloc[0]
        pe = float(row["pe_ratio"])
        pb = float(row["pb_ratio"])
        # 将总市值从“万元”转为“亿元”
        market_cap = float(row["market_cap"]) / 1e4

        return {"pe_ttm": pe, "pb": pb, "market_cap": market_cap}
    except Exception as e:
        print(f"❌ get_pe_pb error for {code}: {e}")
        return {"pe_ttm": None, "pb": None, "market_cap": None}
    

'''#import akshare as ak
# import tushare as ts

# 1) Set your token once when the module loads
ts.set_token("b336f2995fbae151cab446279b39cbb5e5dcb5298a9b690afec2cc7f")
# 2) Create the pro API client
pro = ts.pro_api()

def get_pe_pb(code, trade_date="20250116"):
    """
    用 Tushare Pro 拿最新一日的 PE、PB、总市值（单位：亿元）。
    code: '600519.SH' or '000001.SZ'
    trade_date: 最近可用交易日，格式 YYYYMMDD
    """
    try:
        # Tushare 要用 "SH" / "SZ" 前缀
        ts_code = code
        # daily_basic 包含 pe、pb、total_mv(单位：万元)
        df = pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
        if df.empty:
            return {"pe_ttm": None, "pb": None, "market_cap": None}
        row = df.iloc[0]
        pe = float(row["pe"])
        pb = float(row["pb"])
        # total_mv 单位：万元，转为 亿元
        market_cap = float(row["total_mv"]) / 10000
        return {"pe_ttm": pe, "pb": pb, "market_cap": market_cap}
    except Exception as e:
        print(f"❌ get_pe_pb error for {code}: {e}")
        return {"pe_ttm": None, "pb": None, "market_cap": None}

def get_pe_pb(code):
    """
    Fetch PE, PB, and total market cap (in 亿元) for a given A-share code.
      - code: e.g. "600519.SH" or "000001.SZ"
    """
    # 1) 去掉后缀
    raw_code = code.split(".")[0]  

    try:
        # 2) 基本面指标：市盈率、市净率
        df = ak.stock_a_indicator_lg()
        row = df[df["代码"] == raw_code]
        if row.empty:
            pe_ttm = None
            pb = None
        else:
            pe_ttm = float(row["市盈率(TTM)"].values[0])
            pb = float(row["市净率"].values[0])

        # 3) 总市值，从实时行情接口拿
        spot = ak.stock_zh_a_spot()
        row2 = spot[spot["代码"] == raw_code]
        if row2.empty:
            market_cap = None
        else:
            # 注意单位是“亿元”或者“万元”，这里假设就是亿元
            market_cap = float(row2["总市值"].values[0])

        return {
            "pe_ttm": pe_ttm,
            "pb": pb,
            "market_cap": market_cap
        }
    except Exception:
        return {"pe_ttm": None, "pb": None, "market_cap": None}'''