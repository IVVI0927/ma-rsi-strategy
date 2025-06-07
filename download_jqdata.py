from jqdatasdk import *
import pandas as pd
import os
import time

# Step 1. 登录手机号+密码
auth("13646502365", "yzx0927zufeI") 

def get_hs300_codes(limit: int = 50) -> list:
    all_codes = [
        "600519.XSHG", "000858.XSHE", "601318.XSHG", "300750.XSHE", "600036.XSHG",
        "000333.XSHE", "000001.XSHE", "002415.XSHE", "600104.XSHG", "000725.XSHE",
        "600030.XSHG", "600309.XSHG", "600048.XSHG", "000651.XSHE", "002714.XSHE",
        "600438.XSHG", "002352.XSHE", "600031.XSHG", "002236.XSHE", "600000.XSHG"
    ]
    return all_codes[:limit]

def download_single_stock(code: str, start_date: str = "2024-01-10", end_date: str = "2025-01-16") -> None:
    """下载某支股票的日线数据"""
    try:
        df = get_price(code, start_date=start_date, end_date=end_date, frequency='daily')
        if df.empty:
            print(f"⚠️ No data for {code}")
            return
        os.makedirs("data", exist_ok=True)
        code_clean = code.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")
        df.to_csv(f"data/{code_clean}.csv")
        print(f"✅ Saved: {code_clean}")
    except Exception as e:
        print(f"❌ Error downloading {code}: {e}")

def download_batch(n: int = 50) -> None:
    codes = get_hs300_codes(limit=n)
    for code in codes:
        download_single_stock(code)
        time.sleep(1.0)

if __name__ == "__main__":
    download_batch(n=50)