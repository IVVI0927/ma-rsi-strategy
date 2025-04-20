import tushare as ts
import pandas as pd
import os
import time

# === Step 1: 设置你的 Tushare token
ts.set_token("b336f2995fbae151cab446279b39cbb5e5dcb5298a9b690afec2cc7f")  
pro = ts.pro_api()

def get_hs300_codes(limit=50):
    """获取沪深300成分股代码（前limit个）"""
    df = pro.index_weight(index_code='000300.SH', trade_date='20230401')
    codes = df['con_code'].unique().tolist()
    return codes[:limit]

def download_single_stock(code, start_date="20230401", end_date="20250401"):
    """下载某支股票的日线数据"""
    try:
        df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
        if df.empty:
            print(f"⚠️ No data for {code}")
            return
        os.makedirs("data", exist_ok=True)
        df.to_csv(f"data/{code}.csv", index=False)
        print(f"✅ Saved: {code}")
    except Exception as e:
        print(f"❌ Error downloading {code}: {e}")

def download_hs300_batch(n=50):
    codes = get_hs300_codes(limit=n)
    for code in codes:
        download_single_stock(code)
        time.sleep(1.2)  # 防止频率限制

if __name__ == "__main__":
    download_hs300_batch(n=50)