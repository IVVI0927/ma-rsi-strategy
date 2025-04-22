import pandas as pd
import os

data_dir = "data"
standard_cols = {"date", "close"}  # 可以扩展支持 open, high, low, volume 等

def standardize_file(file_path):
    df = pd.read_csv(file_path)
    original_cols = list(df.columns)
    lower_cols = [col.lower().strip() for col in original_cols]
    df.columns = lower_cols

    missing = standard_cols - set(lower_cols)
    if missing:
        print(f"⚠️ {os.path.basename(file_path)} 缺少列: {missing}")
        return

    df.to_csv(file_path, index=False)
    print(f"✅ 已标准化列名: {os.path.basename(file_path)} → {lower_cols}")

if __name__ == "__main__":
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and not f.startswith("hs300")]
    for file in files:
        path = os.path.join(data_dir, file)
        standardize_file(path)