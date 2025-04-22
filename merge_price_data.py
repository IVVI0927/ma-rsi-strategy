import pandas as pd
import os

data_dir = "data"
all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and not f.startswith("hs300")]

merged = []

for file in all_files:
    code = file.replace(".csv", "")
    path = os.path.join(data_dir, file)

    try:
        df = pd.read_csv(path, header=None)

        # 尝试识别数据结构并赋予列名
        if df.shape[1] >= 6:
            df = df.iloc[:, :6]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
        elif df.shape[1] == 2:
            df.columns = ["date", "close"]
        else:
            print(f"⚠️ 无法识别列结构：{file}（列数: {df.shape[1]}）")
            continue

        # 确保日期列为日期格式
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"])  # 至少需要这两列有效

        df["code"] = code
        merged.append(df[["date", "code", "close"]])

    except Exception as e:
        print(f"❌ 处理文件 {file} 失败: {e}")

final_df = pd.concat(merged)
final_df = final_df.sort_values(by=["date", "code"]).reset_index(drop=True)
final_df.to_csv("data/hs300_daily_2025_01.csv", index=False)
print("✅ 合并完成，输出文件：data/hs300_daily_2025_01.csv")