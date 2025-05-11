import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import logging
from signal_engine.score_and_suggest import score_stock
import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # 获取所有A股代码
    all_codes = [
        f.replace(".csv", "")
        for f in os.listdir("data")
        if f.endswith(".csv") and (".SH" in f or ".SZ" in f)
    ]
    
    # 对每只股票进行评分
    results = []
    for code in all_codes:
        try:
            result = score_stock(code)
            if result:  # 只添加非空结果
                results.append(result)
                logging.info(f"Scored {code}: {result['score']}")
        except Exception as e:
            logging.error(f"Error processing {code}: {e}")
    
    # 转换为DataFrame并保存
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="score", ascending=False)
        df.to_csv('today_recommendations.csv', index=False)
        logging.info("Results saved to today_recommendations.csv")
        
        # 打印结果
        print("\nStock Scores:")
        print(df[['code', 'score', 'suggest']])

if __name__ == "__main__":
    main()
