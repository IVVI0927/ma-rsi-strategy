import os
from signal_engine.score_and_suggest import score_stock

def get_top_stocks(top_n=10):
    results = []
    for f in os.listdir("data"):
        if not f.endswith(".csv"):
            continue
        code = f.replace(".csv", "")
        info = score_stock(code)            # info 必须是 dict，包含下面几项
        info.setdefault("suggest", info.get("signal", ""))  # 如果你用 signal 字段，也同步到 suggest
        results.append(info)
    # 按 score 排序
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results[:top_n]