# signal_engine/ai_score_stock.py

def ai_score_stock(features: dict) -> float:
    """
    模拟 AI 模型打分函数（可后续替换为真实模型）
    输入是一组特征，如 RSI, MA差值, PE, PB等
    返回一个 0~1 的评分值
    """
    rsi       = features.get("rsi", 50)
    ma_diff   = features.get("ma_diff", 0)  # MA_short - MA_long
    pe        = features.get("pe", 10)
    pb        = features.get("pb", 1)
    volatility = features.get("volatility", 0.02)

    # 简化版模型：可以看作是你未来ML模型的线性近似
    score = (
        0.3 * (rsi / 100) +
        0.3 * ma_diff -
        0.2 * (pe / 100) -
        0.1 * pb +
        0.1 * (1 - volatility)
    )

    # 限制在 0~1 范围内
    return max(0, min(score, 1))