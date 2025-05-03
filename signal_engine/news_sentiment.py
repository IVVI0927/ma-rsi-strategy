import requests
from bs4 import BeautifulSoup

# 情绪关键词词典（继续扩展）
POSITIVE_WORDS = ["看好", "利好", "涨停", "创新高", "加仓", "布局", "强势"]
NEGATIVE_WORDS = ["利空", "割肉", "亏", "大跌", "跳水", "崩了", "出货"]

def fetch_guba_titles(stock_code: str, count: int = 10) -> list[str]:
    """
    爬取东方财富股吧的热帖标题
    stock_code 示例：'sz300750'（宁德时代）
    """
    url = f"https://guba.eastmoney.com/list,{stock_code}.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        titles = [tag.text.strip() for tag in soup.select(".note-list .note")][:count]
        return titles
    except Exception as e:
        print(f"❌ 获取股吧失败：{e}")
        return []

def analyze_sentiment(titles: list[str]) -> str:
    """
    简单关键词匹配，返回情绪标签：'positive' / 'neutral' / 'negative'
    """
    score = 0
    for title in titles:
        if any(word in title for word in POSITIVE_WORDS):
            score += 1
        if any(word in title for word in NEGATIVE_WORDS):
            score -= 1
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

def get_sentiment(stock_code: str) -> str:
    titles = fetch_guba_titles(stock_code)
    sentiment = analyze_sentiment(titles)
    return sentiment

# 示例
if __name__ == "__main__":
    code = "sz300750"  # 可替换
    sentiment = get_sentiment(code)
    print(f"🎯 股票 {code} 的情绪判断结果：{sentiment}")