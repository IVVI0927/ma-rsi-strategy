import requests
import re
import json

DEEPSEEK_API_KEY = "sk-969b8f7d7448431cab9bbecd2569d83d"

def call_ai_model(factor_info: dict) -> dict:
    # 示例情绪（后续爬虫/API动态）
    simulated_sentiment = "positive"  # negative / neutral
    factor_info["news_sentiment"] = simulated_sentiment

    prompt = f"""你是一个智能股票分析顾问。根据以下 A 股因子数据，给出一个 0-100 分的评分，并说明理由。
注意：情绪因子值越正代表市场情绪越积极，越负代表情绪低迷。请综合判断：
{factor_info}

请返回一个标准 JSON 格式：
{{"score": int, "reason": str}}"""

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(url, json=body, headers=headers)
    result = response.json()

    print("🔁 Raw API response:", result)

    if 'choices' not in result:
        return {
            "score": 50,
            "reason": f"LLM API错误响应：{result.get('error', '未知错误')}"
        }

    content = result['choices'][0]['message']['content']

    try:
        # 提取 markdown 格式中的 JSON 内容
        match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
        if match:
            clean_json = match.group(1)
            return json.loads(clean_json)
        else:
            return {"score": 50, "reason": f"解析失败，AI输出内容：{content}"}
    except Exception as e:
        return {"score": 50, "reason": f"解析异常：{str(e)}，原始内容：{content}"}