import requests

DEEPSEEK_API_KEY = "sk-03d4934a5587424da890b891721ebbfd"

def call_ai_model(factor_info: dict) -> dict:
    prompt = f"""你是一个智能股票分析顾问。根据以下 A 股因子数据，给出一个 0-100 分的评分，并说明理由：
{factor_info}
请返回一个标准 JSON 格式：
{{"score": int, "reason": str}}"""

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "deepseek-chat",  # 根据平台的模型名称填写
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(url, json=body, headers=headers)
    result = response.json()
    content = result['choices'][0]['message']['content']

    try:
        return eval(content)  # ⚠️建议替换为 json.loads(content)
    except:
        return {"score": 50, "reason": "解析失败，请检查AI输出"}