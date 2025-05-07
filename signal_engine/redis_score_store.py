import redis
import json
from datetime import date

class RedisScoreStore:
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.Redis(host=host, port=port, db=db)

    def save_score(self, score_dict: dict, date_key: str = None):
        if not date_key:
            date_key = date.today().isoformat()
        key = f"score:{date_key}"
        self.r.set(key, json.dumps(score_dict))

    def load_score(self, date_key: str = None):
        if not date_key:
            date_key = date.today().isoformat()
        key = f"score:{date_key}"
        data = self.r.get(key)
        return json.loads(data) if data else None
